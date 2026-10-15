"""Rolling-origin evaluation, final fit and artifact saving."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from . import __version__
from . import features as F
from . import metrics as M

log = logging.getLogger(__name__)
OUT_COLS = ["house", "utc", "local_date", "hour", "y", "scale", *F.BASELINES]


METRIC = {"l1": "l1", "regression_l1": "l1", "mae": "l1", "tweedie": "tweedie", "poisson": "poisson"}


def lgb_params(mc: dict) -> dict:
    # early stopping uses the metric that matches the objective
    return {
        "objective": mc["objective"], "metric": mc.get("metric") or METRIC.get(mc["objective"], "l2"),
        "learning_rate": mc["learning_rate"],
        "num_leaves": mc["num_leaves"], "min_data_in_leaf": mc["min_data_in_leaf"],
        "feature_fraction": mc["feature_fraction"], "bagging_fraction": mc["bagging_fraction"],
        "bagging_freq": mc["bagging_freq"], "lambda_l2": mc["lambda_l2"], "seed": mc["seed"],
        "deterministic": True, "force_row_wise": True, "num_threads": mc.get("num_threads", 0),
        "verbosity": -1,
    }


def _dataset(d: pd.DataFrame, feats: list[str], cats: list[str], reference=None) -> lgb.Dataset:
    # Weighting by the scale turns L1 on the normalized target into absolute error in kWh.
    return lgb.Dataset(d[feats], label=d["y_n"].to_numpy(float), weight=d["scale"].to_numpy(float),
                       categorical_feature=list(cats) or "auto", reference=reference, free_raw_data=True)


def fit_booster(tr: pd.DataFrame, feats: list[str], cats: list[str], cfg: dict, rounds: int | None = None):
    """Early-stop on the last val_days of the window, then refit on the whole window."""
    mc = cfg["model"]
    params = lgb_params(mc)
    if rounds is None:
        rounds = int(mc["default_rounds"])
        cut = tr["local_date"].max() - pd.Timedelta(days=cfg["cv"]["val_days"] - 1)
        fit, val = tr[tr["local_date"] < cut], tr[tr["local_date"] >= cut]
        if len(val) >= 200 and len(fit) >= 1000:
            dfit = _dataset(fit, feats, cats)
            probe = lgb.train(params, dfit, num_boost_round=mc["max_rounds"],
                              valid_sets=[_dataset(val, feats, cats, reference=dfit)],
                              callbacks=[lgb.early_stopping(mc["early_stopping_rounds"], verbose=False)])
            rounds = max(int(probe.best_iteration or mc["max_rounds"]), 20)
    return lgb.train(params, _dataset(tr, feats, cats), num_boost_round=rounds), rounds


def predict_kwh(booster: lgb.Booster, d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return np.clip(booster.predict(d[feats]), 0, None) * d["scale"].to_numpy(float)


def make_folds(df: pd.DataFrame, cfg: dict) -> list[dict]:
    """One fold per test month. Each fold trains on everything before its month."""
    cv = cfg["cv"]
    dates = df.loc[df["eval_ok"], "local_date"]
    first = pd.Period(cv["first_test_month"], freq="M")
    last = pd.Period(cv["last_test_month"], freq="M") if cv.get("last_test_month") else dates.max().to_period("M")
    return [{"fold": i, "test_month": str(p), "test_start": str(p.start_time.date()),
             "test_end": str((p + 1).start_time.date())}
            for i, p in enumerate(pd.period_range(first, last, freq="M"))]


def run_training(cfg: dict, run_name: str | None = None, compare_local: bool = False,
                 cold_start: bool = False, extra_path: str | None = None) -> Path:
    proc = Path(cfg["paths"]["processed_dir"])
    load = pd.read_parquet(proc / "hourly_load.parquet")
    env_file = proc / "hourly_env.parquet"
    env = pd.read_parquet(env_file) if env_file.exists() else None
    extra = pd.read_parquet(extra_path) if extra_path else None
    static = F.load_static()
    levels = F.heating_levels(static)
    df, feats, cats = F.build_frame(load, env, static, cfg, extra, heating=levels)
    log.info("frame: %d hours, %d trainable, %d features", len(df), int(df["train_ok"].sum()), len(feats))

    run_dir = Path(cfg["paths"]["runs_dir"]) / (run_name or time.strftime("baseline_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(df, cfg)
    local_feats = [f for f in feats if f not in F.HOUSE_LEVEL]
    cold_feats = [f for f in feats if f != "house"]
    cold_cats = [c for c in cats if c != "house"]
    preds, rounds_used = [], []
    for f in folds:
        start, end = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        tr = df[df["train_ok"] & (df["local_date"] < start)]
        te = df[df["eval_ok"] & (df["local_date"] >= start) & (df["local_date"] < end)]
        if tr.empty or te["local_date"].nunique() < cfg["cv"].get("min_test_days", 7):
            log.warning("fold %s skipped (train %d hours, test %d days)",
                        f["test_month"], len(tr), te["local_date"].nunique())
            f["skipped"] = True
            continue
        booster, rounds = fit_booster(tr, feats, cats, cfg)
        rounds_used.append(rounds)
        f.update(rounds=rounds, train_hours=int(len(tr)), test_hours=int(len(te)))
        out = te[OUT_COLS].copy()
        out["test_month"] = f["test_month"]
        out["lgbm_global"] = predict_kwh(booster, te, feats)
        if compare_local:  # one model per house, same features minus house-level descriptors
            out["lgbm_local"] = np.nan
            for h, trh in tr.groupby("house"):
                m = (te["house"] == h).to_numpy()
                if m.any() and trh["local_date"].nunique() >= cfg["cv"]["min_local_train_days"]:
                    b, _ = fit_booster(trh, local_feats, [], cfg)
                    out.loc[m, "lgbm_local"] = predict_kwh(b, te.loc[m], local_feats)
        if cold_start:  # each test house is unseen in training, as a new user would be
            out["lgbm_coldstart"] = np.nan
            for h in te["house"].unique():
                b, _ = fit_booster(tr[tr["house"] != h], cold_feats, cold_cats, cfg, rounds=rounds)
                m = (te["house"] == h).to_numpy()
                out.loc[m, "lgbm_coldstart"] = predict_kwh(b, te.loc[m], cold_feats)
        preds.append(out)
        log.info("fold %s: train %d h, test %d h, %d rounds, WAPE %.3f (seasonal naive %.3f)",
                 f["test_month"], len(tr), len(te), rounds,
                 M.wape(out["y"], out["lgbm_global"]), M.wape(out["y"], out["seasonal_d7"]))
    if not preds:
        raise RuntimeError("No fold produced predictions. Check cv.first_test_month against the data range.")
    pred = pd.concat(preds, ignore_index=True)
    models = ["lgbm_global", *F.BASELINES]
    models += (["lgbm_local"] if compare_local else []) + (["lgbm_coldstart"] if cold_start else [])
    res = M.evaluate(pred, models)

    final_rounds = int(np.median(rounds_used))
    final, _ = fit_booster(df[df["train_ok"]], feats, cats, cfg, rounds=final_rounds)
    _save(run_dir, cfg, df, feats, cats, folds, pred, res, final, final_rounds, extra_path, models, levels)
    log.info("saved run to %s", run_dir)
    return run_dir


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def _save(run_dir, cfg, df, feats, cats, folds, pred, res, booster, rounds, extra_path, models, levels) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (run_dir / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    res["overall"].to_csv(run_dir / "metrics_overall.csv", float_format="%.4f")
    res["by_house"].to_csv(run_dir / "metrics_by_house.csv", float_format="%.4f")
    res["by_month"].to_csv(run_dir / "metrics_by_month.csv", float_format="%.4f")
    booster.save_model(str(run_dir / "model.txt"))
    pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance("gain"),
        "splits": booster.feature_importance("split"),
    }).sort_values("gain", ascending=False).to_csv(run_dir / "feature_importance.csv", index=False)
    meta = {
        "package_version": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": feats,
        "categorical": cats,
        "final_rounds": rounds,
        "models_evaluated": models,
        "evaluated_hours": res["n_common"],
        "target": "hourly kWh divided by the trailing mean daily load, loss weighted by that scale",
        "objective": cfg["model"]["objective"],
        "origin": "local midnight (Europe/Athens) before the target day",
        "houses": sorted(df["house"].unique().tolist()),
        "train_hours": int(df["train_ok"].sum()),
        "train_period_utc": [str(df.loc[df["train_ok"], "utc"].min()), str(df.loc[df["train_ok"], "utc"].max())],
        "heating_levels": levels,
        "data_sha1": _sha1(Path(cfg["paths"]["processed_dir"]) / "hourly_load.parquet"),
        "extra_features": Path(extra_path).name if extra_path else None,
        "versions": {"python": platform.python_version(), "lightgbm": lgb.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (run_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _plot(run_dir, res)


def _plot(run_dir: Path, res: dict) -> None:
    try:
        import matplotlib

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    bh = res["by_house"][[c for c in res["by_house"].columns if "[" not in c]]
    ax = bh.plot.bar(figsize=(11, 4), width=0.8)
    ax.set_ylabel("hourly WAPE")
    ax.set_xlabel("house")
    ax.set_title("Day-ahead hourly WAPE by house, rolling-origin test months")
    ax.legend(ncol=len(bh.columns), fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "wape_by_house.png", dpi=130)
    plt.close("all")
