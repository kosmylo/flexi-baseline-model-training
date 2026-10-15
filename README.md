# plegma-baseline

Day-ahead forecasting of hourly household electricity use, built on the [Plegma dataset](https://doi.org/10.15129/3b01a6c6-2efd-424a-b8b8-5fe7fa445ded) (13 homes in Attica, Greece).

This repository holds the baseline model for KPI1.1 of the FLEXI project. It is the reference against which the context-aware forecasts of the FLEXI energy assistant are measured.

## Highlights

- One LightGBM model serves every home. At local midnight it forecasts each hour of the next local day.
- Greek calendar inputs: public holidays, Orthodox Easter week, the August and Christmas holidays, and reduced-tariff hours.
- A reproducible pipeline from the public archive: clock check, hourly tables, rolling-origin evaluation and model export.
- A Python API that forecasts any household with an hourly meter series.
- A paired comparison for KPI1.1: the baseline against a model with assistant context, on identical test hours.

## Results on Plegma

Rolling-origin evaluation over the test months January to September 2023: 64,020 test hours across 13 homes.

| Forecast | Hourly WAPE | Daily WAPE | Bias | RMSE (kWh) |
|---|---|---|---|---|
| **Model, Tweedie objective (default)** | **0.763** | **0.350** | **-7.5%** | **0.507** |
| Model, L2 objective | 0.800 | 0.349 | +2.0% | 0.504 |
| Model, L1 objective (median) | 0.665 | 0.504 | -47.0% | 0.552 |
| Similar-day average | 0.813 | 0.381 | -0.4% | 0.541 |
| Same hour yesterday | 0.879 | 0.353 | +0.1% | 0.663 |
| Same hour last week | 0.953 | 0.495 | -0.8% | 0.691 |

- Summed over all homes, the error falls to 0.37 per hour and 0.16 per day.
- One shared model beats per-house models in all 13 homes. A model that never saw a home scores within about 1% of the shared model, so new households start with a working forecast.
- The Tweedie objective keeps energy totals close to reality and gives the best hourly score of the mean forecasts. The L1 objective scores best per hour because it forecasts the median hour, which leaves out much of the energy in short appliance bursts.
- For daily totals, the model matches "same as yesterday". Day-to-day changes come from household behaviour, which is the information the FLEXI assistant collects.

These development runs also used the Plegma house ID as an input. The released configuration leaves it out, which changed scores by about 1% in tests. Each exported model carries its own scores in `MODEL_CARD.md`.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/kosmylo/flexi-baseline-model-training plegma-baseline
cd plegma-baseline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

1. Download `PlegmaDataset_Clean.7z` (about 290 MB) in a browser from the [dataset page](https://doi.org/10.15129/3b01a6c6-2efd-424a-b8b8-5fe7fa445ded).
2. Run the pipeline from the repository folder:

```bash
python -m plegma_baseline download --archive /path/to/PlegmaDataset_Clean.7z
python -m plegma_baseline prepare
python -m plegma_baseline.diagnose
python -m plegma_baseline train --run-name final_tweedie
python -m plegma_baseline export --run runs/final_tweedie --name plegma-tweedie-v1
```

`download` without `--archive` fetches the file directly when the repository allows scripted downloads. It also finds the archive in `data/raw/`, the current folder or `~/Downloads`.

## Using the saved model

```python
import pandas as pd
from plegma_baseline import Forecaster

model = Forecaster.load("models/plegma-tweedie-v1")

load = pd.read_parquet("data/processed/hourly_load.parquet")      # house, utc, load_kwh
weather = pd.read_parquet("data/processed/hourly_env.parquet")    # house, utc, ext_temp, ext_hum

forecast = model.forecast(load, day="2023-09-20", weather=weather)
print(forecast.head(24))
```

The result has one row per home and local hour:

| Column | Content |
|---|---|
| `house` | household ID, as in the input |
| `local_time` | hour start on the Athens clock |
| `utc` | hour start in UTC |
| `forecast_kwh` | forecast energy in that hour |
| `similar_day`, `seasonal_d7` | reference forecasts, for comparison |

### Inputs

| Argument | Columns | Notes |
|---|---|---|
| `load` | `house`, `utc`, `load_kwh` | Hourly energy per home, one row per hour. At least 7 full days before the target day, 28 work best. |
| `weather` | `utc`, `ext_temp`, `ext_hum`, optional `house` | Outdoor temperature (°C) and relative humidity (%). Without `house`, one series serves every home. Target-day hours without values take the previous day's values. |
| `households` | `house` plus any of `occupants`, `rooms`, `built_year`, `dwelling`, `three_phase`, `heating`, `n_ac`, `has_boiler`, `has_dishwasher` | Optional. Partial descriptors are fine. The Plegma descriptors are the default. |
| `extra` | `house`, `utc` and extra feature columns | For models trained with `--extra-features`. |

`utc` takes UTC timestamps, with or without a time zone. `load_kwh` is the energy used in that hour, which equals the mean power in kW.

### Your own household

```python
own = pd.DataFrame({"house": "my-home", "utc": hours_utc, "load_kwh": hourly_kwh})
city_weather = pd.DataFrame({"utc": hours_utc_with_target_day, "ext_temp": temp_c, "ext_hum": rh_pct})
home = pd.DataFrame([{"house": "my-home", "occupants": 2, "rooms": 3, "n_ac": 1, "has_boiler": 1}])

forecast = model.forecast(own, day="2026-11-03", weather=city_weather, households=home)
```

`examples/forecast_example.py` runs both cases on the prepared Plegma tables. From the command line:

```bash
python -m plegma_baseline predict --run models/plegma-tweedie-v1 --date 2023-09-20 --house 3
```

## How the model works

- **Target**: hourly energy of the whole home, divided by the home's mean load over the previous 28 days. One model then serves small and large homes alike.
- **Origin**: local midnight before the target day. Every input comes from earlier days. Outdoor weather takes the recorded values as a stand-in for a perfect day-ahead forecast.
- **Inputs** (37 in the default configuration):
  - load at the same local hour 1, 2, 3, 7 and 14 days earlier, plus the 7-day mean and spread of that hour
  - previous-day mean, maximum and minimum, and the previous-week mean
  - calendar: hour, weekday, day off, Greek public holidays, Orthodox Easter week, August and Christmas periods, day of year
  - reduced-tariff hours: the classic night tariff until January 2025 and the two-zone tariff from February 2025
  - outdoor temperature and humidity at the hour, plus daily mean, maximum and minimum
  - household descriptors from Tables 2 and 3 of the data descriptor (`plegma_baseline/data/houses.csv`)
- **Objective**: Tweedie, which forecasts the mean and lets the spread grow with the level. `train --objective l1` or `l2` switches it, and early stopping follows the objective.
- **Evaluation**: one fold per test month. Each fold trains on all earlier data, stops early on its last 28 days, and refits on the whole window. The final model trains on all data with the median number of trees from the folds.
- **Reference forecasts**: same hour yesterday, same hour last week, and the mean of the last five days of the same type.

## Pipeline commands

| Command | Purpose |
|---|---|
| `download` | Extract the clean archive and locate the house folders |
| `prepare` | Hourly whole-home, appliance and weather tables, clock check, quality report |
| `plegma_baseline.diagnose` | Clock check against the sun and daily routines, coverage per house |
| `devices` | Per-device table with dataset wattage and measured operating power |
| `train` | Rolling-origin evaluation, final model, saved run in `runs/<name>` |
| `export` | Portable model folder in `models/<name>` with a model card |
| `predict` | Day-ahead forecast for one local day |
| `compare` | Paired comparison of two runs on identical test months |

Options appear with `python -m plegma_baseline <command> --help`.

## Metrics

- `wape`: total absolute error over total consumption, on hourly values. This is the headline metric for KPI1.1.
- `wape_house_mean`: WAPE per home, then averaged, so each home counts equally.
- `daily_wape`: WAPE on daily energy, for days with at least 20 scored hours.
- `bias`: total forecast minus total consumption, over total consumption.
- `skill_vs_seasonal_d7`: one minus the WAPE ratio against the same-hour-last-week forecast.

The shared model and the three reference forecasts are scored on the same hours. Per-house and cold-start models (`--compare-local`, `--cold-start`) cover fewer hours, because a home needs history before its own model exists. Each appears with a paired row, such as `lgbm_global [lgbm_local hours]`, which scores the shared model on exactly those hours.

## KPI1.1 protocol

1. Build a parquet file with `house`, `utc` (hourly, UTC) and one column per context signal, such as declared appliance runs, planned AC use or a declared absence. Each value uses only information available before local midnight of its day.
2. Train with the same configuration and compare:

```bash
python -m plegma_baseline train --run-name context_v1 --extra-features context.parquet
python -m plegma_baseline compare --a runs/final_tweedie --b runs/context_v1
```

`compare` reports both WAPE values, the relative reduction, a 95% interval from resampling house-days, and the result per home. KPI1.1 targets a 20% relative reduction.

## Plegma data notes

- **Clock**: the timestamps use a fixed UTC+3 clock. `diagnose` confirms it with the outdoor temperature cycle and daily routines around the October 2022 and March 2023 clock changes.
- **Coverage**: homes join between July 2022 and April 2023, and House 2 ends in March 2023. Early test months therefore hold fewer homes, so per-home and seasonal results read best alongside the totals.
- **Whole-home totals**: seven homes match Table 6 of the data descriptor within 2%. Homes with more missing hours total less, since the pipeline counts recorded hours. The published clean files are used as they are.
- **Appliance level** (for device twins): House 8's `ac_1` column repeats its dishwasher series. House 2's kettle readings stay below the release's cleaning limit. House 5's fridge plug covers 99 days. House 11 has six monitored appliances, as in Table 3.

## Configuration

`config.yaml` holds every setting. The main ones:

| Setting | Default | Meaning |
|---|---|---|
| `prepare.clock` | `fixed_utc3` | Clock of the CSV timestamps (`auto` runs the check) |
| `features.weather` | `actual` | `actual`, `persistence` (previous day) or `none` |
| `features.use_house_id` | `false` | Adds the Plegma house ID as an input |
| `cv.first_test_month` | `2023-01` | First rolling-origin test month |
| `model.objective` | `tweedie` | `tweedie`, `l2` or `l1` |

## Tests

```bash
pytest -q                     # fast tests, about ten seconds
python tests/smoke_test.py    # full run on synthetic data, a few minutes
```

The synthetic data follow the folder layout, file names, columns and timestamp format of the clean release. GitHub Actions runs the fast tests on every push.

## Repository layout

```
plegma-baseline/
  config.yaml
  pyproject.toml
  plegma_baseline/
    __main__.py      command line
    forecaster.py    Forecaster: load a model, forecast a day
    download.py      archive extraction
    prepare.py       hourly tables and clock check
    diagnose.py      clock and coverage checks
    devices.py       device table with dataset and measured power
    calendar_gr.py   Greek holidays, Easter week, tariff hours
    features.py      features and reference forecasts
    train.py         folds, models, saving
    metrics.py       WAPE and related scores
    export.py        model folders and model cards
    predict.py       command-line forecast
    compare.py       paired run comparison
    data/houses.csv  household descriptors (Tables 2 and 3)
  models/            exported models
  examples/          usage examples
  tests/             fast tests, smoke test, synthetic data
```

## Citation

Please cite this repository (see `CITATION.cff`) together with the dataset:

Athanasoulias, S. et al. The Plegma dataset: Domestic appliance-level and aggregate electricity demand with metadata from Greece. *Scientific Data* 11, 376 (2024). https://doi.org/10.1038/s41597-024-03208-0

## License

The code is released under the Apache License 2.0 (see `LICENSE`). The Plegma data are released under CC BY 4.0. Exported models are trained on Plegma, so please credit the data descriptor when you share them.

## Acknowledgement

Developed in FLEXI, "Human-centered AI and Digital Twin powered energy system integration for flexibility markets". FLEXI is funded by CETPartnership, the Clean Energy Transition Partnership, under the 2024 joint call for research proposals, co-funded by the European Commission (GA N°101069750) and with the funding organisations FFG Austrian Research Promotion Agency (Austria), NWO Dutch Research Council (the Netherlands), Swedish Energy Agency (Sweden) and GSRI (Greece).

Co-funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
