# Models

Each subfolder holds one exported model, ready for `Forecaster.load`:

- `model.txt`: the LightGBM model
- `model_meta.json`: feature list, training period, heating codes and library versions
- `config.yaml`: the feature and model settings used in training
- `households.csv`: the Plegma household descriptors
- `MODEL_CARD.md`: data, evaluation protocol and scores
- `metrics_*.csv`, `folds.json`, `feature_importance.csv`: the full evaluation

Create a model folder from a training run with:

```bash
python -m plegma_baseline export --run runs/final_tweedie --name plegma-tweedie-v1
```
