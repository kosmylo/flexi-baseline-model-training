# Model card: plegma-tweedie-v1

Day-ahead forecast of hourly whole-home electricity use. Trained on the Plegma dataset: 13 homes in Attica, Greece.

## Model

- LightGBM, objective `tweedie`, 83 trees, 37 inputs.
- Inputs: calendar (Greek holidays, Orthodox Easter week, tariff hours), same-hour load history, household descriptors, outdoor weather.
- Target: hourly kWh divided by each home's mean load over the previous 28 days.
- Forecast made at local midnight (Europe/Athens) for the 24 hours of the next local day (23 or 25 on clock-change days).
- Package version 1.0.0, LightGBM 4.7.0.

## Training data

- Plegma clean release, homes 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13.
- 80,302 training hours from 2022-07-18 to 2023-09-30 (UTC).
- Data: Athanasoulias et al., *Scientific Data* 11, 376 (2024), CC BY 4.0.

## Evaluation

Rolling origin with one test month per fold, 2023-01 to 2023-09. Each fold trains on all earlier data. All forecasts are scored on the same hours.

| Forecast | Hourly WAPE | Daily WAPE | Bias | RMSE (kWh) | Skill vs last week |
|---|---|---|---|---|---|
| This model | 0.758 | 0.343 | -0.076 | 0.505 | 0.204 |
| Similar-day average | 0.813 | 0.381 | -0.004 | 0.541 | 0.147 |
| Same hour yesterday | 0.879 | 0.353 | 0.001 | 0.663 | 0.078 |
| Same hour last week | 0.953 | 0.495 | -0.008 | 0.691 | 0.000 |

## Intended use

- Reference baseline for FLEXI KPI1.1: the context-aware model reuses these settings and test months.
- Household-level demand forecasts for digital twins and simulated households.

## Scope notes

- Outdoor weather inputs take recorded values as a stand-in for a perfect day-ahead forecast. Real weather forecasts add some error.
- The model forecasts the mean hour. Short appliance bursts spread over neighbouring hours, which keeps energy totals close to reality.
- The scores describe 13 homes in Attica. Other climates and building stocks call for retraining with the same pipeline.
