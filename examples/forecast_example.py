"""Day-ahead forecast with a saved model.

Run from the repository folder after `prepare` and `export`:
    python examples/forecast_example.py
"""
import pandas as pd

from plegma_baseline import Forecaster

model = Forecaster.load("models/plegma-tweedie-v1")

# Plegma homes: hourly energy and outdoor weather prepared by `python -m plegma_baseline prepare`
load = pd.read_parquet("data/processed/hourly_load.parquet")      # house, utc, load_kwh
weather = pd.read_parquet("data/processed/hourly_env.parquet")    # house, utc, ext_temp, ext_hum

forecast = model.forecast(load, day="2023-09-20", weather=weather)
print(forecast[forecast["house"] == 3].to_string(index=False))

# Any other household: its own ID, one weather series, and the descriptors you know
own = load[load["house"] == 3].assign(house="my-home")
city_weather = weather[weather["house"] == 3].drop(columns="house")
home = pd.DataFrame([{"house": "my-home", "occupants": 2, "rooms": 3, "n_ac": 1, "has_boiler": 1}])
print(model.forecast(own, day="2023-09-20", weather=city_weather, households=home).head())
