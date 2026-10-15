"""Day-ahead household load forecasting baseline on the Plegma dataset (FLEXI uUBI, KPI1.1)."""
__version__ = "1.0.0"

from .forecaster import Forecaster  # noqa: E402

__all__ = ["Forecaster", "__version__"]
