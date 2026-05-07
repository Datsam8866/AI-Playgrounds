"""
Compatibility layer for CPBL regime walk-forward evaluation.

The daily predictor owns the current advanced feature builder and model
parameters.  The regime evaluator imports this module name, so keep this file
as a thin wrapper to avoid duplicating feature logic.
"""

from datetime import date

from predict_today import (
    ADVANCED_FALLBACK_FEATURES,
    ADVANCED_PRIMARY_FEATURES,
    DB_PATH,
    ELO_HOME_ADV as ELO_HOME_ADVANTAGE,
    ELO_K,
    ELO_REGRESSION,
    FRANCHISE_MAP,
    XGB_PARAMS,
    load_data,
)

BACKTEST_START_YEAR = 2016
BACKTEST_END_YEAR = 2025


def build_advanced_rows(conn) -> list[dict]:
    """Return all completed CPBL regular-season training rows."""
    train_rows, _predict_games = load_data(conn, date(2100, 1, 1))
    return train_rows
