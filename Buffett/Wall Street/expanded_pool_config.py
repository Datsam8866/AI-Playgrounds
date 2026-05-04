# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "stock_forecast.sqlite"
TRAIN_START = "2000-01-01"
DATA_END = "2026-05-05"
CORE_TICKER = "VOO"
REFERENCE_TICKERS = ["SPY", "QQQ"]

BASE_POOL_TICKERS = [
    "VOO",
    "GOOGL",
    "NVDA",
    "AVGO",
    "TSLA",
    "TSM",
    "SSO",
    "SOXX",
    "QQQ",
    "PLTR",
    "MSTR",
    "VRT",
    "AMD",
    "DELL",
    "MRVL",
    "QCOM",
]

REQUESTED_POOL_TICKERS = [
    "AAPL",
    "ACN",
    "ADBE",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "CLS",
    "COHR",
    "CRM",
    "CRWD",
    "CRWV",
    "DELL",
    "GOOGL",
    "IBM",
    "IFNNY",
    "INTC",
    "LITE",
    "LOGI",
    "META",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "NFLX",
    "NOW",
    "NTAP",
    "NVDA",
    "OKTA",
    "ORCL",
    "PLTR",
    "P",       # Pure Storage → Everpure，2026-04-17 改 ticker PSTG→P
    "QCOM",
    "SMCI",
    "SNDK",
    "SOXL",
    "SSO",
    "TLT",
    "TQQQ",
    "TSLA",
    "TSM",
]

EXPANDED_POOL_TICKERS: list[str] = []
for ticker in BASE_POOL_TICKERS + REQUESTED_POOL_TICKERS:
    if ticker not in EXPANDED_POOL_TICKERS:
        EXPANDED_POOL_TICKERS.append(ticker)

SATELLITE_POOL_TICKERS = [ticker for ticker in EXPANDED_POOL_TICKERS if ticker != CORE_TICKER]
PRICE_TICKERS = sorted(set(EXPANDED_POOL_TICKERS + REFERENCE_TICKERS))
LEVERAGED_ETFS = ["SSO", "SOXL", "TQQQ"]
NON_LEVERAGED_EXPANDED_POOL_TICKERS = [
    ticker for ticker in EXPANDED_POOL_TICKERS if ticker not in LEVERAGED_ETFS
]
NON_LEVERAGED_PRICE_TICKERS = sorted(set(NON_LEVERAGED_EXPANDED_POOL_TICKERS + REFERENCE_TICKERS))
