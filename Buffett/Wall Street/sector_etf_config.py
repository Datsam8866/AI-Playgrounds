# -*- coding: utf-8 -*-
"""
sector_etf_config.py
────────────────────
Ticker → Sector ETF mapping for expanded pool.

Rules:
- XLK  : Technology
- XLC  : Communication Services
- XLY  : Consumer Discretionary
- SPY  : fallback for ETFs / bonds
"""
from __future__ import annotations

TICKER_TO_SECTOR_ETF: dict[str, str] = {
    # XLK — Technology
    "NVDA": "XLK",
    "AMD": "XLK",
    "AVGO": "XLK",
    "TSM": "XLK",
    "QCOM": "XLK",
    "INTC": "XLK",
    "MU": "XLK",
    "SNDK": "XLK",
    "MRVL": "XLK",
    "SMCI": "XLK",
    "ARM": "XLK",
    "LITE": "XLK",
    "COHR": "XLK",
    "P": "XLK",
    "AAPL": "XLK",
    "MSFT": "XLK",
    "ADBE": "XLK",
    "CRM": "XLK",
    "NOW": "XLK",
    "OKTA": "XLK",
    "CRWD": "XLK",
    "PLTR": "XLK",
    "CRWV": "XLK",
    "ORCL": "XLK",
    "IBM": "XLK",
    "ACN": "XLK",
    "DELL": "XLK",
    "NTAP": "XLK",
    "LOGI": "XLK",
    "VRT": "XLK",
    "CLS": "XLK",
    "IFNNY": "XLK",
    "MSTR": "XLK",
    # XLC — Communication Services
    "GOOGL": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    # XLY — Consumer Discretionary
    "TSLA": "XLY",
    "AMZN": "XLY",
    # ETFs — use sector or SPY as proxy
    "VOO": "SPY",
    "QQQ": "XLK",
    "SSO": "SPY",
    "SOXX": "XLK",
    "SOXL": "XLK",
    "TQQQ": "XLK",
    "TLT": "SPY",
}

# Sector ETFs that need historical price data downloaded
SECTOR_ETFS: list[str] = ["XLK", "XLC", "XLY"]
