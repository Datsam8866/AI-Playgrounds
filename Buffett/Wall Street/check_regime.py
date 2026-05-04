import sqlite3, warnings, os, sys
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
os.environ.setdefault("LOCALAPPDATA", ".sandbox_cache")
os.environ.setdefault("TEMP", ".sandbox_cache")
os.environ.setdefault("TMP", ".sandbox_cache")

db = sqlite3.connect("stock_forecast.sqlite")

spy = pd.read_sql(
    "SELECT date, adj_close FROM price_history WHERE ticker='SPY' ORDER BY date DESC LIMIT 250",
    db
).sort_values("date").reset_index(drop=True)
db.close()

spy_close = spy["adj_close"].iloc[-1]
spy_sma200 = spy["adj_close"].tail(200).mean()
spy_date = spy["date"].iloc[-1]

tnx = yf.download("^TNX", period="30d", progress=False)["Close"].dropna()
vix_df = yf.download("^VIX", period="5d", progress=False)["Close"].dropna()

tnx_last = float(tnx.iloc[-1].item() if hasattr(tnx.iloc[-1], "item") else tnx.iloc[-1])
tnx_sma20 = float(tnx.tail(20).mean().item() if hasattr(tnx.tail(20).mean(), "item") else tnx.tail(20).mean())
vix_last = float(vix_df.iloc[-1].item() if hasattr(vix_df.iloc[-1], "item") else vix_df.iloc[-1])

spy_ok = spy_close >= spy_sma200
tnx_ok = tnx_last <= tnx_sma20
vix_ok = vix_last <= 25

fails = sum([not vix_ok, not spy_ok, not tnx_ok])
regime = "risk_on" if fails == 0 else ("risk_off" if fails >= 2 else "caution")

print(f"\n=== Regime Check [{pd.Timestamp.today().date()}] ===")
print(f"  VIX  = {vix_last:.1f}   <= 25          -> {'OK' if vix_ok else 'FAIL'}")
print(f"  SPY  = {spy_close:.2f}  >= SMA200 {spy_sma200:.2f} -> {'OK' if spy_ok else 'FAIL'}")
print(f"  TNX  = {tnx_last:.3f}  <= SMA20  {tnx_sma20:.3f}  -> {'OK' if tnx_ok else 'FAIL'}")
print(f"\n  => REGIME: {regime.upper()}  ({fails}/3 conditions failed)")
print(f"  (SPY data as of {spy_date})")
