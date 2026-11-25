#!/usr/bin/env python3
"""
Download intraday bars from Alpaca into the CSV format orb_backtest.py expects.

NOTE: written against the alpaca-py SDK but NOT tested end-to-end (no network access
where it was written). If it errors, check the alpaca-py docs for current signatures.

Setup:
    pip install alpaca-py pandas
    export ALPACA_API_KEY=...      # a paper-trading key is fine for market data
    export ALPACA_API_SECRET=...

Usage:
    python fetch_alpaca.py SPY --start 2021-01-01 --end 2024-12-31 --minutes 5 --out spy_5min.csv

The free plan only includes the IEX feed (a single exchange): prices are fine for a first
look, but volumes are partial and the occasional bar may be missing. The backtester skips
days with an incomplete opening range rather than guessing.
"""
import argparse
import os
from datetime import datetime

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("symbol")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--minutes", type=int, default=5)
    p.add_argument("--feed", choices=["iex", "sip"], default="iex")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    req = StockBarsRequest(
        symbol_or_symbols=a.symbol,
        timeframe=TimeFrame(a.minutes, TimeFrameUnit.Minute),
        start=datetime.fromisoformat(a.start),
        end=datetime.fromisoformat(a.end),
        feed=DataFeed.IEX if a.feed == "iex" else DataFeed.SIP,
    )
    df = client.get_stock_bars(req).df
    if "symbol" in (df.index.names or []):
        df = df.xs(a.symbol, level="symbol")
    df = df[["open", "high", "low", "close", "volume"]]
    # index is tz-aware UTC; orb_backtest converts to US/Eastern and filters to regular hours
    df.to_csv(a.out, index_label="timestamp", date_format="%Y-%m-%dT%H:%M:%S+00:00")
    print(f"Wrote {len(df):,} bars to {a.out}")


if __name__ == "__main__":
    main()
