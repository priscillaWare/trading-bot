#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) backtester.

Strategy (all rules are mechanical, nothing is discretionary):
  1. The opening range (OR) is the high/low of the first `or_minutes` of the regular session.
  2. After the OR completes, the first bar that CLOSES beyond the OR is the signal.
  3. Entry is at the OPEN of the NEXT bar (never at the signal bar's close), plus slippage.
  4. Stop is placed inside the range: `stop_frac` = 1.0 means the opposite side of the OR,
     0.5 means the midpoint.
  5. Optional profit target at `r_multiple` x initial risk.
  6. Always flat by `flatten_time`. One trade per day maximum.

Honesty features:
  * No look-ahead: decisions on bar i use only bars <= i; fills happen on bar i+1.
  * If stop and target are both touched within one bar, the STOP is assumed to hit first.
  * Costs (commission + slippage on every fill, including stops) are on by default.
  * `grid` splits data into in-sample / out-of-sample and reports how much the
    in-sample "winner" degrades, plus the Sharpe you'd expect from pure luck given the
    number of variants tried.

Usage:
  python orb_backtest.py demo                      # synthetic random-walk data (should show NO edge)
  python orb_backtest.py demo --trend 1.0          # synthetic data WITH an edge (sanity check)
  python orb_backtest.py run  spy_5min.csv --or-minutes 15 --r-multiple 2 --plot out.png
  python orb_backtest.py grid spy_5min.csv --split 0.6

CSV format: columns timestamp,open,high,low,close[,volume]. Timestamps either carry a UTC offset
(e.g. 2024-03-04T14:30:00+00:00) or are naive and treated as US/Eastern.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Optional

import numpy as np
import pandas as pd

TZ = "America/New_York"
OPEN_MIN = 9 * 60 + 30   # 09:30 ET in minutes since midnight
CLOSE_MIN = 16 * 60      # 16:00 ET

TRADE_COLS = [
    "date", "side", "entry_time", "exit_time", "shares", "entry_px", "exit_px",
    "reason", "gross_pnl", "costs", "pnl",
]


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    # strategy
    or_minutes: int = 15
    stop_frac: float = 1.0            # 1.0 = stop at opposite side of OR, 0.5 = midpoint
    r_multiple: Optional[float] = None  # None = no target, hold until flatten_time
    entry_cutoff: str = "15:00"       # no new entries at/after this time
    flatten_time: str = "15:55"       # exit at the open of the first bar at/after this time
    allow_long: bool = True
    allow_short: bool = True
    # risk / account
    risk_frac: float = 0.01           # fraction of equity risked per trade (to the stop)
    max_leverage: float = 1.0         # cap on notional / equity (1.0 = cash-account-like)
    start_equity: float = 100_000.0
    # costs
    commission_per_share: float = 0.005
    slippage_bps: float = 1.0         # applied adversely on every fill


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# --------------------------------------------------------------------------- data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to: tz-aware US/Eastern index, lowercase OHLC(V) columns, regular session only."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    else:
        idx = idx.tz_convert(TZ)
    df.index = idx
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= OPEN_MIN) & (mins < CLOSE_MIN)]
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    return df[cols].astype(float)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    raw = df[ts_col].astype(str)
    has_tz = raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all()
    ts = pd.to_datetime(raw, utc=True) if has_tz else pd.to_datetime(raw)
    df = df.drop(columns=[ts_col])
    df.index = pd.DatetimeIndex(ts)
    return prepare(df)


def infer_bar_minutes(bars: pd.DataFrame) -> int:
    step = bars.index.to_series().diff().dt.total_seconds().median() / 60
    return int(round(step))


def make_synthetic(days: int = 756, bar_minutes: int = 5, seed: int = 0, trend: float = 0.0,
                   start_price: float = 400.0, daily_vol: float = 0.01,
                   gap_vol: float = 0.004) -> pd.DataFrame:
    """Synthetic 5-min bars. trend=0 -> pure random walk (no edge by construction).
    trend>0 adds a persistent per-day drift (sd = trend * daily_vol), i.e. an exploitable
    intraday momentum, which lets you confirm the backtester CAN detect an edge."""
    rng = np.random.default_rng(seed)
    n = 390 // bar_minutes
    x = np.linspace(0, 1, n)
    shape = 1 + 1.5 * (np.exp(-8 * x) + np.exp(-8 * (1 - x)))  # U-shaped intraday volatility
    sigma = daily_vol * shape / np.sqrt((shape ** 2).sum())
    offsets = pd.to_timedelta(OPEN_MIN + np.arange(n) * bar_minutes, unit="m")
    parts, prev_close = [], start_price
    for d in pd.bdate_range("2021-01-04", periods=days):
        drift = trend * daily_vol * rng.standard_normal()
        rets = rng.standard_normal(n) * sigma + drift / n
        first_open = prev_close * (1 + rng.standard_normal() * gap_vol)
        closes = first_open * np.cumprod(1 + rets)
        opens = np.concatenate([[first_open], closes[:-1]])
        wick = np.abs(rng.standard_normal((2, n))) * sigma * 0.5
        highs = np.maximum(opens, closes) * (1 + wick[0])
        lows = np.minimum(opens, closes) * (1 - wick[1])
        idx = pd.Timestamp(d).tz_localize(TZ) + offsets
        parts.append(pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": rng.integers(10_000, 100_000, n).astype(float)}, index=idx))
        prev_close = closes[-1]
    return pd.concat(parts)


# --------------------------------------------------------------------------- simulation
def _simulate_day(day: pd.DataFrame, cfg: Config, equity: float, bar_min: int) -> Optional[dict]:
    t = day.index
    mins = (t.hour * 60 + t.minute).to_numpy()
    o, h, l, c = (day[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    or_end = OPEN_MIN + cfg.or_minutes
    in_or = mins < or_end
    n_or = int(in_or.sum())
    if mins[0] != OPEN_MIN or n_or != cfg.or_minutes // bar_min:
        return None  # incomplete opening range (missing bars), skip the day
    or_hi, or_lo = h[:n_or].max(), l[:n_or].min()
    rng = or_hi - or_lo
    if rng <= 0:
        return None

    cutoff, flat = _hhmm(cfg.entry_cutoff), _hhmm(cfg.flatten_time)

    # ---- find signal on bar i (using its close), enter at open of bar j = i + 1
    entry = None
    for i in range(n_or, len(t) - 1):
        j = i + 1
        if mins[j] >= cutoff:
            break
        if mins[j] - mins[i] != bar_min:
            continue  # gap in data, don't trade across it
        if c[i] > or_hi and cfg.allow_long:
            entry = (j, 1)
            break
        if c[i] < or_lo and cfg.allow_short:
            entry = (j, -1)
            break
    if entry is None:
        return None
    j, side = entry

    slip = cfg.slippage_bps / 1e4
    raw_entry = o[j]
    entry_px = raw_entry * (1 + side * slip)
    stop = or_hi - cfg.stop_frac * rng if side == 1 else or_lo + cfg.stop_frac * rng
    risk = side * (entry_px - stop)
    if risk <= 0:
        return None  # price already beyond the stop at entry time
    target = entry_px + side * cfg.r_multiple * risk if cfg.r_multiple else None

    shares = math.floor(min(equity * cfg.risk_frac / risk, equity * cfg.max_leverage / entry_px))
    if shares < 1:
        return None

    # ---- walk forward to the exit
    raw_exit, reason, k_exit = None, None, None
    for k in range(j, len(t)):
        if mins[k] >= flat:
            raw_exit, reason, k_exit = o[k], "eod", k
            break
        if side == 1:
            if l[k] <= stop:
                raw_exit, reason, k_exit = min(stop, o[k]), "stop", k
                break
            if target is not None and h[k] >= target:
                raw_exit, reason, k_exit = max(target, o[k]), "target", k
                break
        else:
            if h[k] >= stop:
                raw_exit, reason, k_exit = max(stop, o[k]), "stop", k
                break
            if target is not None and l[k] <= target:
                raw_exit, reason, k_exit = min(target, o[k]), "target", k
                break
    if raw_exit is None:  # data ended before flatten_time (e.g. half-day)
        raw_exit, reason, k_exit = c[-1], "eod_data", len(t) - 1

    exit_px = raw_exit * (1 - side * slip)
    gross = side * (raw_exit - raw_entry) * shares
    pnl = side * (exit_px - entry_px) * shares - 2 * cfg.commission_per_share * shares
    return {
        "date": t[0].date(), "side": side, "entry_time": t[j], "exit_time": t[k_exit],
        "shares": shares, "entry_px": entry_px, "exit_px": exit_px, "reason": reason,
        "gross_pnl": gross, "costs": gross - pnl, "pnl": pnl,
    }


@dataclass
class Result:
    cfg: Config
    trades: pd.DataFrame
    daily: pd.Series  # daily strategy return on equity (0 on no-trade days)


def run_backtest(bars: pd.DataFrame, cfg: Config) -> Result:
    bar_min = infer_bar_minutes(bars)
    if cfg.or_minutes % bar_min:
        raise ValueError(f"or_minutes={cfg.or_minutes} is not a multiple of the {bar_min}-minute bar size")
    equity, rows, daily = cfg.start_equity, [], {}
    for date, day in bars.groupby(bars.index.date):
        tr = _simulate_day(day, cfg, equity, bar_min)
        if tr is None:
            daily[date] = 0.0
        else:
            daily[date] = tr["pnl"] / equity
            equity += tr["pnl"]
            rows.append(tr)
    s = pd.Series(daily)
    s.index = pd.to_datetime(s.index)
    return Result(cfg, pd.DataFrame(rows, columns=TRADE_COLS), s)


# --------------------------------------------------------------------------- statistics
def summarize(daily: pd.Series, trades: Optional[pd.DataFrame] = None) -> dict:
    n = len(daily)
    if n < 2:
        return {"days": n}
    eq = (1 + daily).cumprod()
    years = n / 252
    sd = daily.std()
    out = {"days": n}
    out["total_return"] = float(eq.iloc[-1] - 1)
    out["cagr"] = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    out["vol"] = float(sd * math.sqrt(252))
    out["sharpe"] = float(daily.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")
    out["t_stat"] = out["sharpe"] * math.sqrt(years) if sd > 0 else float("nan")
    path = pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)])
    out["max_dd"] = float((path / path.cummax() - 1).min())
    if trades is not None:
        nt = len(trades)
        out["trades"] = nt
        if nt:
            wins, losses = trades.loc[trades.pnl > 0, "pnl"], trades.loc[trades.pnl <= 0, "pnl"]
            out["win_rate"] = float(len(wins) / nt)
            out["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
            out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
            out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
            out["pct_long"] = float((trades.side == 1).mean())
            out["gross_pnl"] = float(trades.gross_pnl.sum())
            out["costs"] = float(trades.costs.sum())
            out["net_pnl"] = float(trades.pnl.sum())
            out["days_in_market"] = nt / n
    return out


def buy_hold_daily(bars: pd.DataFrame) -> pd.Series:
    close = bars["close"].groupby(bars.index.date).last()
    close.index = pd.to_datetime(close.index)
    return close.pct_change().dropna()


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Rough annualised Sharpe you'd expect the BEST of n_trials zero-edge strategies to show
    by luck alone (Bailey & Lopez de Prado). Overstates when trials are highly correlated."""
    if n_trials < 2 or years <= 0:
        return 0.0
    g, nd = 0.5772156649, NormalDist()
    z = (1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e))
    return z / math.sqrt(years)


def print_stats(label: str, s: dict) -> None:
    def pct(k): return f"{s[k] * 100:8.2f}%" if k in s else "     n/a"
    print(f"\n=== {label} ===")
    if s.get("days", 0) < 2:
        print("  not enough data")
        return
    extra = f"   trades {s['trades']:>5}   days in market {s.get('days_in_market', 0) * 100:5.1f}%" if "trades" in s else ""
    print(f"  days {s['days']:>5}{extra}")
    print(f"  total return {pct('total_return')}   CAGR {pct('cagr')}   vol {pct('vol')}   maxDD {pct('max_dd')}")
    print(f"  Sharpe {s['sharpe']:6.2f}   t-stat {s['t_stat']:6.2f}   (|t| < 2 is indistinguishable from luck)")
    if s.get("trades"):
        print(f"  win rate {s['win_rate'] * 100:5.1f}%   profit factor {s['profit_factor']:5.2f}   "
              f"avg win {s['avg_win']:8.0f}   avg loss {s['avg_loss']:8.0f}   long {s['pct_long'] * 100:4.0f}%")
        print(f"  gross P&L {s['gross_pnl']:10.0f}   costs {s['costs']:10.0f}   net P&L {s['net_pnl']:10.0f}")


# --------------------------------------------------------------------------- grid / out-of-sample
DEFAULT_GRID = {
    "or_minutes": [5, 15, 30],
    "stop_frac": [0.5, 1.0],
    "r_multiple": [None, 1.0, 2.0],
}


def grid_search(bars: pd.DataFrame, base: Config, grid: dict, split: float = 0.6) -> pd.DataFrame:
    dates = np.array(sorted(set(bars.index.date)))
    cut = dates[int(len(dates) * split)]
    is_bars = bars[bars.index.date < cut]
    oos_bars = bars[bars.index.date >= cut]
    bar_min = infer_bar_minutes(bars)
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        if params.get("or_minutes", base.or_minutes) % bar_min:
            continue
        cfg = replace(base, **params)
        a, b = run_backtest(is_bars, cfg), run_backtest(oos_bars, cfg)
        sa, sb = summarize(a.daily, a.trades), summarize(b.daily, b.trades)
        rows.append({**params, "is_sharpe": sa.get("sharpe", np.nan), "is_trades": sa.get("trades", 0),
                     "oos_sharpe": sb.get("sharpe", np.nan), "oos_trades": sb.get("trades", 0),
                     "oos_return": sb.get("total_return", np.nan)})
    table = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    if "r_multiple" in table.columns:
        table["r_multiple"] = table["r_multiple"].map(lambda v: "none" if pd.isna(v) else f"{v:g}")
    table.attrs["cut"], table.attrs["is_years"] = cut, len(set(is_bars.index.date)) / 252
    return table


# --------------------------------------------------------------------------- CLI
def _cfg_from_args(a) -> Config:
    return Config(or_minutes=a.or_minutes, stop_frac=a.stop_frac, r_multiple=a.r_multiple,
                  risk_frac=a.risk_frac, max_leverage=a.max_leverage,
                  commission_per_share=a.commission, slippage_bps=a.slippage_bps)


def _do_run(bars: pd.DataFrame, cfg: Config, plot: Optional[str] = None) -> None:
    res = run_backtest(bars, cfg)
    print(f"\nConfig: {cfg}")
    print_stats("ORB strategy (after costs)", summarize(res.daily, res.trades))
    print_stats("Buy & hold (close-to-close, same data)", summarize(buy_hold_daily(bars)))
    print("\nNote: the strategy is in the market for part of the day only, so compare risk-adjusted "
          "(Sharpe), not raw return.")
    if not res.trades.empty:
        print("\nExit reasons:", res.trades.reason.value_counts().to_dict())
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        (1 + res.daily).cumprod().plot(ax=ax, label="ORB (after costs)")
        (1 + buy_hold_daily(bars)).cumprod().plot(ax=ax, label="Buy & hold", alpha=0.7)
        ax.set_title("Equity curves (growth of 1.0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot, dpi=120)
        print(f"Saved plot to {plot}")


def _do_grid(bars: pd.DataFrame, base: Config, split: float) -> None:
    t = grid_search(bars, base, DEFAULT_GRID, split)
    pd.set_option("display.width", 140)
    print(f"\nIn-sample: before {t.attrs['cut']}   Out-of-sample: from {t.attrs['cut']}   "
          f"({len(t)} variants tried)\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
    best = t.iloc[0]
    luck = expected_max_sharpe(len(t), t.attrs["is_years"])
    print(f"\nBest in-sample variant: IS Sharpe {best.is_sharpe:.2f}  ->  OOS Sharpe {best.oos_sharpe:.2f}")
    print(f"Median OOS Sharpe across ALL variants: {t.oos_sharpe.median():.2f}")
    print(f"Rough Sharpe the best of {len(t)} zero-edge variants would show by luck alone: ~{luck:.2f}")
    print("If the in-sample winner isn't clearly above that, and the OOS Sharpe collapses, "
          "treat the result as noise.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--or-minutes", type=int, default=15)
        sp.add_argument("--stop-frac", type=float, default=1.0)
        sp.add_argument("--r-multiple", type=float, default=None)
        sp.add_argument("--risk-frac", type=float, default=0.01)
        sp.add_argument("--max-leverage", type=float, default=1.0)
        sp.add_argument("--commission", type=float, default=0.005, help="$ per share per side")
        sp.add_argument("--slippage-bps", type=float, default=1.0, help="bps per fill")

    r = sub.add_parser("run", help="backtest one configuration on a CSV")
    r.add_argument("csv")
    r.add_argument("--plot", default=None)
    common(r)
    g = sub.add_parser("grid", help="parameter grid with in-sample / out-of-sample split")
    g.add_argument("csv")
    g.add_argument("--split", type=float, default=0.6)
    common(g)
    d = sub.add_parser("demo", help="run on synthetic data")
    d.add_argument("--trend", type=float, default=0.0, help="0 = no edge; >0 injects intraday momentum")
    d.add_argument("--days", type=int, default=756)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--save", default=None, help="write the synthetic data to this CSV")
    d.add_argument("--plot", default=None)
    common(d)

    a = p.parse_args(argv)
    cfg = _cfg_from_args(a)
    if a.cmd == "demo":
        bars = prepare(make_synthetic(days=a.days, seed=a.seed, trend=a.trend))
        if a.save:
            bars.to_csv(a.save, index_label="timestamp")
        print(f"Synthetic data: {a.days} days, trend={a.trend} "
              f"({'NO edge by construction' if a.trend == 0 else 'edge injected'})")
        _do_run(bars, cfg, a.plot)
        _do_grid(bars, cfg, 0.6)
    else:
        bars = load_csv(a.csv)
        if a.cmd == "run":
            _do_run(bars, cfg, a.plot)
        else:
            _do_grid(bars, cfg, a.split)


if __name__ == "__main__":
    main()
