"""Hand-checkable tests for orb_backtest. Run: python test_orb.py  (or: pytest test_orb.py)"""
import numpy as np
import pandas as pd

from orb_backtest import Config, OPEN_MIN, TZ, run_backtest

ZERO = dict(commission_per_share=0.0, slippage_bps=0.0, max_leverage=10.0)


def make_day(rows, date="2024-03-04"):
    """rows: list of (open, high, low, close), 5-minute bars starting 09:30 ET."""
    idx = pd.Timestamp(date).tz_localize(TZ) + pd.to_timedelta(OPEN_MIN + 5 * np.arange(len(rows)), unit="m")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# OR (first 3 bars): high 101, low 99. Bar 3 closes at 101.5 -> long signal, entry at bar 4 open.
BASE = [
    (100.0, 101.0, 99.5, 100.5),
    (100.5, 101.0, 100.0, 100.8),
    (100.8, 101.0, 99.0, 100.0),
    (100.0, 101.5, 100.0, 101.5),   # signal bar
    (101.6, 102.0, 101.4, 101.9),   # entry bar (entry @ 101.6)
    (101.9, 103.0, 101.8, 102.5),
]


def test_long_stopped_out_exact_numbers():
    rows = BASE + [(102.5, 102.6, 98.5, 99.0)]  # low pierces stop at 99.0
    t = run_backtest(make_day(rows), Config(or_minutes=15, **ZERO)).trades
    assert len(t) == 1
    r = t.iloc[0]
    assert r.side == 1 and r.reason == "stop"
    assert r.entry_px == 101.6 and r.exit_px == 99.0
    assert r.shares == 384                       # floor(1000 / 2.6)
    assert abs(r.pnl - (99.0 - 101.6) * 384) < 1e-9


def test_entry_is_next_bar_open_not_signal_close():
    t = run_backtest(make_day(BASE + [(102.5, 102.6, 98.5, 99.0)]), Config(or_minutes=15, **ZERO)).trades
    assert t.iloc[0].entry_px == 101.6 and t.iloc[0].entry_px != 101.5


def test_no_lookahead_future_bars_dont_change_entry():
    a = BASE + [(102.5, 102.6, 98.5, 99.0)]
    b = BASE + [(102.5, 150.0, 1.0, 140.0)]     # wildly different future
    ta = run_backtest(make_day(a), Config(or_minutes=15, **ZERO)).trades.iloc[0]
    tb = run_backtest(make_day(b), Config(or_minutes=15, **ZERO)).trades.iloc[0]
    assert (ta.entry_time, ta.entry_px, ta.side, ta.shares) == (tb.entry_time, tb.entry_px, tb.side, tb.shares)


def test_stop_wins_when_stop_and_target_in_same_bar():
    rows = BASE + [(101.9, 105.0, 98.0, 100.0)]  # touches both 104.2 target and 99 stop
    t = run_backtest(make_day(rows), Config(or_minutes=15, r_multiple=1.0, **ZERO)).trades
    assert t.iloc[0].reason == "stop"


def test_target_hit():
    rows = BASE + [(102.5, 104.5, 102.0, 104.0)]  # target = 101.6 + 2.6 = 104.2
    t = run_backtest(make_day(rows), Config(or_minutes=15, r_multiple=1.0, **ZERO)).trades
    r = t.iloc[0]
    assert r.reason == "target" and abs(r.exit_px - 104.2) < 1e-9


def test_short_side_mirror():
    rows = [
        (100.0, 101.0, 99.5, 100.5),
        (100.5, 101.0, 100.0, 100.8),
        (100.8, 101.0, 99.0, 100.0),
        (100.0, 100.0, 98.5, 98.5),    # closes below OR low (99) -> short
        (98.4, 98.6, 98.0, 98.2),      # entry @ 98.4, stop = 101.0, risk 2.6
        (98.2, 98.3, 95.5, 96.5),
    ]
    t = run_backtest(make_day(rows), Config(or_minutes=15, r_multiple=1.0, **ZERO)).trades
    r = t.iloc[0]
    assert r.side == -1 and r.entry_px == 98.4
    assert r.reason == "target" and abs(r.exit_px - (98.4 - 2.6)) < 1e-9
    assert r.pnl > 0


def test_eod_flatten_at_1555_open():
    quiet = [(101.9, 102.0, 101.8, 101.9)] * (78 - 5)   # indices 5..77 (77 = 15:55)
    rows = BASE[:5] + [(101.9, 102.0, 101.8, 101.9)] + quiet[1:]
    rows = list(rows)
    rows[77] = (102.2, 102.3, 102.0, 102.1)
    assert len(rows) == 78
    t = run_backtest(make_day(rows), Config(or_minutes=15, **ZERO)).trades
    r = t.iloc[0]
    assert r.reason == "eod" and r.exit_px == 102.2
    assert r.exit_time.hour == 15 and r.exit_time.minute == 55


def test_costs_reduce_pnl():
    rows = BASE + [(102.5, 104.5, 102.0, 104.0)]
    free = run_backtest(make_day(rows), Config(or_minutes=15, r_multiple=1.0, **ZERO)).trades.iloc[0]
    paid = run_backtest(make_day(rows), Config(or_minutes=15, r_multiple=1.0, max_leverage=10.0,
                                               commission_per_share=0.005, slippage_bps=5.0)).trades.iloc[0]
    assert paid.pnl < free.pnl
    assert paid.costs > 0 and abs(paid.gross_pnl - paid.pnl - paid.costs) < 1e-9


def test_no_entry_after_cutoff_and_incomplete_or_skipped():
    # signal too late: breakout only on a bar whose NEXT bar starts after 15:00
    rows = [(100.0, 101.0, 99.0, 100.0)] * 3 + [(100.0, 100.5, 99.5, 100.0)] * 60  # 63 bars -> up to 14:40
    rows += [(100.0, 100.5, 99.5, 100.0)] * 14 + [(100.0, 102.0, 100.0, 102.0)]   # signal on bar 77 (15:55)
    assert run_backtest(make_day(rows), Config(or_minutes=15, **ZERO)).trades.empty
    # missing opening-range bar -> day skipped
    day = make_day(BASE + [(102.5, 102.6, 98.5, 99.0)]).iloc[1:]
    assert run_backtest(day, Config(or_minutes=15, **ZERO)).trades.empty


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
