"""Tests for orb_live. Run: python test_orb_live.py  (or: pytest test_orb_live.py)"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from orb_backtest import OPEN_MIN, TZ, Config, make_synthetic, prepare, run_backtest
from orb_live import Bot, OrbEngine, Risk, run_sim
from sim_broker import SimBars, SimBroker

ZERO = dict(commission_per_share=0.0, slippage_bps=0.0)
BM = pd.Timedelta(minutes=5)


def make_day(rows, date="2024-03-04"):
    idx = pd.Timestamp(date).tz_localize(TZ) + pd.to_timedelta(OPEN_MIN + 5 * np.arange(len(rows)), unit="m")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


BASE = [
    (100.0, 101.0, 99.5, 100.5),
    (100.5, 101.0, 100.0, 100.8),
    (100.8, 101.0, 99.0, 100.0),    # OR (15 min): high 101, low 99
    (100.0, 101.5, 100.0, 101.5),   # signal bar (close > 101)
    (101.6, 102.0, 101.4, 101.9),   # entry bar: bot should buy at 101.6
]
QUIET = (101.9, 102.0, 101.8, 101.9)


def full_day(prefix, last=None):
    rows = list(prefix) + [QUIET] * (78 - len(prefix))
    if last:
        rows[77] = last
    return rows


def new_bot(bars, state_dir, risk=None, cfg=None, sim=None):
    cfg = cfg or Config(or_minutes=15, max_leverage=10.0, **ZERO)
    sim = sim or SimBroker(bars, cfg.start_equity)
    bot = Bot(cfg, risk or Risk(), sim, SimBars(sim), "SIM", state_dir, sleep=lambda s: None)
    return bot, sim


def drive(bot, sim, bars, until=None):
    """Step the bot at every bar boundary (and 16:00), like the live loop would."""
    for _, day in bars.groupby(bars.index.date):
        for t in list(day.index) + [day.index[-1] + BM]:
            if until is not None and t > until:
                return
            sim.set_now(t)
            bot.step()


# ------------------------------------------------------------------ engine
def test_engine_signal_and_levels():
    e = OrbEngine(Config(or_minutes=15), 5)
    day = make_day(BASE)
    sig = None
    for ts, r in day.iloc[:4].iterrows():
        sig = e.on_bar(ts, r.high, r.low, r.close, True) or sig
    assert sig and sig.side == 1 and sig.stop == 99.0 and (sig.or_hi, sig.or_lo) == (101.0, 99.0)


def test_engine_never_chases_stale_signal():
    e = OrbEngine(Config(or_minutes=15), 5)
    for ts, r in make_day(BASE).iloc[:4].iterrows():
        assert e.on_bar(ts, r.high, r.low, r.close, False) is None
    assert e.finished and e.reason == "missed_signal"


def test_engine_incomplete_opening_range_skips_day():
    e = OrbEngine(Config(or_minutes=15), 5)
    for ts, r in make_day(BASE).iloc[1:4].iterrows():   # first bar missing
        e.on_bar(ts, r.high, r.low, r.close, True)
    assert e.finished and e.reason == "incomplete_or"


# ------------------------------------------------------------------ parity with the backtest
def test_parity_with_backtest_over_many_days():
    for seed, trend in [(3, 0.5), (11, 0.0)]:
        bars = prepare(make_synthetic(days=260, seed=seed, trend=trend))
        for extra in [dict(), dict(r_multiple=1.5, stop_frac=0.5)]:
            cfg = Config(or_minutes=15, max_leverage=10.0, **ZERO, **extra)
            risk = Risk(max_daily_loss_frac=0.5)    # keep the loss limit out of the comparison
            d = tempfile.mkdtemp()
            try:
                bt = run_backtest(bars, cfg).trades.reset_index(drop=True)
                lv = pd.DataFrame(run_sim(bars, cfg, risk, d).trades)
            finally:
                shutil.rmtree(d)
            assert len(bt) > 50, "need a meaningful number of trades to compare"
            assert len(bt) == len(lv), (len(bt), len(lv))
            assert list(bt.date) == list(lv.date)
            assert list(bt.side) == list(lv.side)
            assert list(bt.entry_time) == list(lv.entry_time)
            assert np.allclose(bt.entry_px, lv.entry_px, atol=1e-9)           # entries must match exactly
            assert np.allclose(bt.exit_px, lv.exit_px, atol=0.0051, rtol=0)   # live stop/target prices are cent-rounded
            assert list(bt.reason.replace({"eod": "close"})) == list(lv.reason)


# ------------------------------------------------------------------ safety layers
def test_one_trade_per_day_even_after_stop_out():
    rows = BASE + [(101.9, 102.0, 98.5, 99.0),      # stopped out
                   (99.0, 100.0, 98.8, 99.5), (99.5, 103.0, 99.4, 102.8), (102.8, 104.0, 102.5, 103.9)]
    rows = full_day(rows)
    bars = make_day(rows)
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d)
    drive(bot, sim, bars)
    assert len(sim.trades) == 1 and sim.trades[0]["reason"] == "stop"


def test_eod_flatten_at_1555():
    bars = make_day(full_day(BASE, last=(102.2, 102.3, 102.0, 102.1)))
    bot, sim = new_bot(bars, tempfile.mkdtemp())
    drive(bot, sim, bars)
    t = sim.trades[0]
    assert t["reason"] == "close" and t["exit_px"] == 102.2 and sim.position_qty("SIM") == 0


def test_daily_loss_limit_flattens_and_halts():
    rows = BASE[:4] + [(101.6, 101.7, 100.0, 100.2)] + [(100.2, 103.0, 100.0, 102.9)] * 3  # drawdown, no stop hit
    bars = make_day(rows)
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d, risk=Risk(max_daily_loss_frac=0.003))
    drive(bot, sim, bars)
    assert sim.position_qty("SIM") == 0 and len(sim.trades) == 1
    assert sim.trades[0]["reason"] == "close" and sim.trades[0]["pnl"] < 0
    assert json.loads((Path(d) / "state.json").read_text())["done_reason"] == "daily_loss_limit"


def test_kill_switch_flattens_and_stops():
    bars = make_day(full_day(BASE))
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d)
    drive(bot, sim, bars, until=bars.index[6])
    assert sim.position_qty("SIM") != 0
    (Path(d) / "KILL").write_text("")
    sim.set_now(bars.index[7])
    assert bot.step() == "KILL"
    assert sim.position_qty("SIM") == 0


def test_restart_adopts_open_position_and_does_not_reenter():
    bars = make_day(full_day(BASE, last=(102.2, 102.3, 102.0, 102.1)))
    d = tempfile.mkdtemp()
    bot1, sim = new_bot(bars, d)
    drive(bot1, sim, bars, until=bars.index[8])       # entered, then "crash"
    assert sim.position_qty("SIM") != 0
    bot2, _ = new_bot(bars, d, sim=sim)               # fresh process, same state dir & broker
    for t in list(bars.index[9:]) + [bars.index[-1] + BM]:
        sim.set_now(t)
        bot2.step()
    assert len(sim.trades) == 1 and sim.trades[0]["reason"] == "close" and sim.position_qty("SIM") == 0
    events = [json.loads(l)["event"] for l in (Path(d) / "events.jsonl").read_text().splitlines()]
    assert "adopted_position" in events


def test_restart_after_signal_is_not_chased():
    bars = make_day(full_day(BASE))
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d)
    for t in bars.index[8:12]:                        # bot only comes online late, after the breakout
        sim.set_now(t)
        bot.step()
    assert sim.trades == [] and sim.position_qty("SIM") == 0
    assert json.loads((Path(d) / "state.json").read_text())["done_reason"] == "missed_signal"


def test_unknown_position_at_start_of_day_is_flattened_and_day_skipped():
    bars = make_day(full_day(BASE))
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d)
    sim.set_now(bars.index[0])
    sim.seed_position(1, 50, 100.0)
    bot.step()
    assert sim.position_qty("SIM") == 0
    assert json.loads((Path(d) / "state.json").read_text())["done_reason"] == "unexpected_position_at_open"
    drive(bot, sim, bars)                              # rest of day: stays out even though a breakout occurs
    assert len(sim.trades) == 1                        # only the flatten of the unknown position


def test_position_size_respects_caps():
    bars = make_day(full_day(BASE, last=(102.2, 102.3, 102.0, 102.1)))
    bot, sim = new_bot(bars, tempfile.mkdtemp(), risk=Risk(max_shares=7))
    drive(bot, sim, bars)
    assert sim.trades[0]["qty"] == 7
    cfg = Config(or_minutes=15, max_leverage=1.0, **ZERO)           # leverage cap binds: 100k / 101.5
    bot, sim = new_bot(bars, tempfile.mkdtemp(), cfg=cfg)
    drive(bot, sim, bars)
    assert sim.trades[0]["qty"] <= 100_000 // 101.5


# ------------------------------------------------------------------ failure paths
class FlakyBroker(SimBroker):
    """Breaks the equity() call after `fail_after` successes, to simulate an API outage."""
    def __init__(self, *a, fail_after=10**9, **k):
        super().__init__(*a, **k)
        self.fail_after, self.calls = fail_after, 0

    def equity(self):
        self.calls += 1
        if self.calls > self.fail_after:
            raise ConnectionError("API down")
        return super().equity()


def test_error_limit_flattens_and_exits_run_loop():
    bars = make_day(full_day(BASE))
    sim = FlakyBroker(bars, 100_000.0, fail_after=6)
    cfg = Config(or_minutes=15, max_leverage=10.0, **ZERO)
    bot = Bot(cfg, Risk(max_consecutive_errors=3, flatten_on_exit=True), sim, SimBars(sim), "SIM",
              tempfile.mkdtemp(), sleep=lambda s: None)
    ticks = iter(list(bars.index) * 3)

    def advancing_sleep(_s):          # each loop iteration, move simulated time forward one bar
        try:
            sim.set_now(next(ticks))
        except StopIteration:
            raise AssertionError("run loop did not exit")
    bot.sleep = advancing_sleep
    sim.set_now(bars.index[0])
    bot.run()                          # must return by itself once errors hit the limit
    assert sim.position_qty("SIM") == 0


def test_shutdown_flattens_open_position():
    bars = make_day(full_day(BASE))
    bot, sim = new_bot(bars, tempfile.mkdtemp())
    drive(bot, sim, bars, until=bars.index[8])
    assert sim.position_qty("SIM") != 0
    bot.shutdown()
    assert sim.position_qty("SIM") == 0 and bot.st.done_reason == "shutdown_flatten"


def test_early_close_day_is_skipped():
    bars = make_day(full_day(BASE))
    bot, sim = new_bot(bars, tempfile.mkdtemp())
    orig = sim.clock
    def half_day():
        c = orig()
        c.next_close = c.next_close.replace(hour=13)
        return c
    sim.clock = half_day
    drive(bot, sim, bars)
    assert sim.trades == [] and bot.st.done_reason == "early_close_day"


def test_state_file_blocks_second_entry_after_restart_following_exit():
    rows = BASE + [(101.9, 102.0, 98.5, 99.0)] + [QUIET] * 72
    bars = make_day(rows)
    d = tempfile.mkdtemp()
    bot1, sim = new_bot(bars, d)
    drive(bot1, sim, bars, until=bars.index[7])       # stopped out, then process "restarts"
    bot2, _ = new_bot(bars, d, sim=sim)
    for t in list(bars.index[8:]) + [bars.index[-1] + BM]:
        sim.set_now(t)
        bot2.step()
    assert len(sim.trades) == 1


def test_state_save_survives_transient_windows_file_lock():
    import os as _os
    bars = make_day(full_day(BASE))
    d = tempfile.mkdtemp()
    bot, sim = new_bot(bars, d)
    sim.set_now(bars.index[0])
    bot.step()                                   # creates today's state
    real, calls = _os.replace, {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(5, "Access is denied")
        return real(src, dst)
    _os.replace = flaky_replace
    try:
        bot._save()
    finally:
        _os.replace = real
    assert calls["n"] == 4 and json.loads((Path(d) / "state.json").read_text())["date"] == "2024-03-04"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
