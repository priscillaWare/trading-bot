#!/usr/bin/env python3
"""
Paper-trading execution layer for the ORB strategy in orb_backtest.py.

PAPER ONLY BY DESIGN: the Alpaca adapter hard-codes paper=True and there is no flag to change it.
Use a dedicated paper account: the bot cancels all open orders in its symbol.

Architecture
  OrbEngine   pure signal logic, no I/O (same rules as the backtest; parity-tested)
  Bot         one `step()` per poll: safety checks -> reconcile -> monitor -> scan -> enter/flatten
  Broker      interface; AlpacaBroker (paper) and SimBroker (sim_broker.py) implement it
  BarSource   interface; AlpacaBars and SimBars implement it

Safety layers
  * Kill switch: create the file named by --kill-file (default KILL in the state dir) -> flatten & exit.
  * Daily loss limit (mark-to-market, default 2% of start-of-day equity) -> flatten & halt for the day.
  * Hard caps: max shares, max leverage, one trade per day (persisted, survives restarts).
  * Protective stop is a BROKER-SIDE order attached to the entry (OTO/bracket), so a crashed
    bot is still protected intraday.
  * Wall-clock EOD flatten (does not depend on data arriving), verified and retried.
  * Restart reconciliation: state file vs broker positions/orders. Unknown position -> flatten and halt.
  * Idempotent entry (deterministic client_order_id) so a retry can't double-enter.
  * Late-start rule: if a signal fired while the bot was down, the trade is SKIPPED, never chased.
  * Early-close days (e.g. day after Thanksgiving) are skipped.
  * Consecutive-error limit: flatten and exit.
  * On exit/crash the bot flattens by default (the stop order expires at the close, so an open
    position would otherwise be unprotected overnight).

Known differences from the backtest (the paper results are the real test of these):
  * Position size is computed from the signal bar's close (the fill price isn't known yet).
  * Stops are live orders triggered by ticks, so they can fill at worse prices than the backtest's
    bar-level model; targets are limit orders and may not fill on a mere touch.
  * Data latency can delay entries slightly; signals older than --signal-grace seconds are skipped.

Usage
  export ALPACA_API_KEY=... ALPACA_API_SECRET=...
  python orb_live.py paper --symbol SPY --or-minutes 15
  python orb_live.py sim spy_5min.csv        # rehearse the whole bot on history (no network, no orders)
  touch bot_state/KILL                       # emergency stop

STATUS: Bot, engine and SimBroker are tested (see test_orb_live.py). AlpacaBroker/AlpacaBars were
written against the alpaca-py SDK but are UNTESTED against the live service: first run during market
hours with a tiny size, watch the logs, and verify orders in the Alpaca dashboard.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import pandas as pd

from orb_backtest import CLOSE_MIN, OPEN_MIN, TZ, Config, _hhmm, load_csv, prepare

log = logging.getLogger("orb_live")


# --------------------------------------------------------------------------- config & interfaces
@dataclass(frozen=True)
class Risk:
    max_daily_loss_frac: float = 0.02   # of start-of-day equity, mark-to-market
    max_shares: int = 5000
    max_consecutive_errors: int = 5
    signal_grace_seconds: int = 120     # skip signals older than this (late data / restart)
    poll_seconds: int = 5
    skip_early_close: bool = True
    flatten_on_exit: bool = True


@dataclass
class Clock:
    is_open: bool
    now: pd.Timestamp
    next_open: Optional[pd.Timestamp] = None
    next_close: Optional[pd.Timestamp] = None


class Broker(ABC):
    @abstractmethod
    def clock(self) -> Clock: ...
    @abstractmethod
    def equity(self) -> float: ...
    @abstractmethod
    def position_qty(self, symbol: str) -> int:
        """Signed: +long, -short, 0 flat."""
    @abstractmethod
    def open_orders(self, symbol: str) -> list: ...
    @abstractmethod
    def order_status(self, client_order_id: str) -> str:
        """One of: 'filled', 'open', 'failed', 'missing'."""
    @abstractmethod
    def submit_entry(self, symbol: str, side: int, qty: int, stop_price: float,
                     take_profit: Optional[float], client_order_id: str) -> str: ...
    @abstractmethod
    def cancel_open_orders(self, symbol: str) -> None: ...
    @abstractmethod
    def close_position(self, symbol: str) -> None: ...
    @abstractmethod
    def fills_today(self, symbol: str) -> list: ...


class BarSource(ABC):
    @abstractmethod
    def completed(self, symbol: str, now: pd.Timestamp) -> pd.DataFrame:
        """Today's COMPLETED regular-session bars (start-time index, US/Eastern, open/high/low/close)."""


# --------------------------------------------------------------------------- pure signal engine
@dataclass
class Signal:
    side: int
    stop: float
    or_hi: float
    or_lo: float
    close: float


class OrbEngine:
    """Feed it completed bars in order. Mirrors orb_backtest._simulate_day's entry rules exactly."""

    def __init__(self, cfg: Config, bar_min: int):
        self.cfg, self.bar_min = cfg, bar_min
        self.n_needed = cfg.or_minutes // bar_min
        self.or_end = OPEN_MIN + cfg.or_minutes
        self.cutoff = _hhmm(cfg.entry_cutoff)
        self.or_hi, self.or_lo, self.n_or = -math.inf, math.inf, 0
        self.finished, self.reason = False, ""

    def _finish(self, reason: str) -> None:
        self.finished, self.reason = True, reason

    def on_bar(self, ts: pd.Timestamp, h: float, l: float, c: float, tradable: bool) -> Optional[Signal]:
        if self.finished:
            return None
        m = ts.hour * 60 + ts.minute
        cfg = self.cfg
        if m < self.or_end:  # still building the opening range
            if m != OPEN_MIN + self.n_or * self.bar_min:
                self._finish("incomplete_or")  # missing/late first bar or a gap inside the OR
                return None
            self.or_hi, self.or_lo, self.n_or = max(self.or_hi, h), min(self.or_lo, l), self.n_or + 1
            return None
        if self.n_or != self.n_needed:
            self._finish("incomplete_or")
            return None
        if self.or_hi - self.or_lo <= 0:
            self._finish("zero_range")
            return None
        if m + self.bar_min >= self.cutoff:  # the entry would be at/after the cutoff
            self._finish("past_entry_cutoff")
            return None
        side = 0
        if c > self.or_hi and cfg.allow_long:
            side = 1
        elif c < self.or_lo and cfg.allow_short:
            side = -1
        if not side:
            return None
        if not tradable:
            self._finish("missed_signal")  # fired while we weren't watching live: never chase
            return None
        rng = self.or_hi - self.or_lo
        stop = self.or_hi - cfg.stop_frac * rng if side == 1 else self.or_lo + cfg.stop_frac * rng
        self._finish("signal")
        return Signal(side, stop, self.or_hi, self.or_lo, c)


# --------------------------------------------------------------------------- bot
@dataclass
class DayState:
    date: str
    stage: str = "scanning"   # scanning | entering | open | done
    start_equity: float = 0.0
    done_reason: str = ""
    side: int = 0
    qty: int = 0
    stop: float = 0.0
    target: Optional[float] = None
    client_order_id: str = ""


class Bot:
    def __init__(self, cfg: Config, risk: Risk, broker: Broker, bars: BarSource, symbol: str,
                 state_dir: str, bar_minutes: int = 5, kill_file: Optional[str] = None, sleep=time.sleep):
        self.cfg, self.risk, self.broker, self.bars, self.symbol = cfg, risk, broker, bars, symbol
        self.bar_min, self.sleep = bar_minutes, sleep
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.kill_file = Path(kill_file) if kill_file else self.dir / "KILL"
        self.state_path, self.events_path = self.dir / "state.json", self.dir / "events.jsonl"
        self.st: Optional[DayState] = None
        self.engine = OrbEngine(cfg, bar_minutes)
        self.last_bar_start: Optional[pd.Timestamp] = None
        self.next_check: Optional[pd.Timestamp] = None
        self._last_heartbeat = None

    # ---- persistence / logging
    def _save(self) -> None:
        """Atomic write, tolerant of brief file locks (Windows antivirus/sync tools can hold the file)."""
        data = json.dumps(asdict(self.st))
        tmp = self.state_path.with_suffix(".tmp")
        for attempt in range(8):
            try:
                tmp.write_text(data)
                os.replace(tmp, self.state_path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))   # real delay on purpose: the lock needs wall-clock time to clear
        try:                                        # last resort: non-atomic direct write
            self.state_path.write_text(data)
        except OSError:
            log.critical("could not save state file %s: restart protection is degraded", self.state_path)

    def _load(self) -> Optional[DayState]:
        if not self.state_path.exists():
            return None
        try:
            d = json.loads(self.state_path.read_text())
            return DayState(**{k: v for k, v in d.items() if k in {f.name for f in fields(DayState)}})
        except Exception:
            log.exception("could not read state file, ignoring it")
            return None

    def _event(self, name: str, **kw) -> None:
        rec = {"ts": pd.Timestamp.now(tz=TZ).isoformat(), "event": name, "symbol": self.symbol, **kw}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log.info("%s %s", name, kw)

    # ---- actions
    def _flatten(self, reason: str) -> bool:
        log.warning("FLATTEN (%s)", reason)
        try:
            self.broker.cancel_open_orders(self.symbol)
        except Exception:
            log.exception("cancel_open_orders failed")
        for _ in range(6):
            try:
                if self.broker.position_qty(self.symbol) == 0:
                    return True
                self.broker.close_position(self.symbol)
            except Exception:
                log.exception("flatten attempt failed")
            self.sleep(1.0)
        try:
            ok = self.broker.position_qty(self.symbol) == 0
        except Exception:
            ok = False
        if not ok:
            log.critical("COULD NOT CONFIRM FLAT in %s: CHECK THE BROKER DASHBOARD NOW", self.symbol)
        return ok

    def _finish(self, reason: str) -> None:
        st = self.st
        st.stage, st.done_reason = "done", reason
        try:
            self.broker.cancel_open_orders(self.symbol)
        except Exception:
            log.exception("cancel at finish failed")
        try:
            eq, fills = self.broker.equity(), self.broker.fills_today(self.symbol)
        except Exception:
            eq, fills = float("nan"), []
        self._event("day_done", reason=reason, equity_start=st.start_equity, equity_end=eq,
                    day_pnl=eq - st.start_equity, fills=fills)
        self._save()

    def shutdown(self) -> None:
        """Called on Ctrl-C / crash. Flatten (the broker-side stop expires at the close)."""
        if self.st and self.st.stage in ("entering", "open") and self.risk.flatten_on_exit:
            self._flatten("shutdown")
            self._finish("shutdown_flatten")

    # ---- day lifecycle
    def _early_close(self, clock: Clock) -> bool:
        nc = clock.next_close
        return bool(self.risk.skip_early_close and nc is not None and nc.hour * 60 + nc.minute != CLOSE_MIN)

    def _ensure_day(self, clock: Clock) -> None:
        today = str(clock.now.date())
        if self.st is not None and self.st.date == today:
            return
        self.engine = OrbEngine(self.cfg, self.bar_min)
        self.last_bar_start = self.next_check = None
        saved = self._load()
        if saved and saved.date == today:
            self.st = saved
            self._event("restart_reconcile", stage=saved.stage)
            self._reconcile()
            return
        eq = self.broker.equity()
        self.st = DayState(date=today, start_equity=eq)
        self._event("day_start", equity=eq)
        if self._early_close(clock):
            self._finish("early_close_day")
        elif self.broker.position_qty(self.symbol) != 0:
            log.critical("unexpected open position at start of day: flattening")
            self._flatten("unexpected_position_at_open")
            self._finish("unexpected_position_at_open")
        self._save()

    def _reconcile(self) -> None:
        st, qty = self.st, self.broker.position_qty(self.symbol)
        if st.stage == "scanning" and qty != 0:
            self._flatten("unknown_position_on_restart")
            self._finish("unknown_position_on_restart")
        elif st.stage in ("entering", "open"):
            if qty != 0:
                st.stage, st.qty = "open", abs(qty)
                self._event("adopted_position", qty=qty)
            else:
                status = self.broker.order_status(st.client_order_id)
                if status == "filled":
                    self._finish("exited_while_down")
                elif status != "open":
                    self._finish("entry_failed")
        self._save()

    # ---- main step
    def step(self) -> str:
        if self.kill_file.exists():
            self._event("kill_switch")
            self._flatten("kill_file")
            if self.st is not None and self.st.stage != "done":
                self._finish("kill_switch")
            return "KILL"
        clock = self.broker.clock()
        if not clock.is_open:
            return "closed"
        self._ensure_day(clock)
        st, now = self.st, clock.now
        self._heartbeat(now)
        if st.stage == "done":
            return "done"

        eq = self.broker.equity()
        if st.start_equity > 0 and (eq - st.start_equity) / st.start_equity <= -self.risk.max_daily_loss_frac:
            self._flatten("daily_loss_limit")
            self._finish("daily_loss_limit")
            return "done"

        if st.stage in ("entering", "open"):
            self._monitor()
            if st.stage == "done":
                return "done"

        mins = now.hour * 60 + now.minute
        if mins >= _hhmm(self.cfg.flatten_time):
            if st.stage in ("entering", "open"):
                self._flatten("eod")
            self._finish("eod")
            return "done"

        if st.stage == "scanning":
            self._scan(now)
        return st.stage

    def _heartbeat(self, now: pd.Timestamp) -> None:
        if self._last_heartbeat is None or (now - self._last_heartbeat).total_seconds() >= 300:
            self._last_heartbeat = now
            log.info("heartbeat %s stage=%s", now.strftime("%H:%M:%S"), self.st.stage)

    def _monitor(self) -> None:
        st, qty = self.st, self.broker.position_qty(self.symbol)
        if st.stage == "entering":
            if qty != 0:
                st.stage, st.qty = "open", abs(qty)
                self._event("entry_filled", qty=qty)
            else:
                status = self.broker.order_status(st.client_order_id)
                if status == "filled":
                    self._finish("exited_by_broker_order")
                elif status != "open":
                    self._finish("entry_failed")
        elif st.stage == "open" and qty == 0:
            self._finish("exited_by_broker_order")  # protective stop or target filled
        self._save()

    def _scan(self, now: pd.Timestamp) -> None:
        if self.next_check is not None and now < self.next_check:
            return
        df = self.bars.completed(self.symbol, now)
        if df is None or df.empty:
            return
        if self.last_bar_start is not None:
            df = df[df.index > self.last_bar_start]
        bm = pd.Timedelta(minutes=self.bar_min)
        for ts, row in df.iterrows():
            tradable = (now - (ts + bm)).total_seconds() <= self.risk.signal_grace_seconds
            sig = self.engine.on_bar(ts, row["high"], row["low"], row["close"], tradable)
            self.last_bar_start = ts
            self.next_check = ts + 2 * bm
            if sig is not None:
                self._enter(sig)
                return
            if self.engine.finished:
                self._finish(self.engine.reason)
                return

    def _enter(self, sig: Signal) -> None:
        cfg, st = self.cfg, self.st
        equity = self.broker.equity()
        est = sig.close * (1 + sig.side * cfg.slippage_bps / 1e4)
        risk_ps = sig.side * (est - sig.stop)
        if risk_ps <= 0:
            self._finish("bad_risk")
            return
        qty = math.floor(min(equity * cfg.risk_frac / risk_ps, equity * cfg.max_leverage / est))
        qty = min(qty, self.risk.max_shares)
        if qty < 1:
            self._finish("size_zero")
            return
        stop = round(sig.stop, 2)
        target = round(est + sig.side * cfg.r_multiple * risk_ps, 2) if cfg.r_multiple else None
        cid = f"orb-{self.symbol}-{st.date}-entry"
        st.stage, st.side, st.qty, st.stop, st.target, st.client_order_id = "entering", sig.side, qty, stop, target, cid
        self._save()  # persist intent BEFORE sending, so a crash right after can be reconciled
        self._event("entry_submit", side=sig.side, qty=qty, stop=stop, target=target, est_entry=est)
        try:
            self.broker.submit_entry(self.symbol, sig.side, qty, stop, target, cid)
        except Exception:
            log.exception("entry submit failed")
            if self.broker.position_qty(self.symbol) == 0:
                self._finish("entry_submit_failed")
            return
        for _ in range(10):  # market orders normally fill within a second or two
            if self.broker.position_qty(self.symbol) != 0:
                st.stage = "open"
                self._event("entry_filled", qty=self.broker.position_qty(self.symbol))
                break
            self.sleep(0.5)
        self._save()

    # ---- runner
    def run(self) -> None:
        errors = 0
        try:
            while True:
                try:
                    r = self.step()
                    errors = 0
                    if r == "KILL":
                        return
                    self.sleep(30 if r == "closed" else self.risk.poll_seconds)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    errors += 1
                    log.exception("step failed (%d/%d)", errors, self.risk.max_consecutive_errors)
                    if errors >= self.risk.max_consecutive_errors:
                        log.critical("too many consecutive errors: flattening and exiting")
                        self._flatten("error_limit")
                        return
                    self.sleep(min(30, 2 ** errors))
        except KeyboardInterrupt:
            log.warning("interrupted")
        finally:
            self.shutdown()


# --------------------------------------------------------------------------- Alpaca (paper only) UNTESTED
def _norm_status(s) -> str:
    s = str(getattr(s, "value", s)).lower()
    if s == "filled":
        return "filled"
    if s in ("canceled", "cancelled", "rejected", "expired", "done_for_day", "suspended"):
        return "failed"
    return "open"


class AlpacaBroker(Broker):
    def __init__(self, key: str, secret: str):
        from alpaca.trading.client import TradingClient
        self.c = TradingClient(key, secret, paper=True)  # paper only. There is deliberately no live switch.

    def clock(self) -> Clock:
        c = self.c.get_clock()
        conv = lambda t: pd.Timestamp(t).tz_convert(TZ)
        return Clock(bool(c.is_open), conv(c.timestamp), conv(c.next_open), conv(c.next_close))

    def equity(self) -> float:
        return float(self.c.get_account().equity)

    def position_qty(self, symbol: str) -> int:
        for p in self.c.get_all_positions():
            if p.symbol == symbol:
                q = abs(int(float(p.qty)))
                return q if str(getattr(p.side, "value", p.side)).lower() == "long" else -q
        return 0

    def _open(self, symbol: str):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        return self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))

    def open_orders(self, symbol: str) -> list:
        return [{"id": str(o.id), "status": str(o.status)} for o in self._open(symbol)]

    def order_status(self, client_order_id: str) -> str:
        try:
            return _norm_status(self.c.get_order_by_client_id(client_order_id).status)
        except Exception:
            return "missing"

    def submit_entry(self, symbol, side, qty, stop_price, take_profit, client_order_id) -> str:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == 1 else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET if take_profit else OrderClass.OTO,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=take_profit) if take_profit else None,
            client_order_id=client_order_id)
        return str(self.c.submit_order(req).id)

    def cancel_open_orders(self, symbol: str) -> None:
        for o in self._open(symbol):
            try:
                self.c.cancel_order_by_id(o.id)
            except Exception:
                log.exception("cancel %s failed", o.id)

    def close_position(self, symbol: str) -> None:
        self.c.close_position(symbol)

    def fills_today(self, symbol: str) -> list:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        after = pd.Timestamp.now(tz=TZ).normalize().to_pydatetime()
        orders = self.c.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol],
                                                    after=after, nested=True, limit=50))
        out = []
        for o in orders:
            for x in [o, *(o.legs or [])]:
                if x.filled_qty and float(x.filled_qty) > 0:
                    out.append({"client_order_id": x.client_order_id, "side": str(getattr(x.side, "value", x.side)),
                                "qty": float(x.filled_qty), "price": float(x.filled_avg_price),
                                "filled_at": str(x.filled_at)})
        return out


class AlpacaBars(BarSource):
    def __init__(self, key: str, secret: str, minutes: int = 5, feed: str = "iex"):
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        self.client, self.minutes = StockHistoricalDataClient(key, secret), minutes
        self.feed = DataFeed.IEX if feed == "iex" else DataFeed.SIP

    def completed(self, symbol: str, now: pd.Timestamp) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        start = pd.Timestamp(now.date()).tz_localize(TZ) + pd.Timedelta(minutes=OPEN_MIN)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(self.minutes, TimeFrameUnit.Minute),
                               start=start.to_pydatetime(), end=now.to_pydatetime(), feed=self.feed)
        df = self.client.get_stock_bars(req).df
        if df.empty:
            return df
        if "symbol" in (df.index.names or []):
            df = df.xs(symbol, level="symbol")
        df = prepare(df[["open", "high", "low", "close"]])
        return df[df.index + pd.Timedelta(minutes=self.minutes) <= now]  # drop the still-forming bar


# --------------------------------------------------------------------------- simulation rehearsal
def run_sim(bars: pd.DataFrame, cfg: Config, risk: Risk, state_dir: str, symbol: str = "SIM"):
    """Drive the whole Bot against a simulated broker over historical bars. Returns the SimBroker."""
    from orb_backtest import infer_bar_minutes
    from sim_broker import SimBars, SimBroker
    bm = infer_bar_minutes(bars)
    sim = SimBroker(bars, cfg.start_equity, cfg.slippage_bps, cfg.commission_per_share)
    bot = Bot(cfg, risk, sim, SimBars(sim), symbol, state_dir, bar_minutes=bm, sleep=lambda s: None)
    for _, day in bars.groupby(bars.index.date):
        for t in list(day.index) + [day.index[-1] + pd.Timedelta(minutes=bm)]:
            sim.set_now(t)
            bot.step()
    return sim


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--or-minutes", type=int, default=15)
        sp.add_argument("--stop-frac", type=float, default=1.0)
        sp.add_argument("--r-multiple", type=float, default=None)
        sp.add_argument("--risk-frac", type=float, default=0.01)
        sp.add_argument("--max-leverage", type=float, default=1.0)
        sp.add_argument("--slippage-bps", type=float, default=1.0)
        sp.add_argument("--commission", type=float, default=0.005)
        sp.add_argument("--max-daily-loss", type=float, default=0.02)
        sp.add_argument("--max-shares", type=int, default=5000)
        sp.add_argument("--state-dir", default="bot_state")

    pp = sub.add_parser("paper", help="trade on an Alpaca PAPER account")
    pp.add_argument("--symbol", default="SPY")
    pp.add_argument("--minutes", type=int, default=5)
    pp.add_argument("--feed", choices=["iex", "sip"], default="iex")
    pp.add_argument("--kill-file", default=None)
    pp.add_argument("--no-flatten-on-exit", action="store_true")
    common(pp)
    ss = sub.add_parser("sim", help="rehearse the bot on historical CSV data with a simulated broker")
    ss.add_argument("csv")
    common(ss)
    a = p.parse_args(argv)

    cfg = Config(or_minutes=a.or_minutes, stop_frac=a.stop_frac, r_multiple=a.r_multiple, risk_frac=a.risk_frac,
                 max_leverage=a.max_leverage, slippage_bps=a.slippage_bps, commission_per_share=a.commission)
    risk = Risk(max_daily_loss_frac=a.max_daily_loss, max_shares=a.max_shares,
                flatten_on_exit=not getattr(a, "no_flatten_on_exit", False))
    Path(a.state_dir).mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(Path(a.state_dir) / "bot.log")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)

    if a.cmd == "sim":
        sim = run_sim(load_csv(a.csv), cfg, risk, a.state_dir)
        t = pd.DataFrame(sim.trades)
        print(f"\nSim finished: {len(t)} trades, net P&L {t.pnl.sum():.0f}" if len(t) else "\nNo trades")
        if len(t):
            print(t.reason.value_counts().to_dict())
        return

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET (paper-trading keys).")
    try:
        broker, feed = AlpacaBroker(key, secret), AlpacaBars(key, secret, a.minutes, a.feed)
    except ImportError:
        raise SystemExit("alpaca-py is not installed: pip install alpaca-py")
    bot = Bot(cfg, risk, broker, feed, a.symbol, a.state_dir, bar_minutes=a.minutes, kill_file=a.kill_file)
    log.info("starting PAPER bot: %s %s | kill switch file: %s", a.symbol, cfg, bot.kill_file)
    bot.run()


if __name__ == "__main__":
    main()
