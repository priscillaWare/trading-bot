"""
Simulated broker + bar source for rehearsing/testing orb_live.Bot without network access.

Fill model (matches the backtest so the two can be compared trade by trade):
  * Market orders fill at the OPEN of the bar starting at the current time, plus adverse slippage.
  * A resting stop/target is evaluated against each bar's high/low once that bar COMPLETES,
    stop first if both are touched; gaps through the level fill at the open.
The bot only ever sees bars whose end <= now, so there is no look-ahead in this harness either.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from orb_backtest import CLOSE_MIN, OPEN_MIN, infer_bar_minutes
from orb_live import BarSource, Broker, Clock


class SimBroker(Broker):
    def __init__(self, bars: pd.DataFrame, start_equity: float = 100_000.0,
                 slippage_bps: float = 0.0, commission_per_share: float = 0.0):
        self.bars = bars
        self.bm = pd.Timedelta(minutes=infer_bar_minutes(bars))
        self.days = {d: g for d, g in bars.groupby(bars.index.date)}
        self.realized = start_equity
        self.slip, self.comm = slippage_bps / 1e4, commission_per_share
        self.now: Optional[pd.Timestamp] = None
        self.pos: Optional[dict] = None
        self.trades: list = []
        self.order_state: dict = {}
        self.last_close: Optional[float] = None
        self._processed: Optional[pd.Timestamp] = None

    # ---- time
    def set_now(self, t: pd.Timestamp) -> None:
        self.now = t
        day = self.days.get(t.date())
        if day is None:
            return
        todo = day[(day.index + self.bm <= t) & ((day.index > self._processed) if self._processed is not None else True)]
        for ts, row in todo.iterrows():
            self._process_bar(ts, row)
            self._processed = ts

    def _bar_open_now(self) -> float:
        day = self.days.get(self.now.date())
        if day is not None and self.now in day.index:
            return float(day.loc[self.now, "open"])
        return float(self.last_close)

    def _process_bar(self, ts, row) -> None:
        self.last_close = float(row["close"])
        p = self.pos
        if not p or (p["stop"] is None and p["target"] is None):
            return
        side, o, h, l = p["side"], float(row["open"]), float(row["high"]), float(row["low"])
        px, reason = None, None
        if side == 1:
            if p["stop"] is not None and l <= p["stop"]:
                px, reason = min(p["stop"], o), "stop"
            elif p["target"] is not None and h >= p["target"]:
                px, reason = max(p["target"], o), "target"
        else:
            if p["stop"] is not None and h >= p["stop"]:
                px, reason = max(p["stop"], o), "stop"
            elif p["target"] is not None and l <= p["target"]:
                px, reason = min(p["target"], o), "target"
        if px is not None:
            self._exit(px, reason, ts + self.bm)

    def _exit(self, raw_px: float, reason: str, when) -> None:
        p = self.pos
        px = raw_px * (1 - p["side"] * self.slip)
        gross = p["side"] * (px - p["entry_px"]) * p["qty"]
        self.realized += gross - self.comm * p["qty"]            # entry commission was charged at entry
        self.trades.append({"date": p["entry_time"].date(), "side": p["side"], "entry_time": p["entry_time"],
                            "exit_time": when, "qty": p["qty"], "entry_px": p["entry_px"], "exit_px": px,
                            "reason": reason, "pnl": gross - 2 * self.comm * p["qty"]})
        self.pos = None

    def seed_position(self, side: int, qty: int, px: float) -> None:
        """Test helper: pretend an unknown position already exists."""
        self.pos = {"side": side, "qty": qty, "entry_px": px, "stop": None, "target": None,
                    "entry_time": self.now, "cid": "seed"}

    # ---- Broker interface
    def clock(self) -> Clock:
        t = self.now
        m = t.hour * 60 + t.minute
        is_open = t.date() in self.days and OPEN_MIN <= m < CLOSE_MIN
        nc = pd.Timestamp(t.date()).tz_localize(t.tz) + pd.Timedelta(minutes=CLOSE_MIN)
        return Clock(is_open, t, None, nc)

    def equity(self) -> float:
        eq = self.realized
        if self.pos:
            mark = self.last_close if self.last_close is not None else self.pos["entry_px"]
            eq += self.pos["side"] * (mark - self.pos["entry_px"]) * self.pos["qty"]
        return eq

    def position_qty(self, symbol: str) -> int:
        return self.pos["side"] * self.pos["qty"] if self.pos else 0

    def open_orders(self, symbol: str) -> list:
        p = self.pos
        return [{"id": "stop"}] if p and p["stop"] is not None else []

    def order_status(self, client_order_id: str) -> str:
        return self.order_state.get(client_order_id, "missing")

    def submit_entry(self, symbol, side, qty, stop_price, take_profit, client_order_id) -> str:
        if client_order_id in self.order_state:
            raise RuntimeError("duplicate client_order_id")
        if self.pos:
            raise RuntimeError("position already open")
        px = self._bar_open_now() * (1 + side * self.slip)
        self.realized -= self.comm * qty
        self.pos = {"side": side, "qty": qty, "entry_px": px, "stop": stop_price, "target": take_profit,
                    "entry_time": self.now, "cid": client_order_id}
        self.order_state[client_order_id] = "filled"
        return client_order_id

    def cancel_open_orders(self, symbol: str) -> None:
        if self.pos:
            self.pos["stop"] = self.pos["target"] = None

    def close_position(self, symbol: str) -> None:
        if self.pos:
            self._exit(self._bar_open_now(), "close", self.now)

    def fills_today(self, symbol: str) -> list:
        return [t for t in self.trades if t["date"] == self.now.date()]


class SimBars(BarSource):
    def __init__(self, sim: SimBroker):
        self.sim = sim

    def completed(self, symbol: str, now: pd.Timestamp) -> pd.DataFrame:
        day = self.sim.days.get(now.date())
        if day is None:
            return pd.DataFrame()
        return day[day.index + self.sim.bm <= now]
