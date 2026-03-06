"""
====================================================
Phase 12 — Adaptive Execution Engine (VWAP / TWAP)
====================================================
Slices large orders into smaller child orders timed
against an intraday volume curve to minimise market
impact.

Strategies
----------
* **VWAP** — Volume-Weighted Average Price: slices
  proportional to historical intraday volume curve.
* **TWAP** — Time-Weighted Average Price: equal-size
  slices at fixed intervals.
* **AGGRESSIVE** — Front-loads 60 % of shares in the
  first 3 slices (for high-conviction / momentum).
* **PASSIVE** — Back-loads slices toward close when
  volumes naturally rise.

Key functions used by sniper.py
-------------------------------
* ``create_execution_plan()``  — build slice schedule
* ``execute_next_slice()``     — fire one child order
* ``should_slice_order()``     — decide slice vs simple
* ``get_executor_status()``    — dashboard summary
* ``save_executor_state() / load_executor_state()``
====================================================
"""

import os, sys, json, math, time
from datetime import datetime, timedelta
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STATE_FILE = os.path.join(DATA_DIR, "adaptive_executor_state.json")

# ── Defaults ─────────────────────────────────────
MIN_QTY_FOR_SLICING   = 20          # orders < this go as single shot
MAX_SLICES            = 12          # cap child orders
DEFAULT_STRATEGY      = "VWAP"     # VWAP | TWAP | AGGRESSIVE | PASSIVE
SLICE_INTERVAL_SEC    = 120         # 2-min gap between slices
URGENCY_THRESHOLD     = 0.85        # confidence above this ⇒ AGGRESSIVE

# ── Synthetic intraday volume curve (NSE typical) ─
# 26 bins of 15 min each from 09:15 → 15:30
_VOLUME_CURVE_RAW = [
    8.5, 6.0, 5.0, 4.5, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8,
    2.8, 3.0, 3.2, 3.5, 3.2, 3.0, 2.8, 3.0, 3.5, 4.0,
    4.5, 5.0, 5.5, 6.0, 7.0, 8.0,
]
_VOLUME_CURVE = np.array(_VOLUME_CURVE_RAW, dtype=float)
_VOLUME_CURVE /= _VOLUME_CURVE.sum()          # normalise

# Runtime state
_plans: dict = {}   # symbol → plan dict


# --------------------------------------------------
#  Volume curve helpers
# --------------------------------------------------
def _current_bin() -> int:
    """Return the 15-min bin index (0-25) for the current IST time."""
    now = datetime.now(IST)
    mins_since_open = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    return max(0, min(25, mins_since_open // 15))


def _remaining_curve(start_bin: int) -> np.ndarray:
    """Return normalised volume weights from *start_bin* to close."""
    curve = _VOLUME_CURVE[start_bin:].copy()
    s = curve.sum()
    return curve / s if s > 0 else np.ones_like(curve) / len(curve)


# --------------------------------------------------
#  Plan builder
# --------------------------------------------------
def should_slice_order(qty: int, confidence: float = 0.5) -> bool:
    """Return True if the order is large enough to benefit from slicing."""
    if qty < MIN_QTY_FOR_SLICING:
        return False
    return True


def create_execution_plan(
    symbol: str,
    total_qty: int,
    price: float,
    strategy: str = DEFAULT_STRATEGY,
    confidence: float = 0.5,
    max_slices: int = MAX_SLICES,
) -> dict:
    """
    Build a child-order schedule.

    Returns
    -------
    dict with keys: symbol, total_qty, strategy, slices (list of dicts),
    status, created_at, filled_qty, remaining_qty.
    """
    if confidence >= URGENCY_THRESHOLD and strategy == "VWAP":
        strategy = "AGGRESSIVE"

    now = datetime.now(IST)
    n_slices = min(max_slices, max(2, total_qty // 5))

    slices = []
    if strategy == "TWAP":
        base_qty = total_qty // n_slices
        remainder = total_qty - base_qty * n_slices
        for i in range(n_slices):
            q = base_qty + (1 if i < remainder else 0)
            slices.append({
                "index": i,
                "qty": q,
                "status": "PENDING",
                "scheduled_at": (now + timedelta(seconds=SLICE_INTERVAL_SEC * i)).isoformat(),
                "filled_at": None,
                "fill_price": None,
            })

    elif strategy == "AGGRESSIVE":
        # 60 % front-loaded into first 3 slices
        front = max(1, int(total_qty * 0.60))
        back = total_qty - front
        front_n = min(3, n_slices)
        back_n = max(1, n_slices - front_n)
        fq = front // front_n
        bq = back // back_n
        f_rem = front - fq * front_n
        b_rem = back - bq * back_n
        idx = 0
        for i in range(front_n):
            q = fq + (1 if i < f_rem else 0)
            slices.append({
                "index": idx, "qty": q, "status": "PENDING",
                "scheduled_at": (now + timedelta(seconds=SLICE_INTERVAL_SEC * idx)).isoformat(),
                "filled_at": None, "fill_price": None,
            })
            idx += 1
        for i in range(back_n):
            q = bq + (1 if i < b_rem else 0)
            slices.append({
                "index": idx, "qty": q, "status": "PENDING",
                "scheduled_at": (now + timedelta(seconds=SLICE_INTERVAL_SEC * idx)).isoformat(),
                "filled_at": None, "fill_price": None,
            })
            idx += 1

    elif strategy == "PASSIVE":
        # Back-load toward close
        curve = _remaining_curve(_current_bin())
        weights = curve[-n_slices:] if len(curve) >= n_slices else np.ones(n_slices) / n_slices
        weights = weights / weights.sum()
        allocated = np.round(weights * total_qty).astype(int)
        diff = total_qty - int(allocated.sum())
        allocated[-1] += diff
        for i in range(n_slices):
            slices.append({
                "index": i, "qty": int(allocated[i]), "status": "PENDING",
                "scheduled_at": (now + timedelta(seconds=SLICE_INTERVAL_SEC * i)).isoformat(),
                "filled_at": None, "fill_price": None,
            })

    else:  # VWAP
        curve = _remaining_curve(_current_bin())
        n = min(n_slices, len(curve))
        weights = curve[:n]
        weights = weights / weights.sum()
        allocated = np.round(weights * total_qty).astype(int)
        diff = total_qty - int(allocated.sum())
        allocated[-1] += diff
        for i in range(n):
            slices.append({
                "index": i, "qty": int(allocated[i]), "status": "PENDING",
                "scheduled_at": (now + timedelta(seconds=SLICE_INTERVAL_SEC * i)).isoformat(),
                "filled_at": None, "fill_price": None,
            })

    # Ensure no zero-qty slices
    slices = [s for s in slices if s["qty"] > 0]

    plan = {
        "symbol": symbol,
        "total_qty": total_qty,
        "strategy": strategy,
        "price_at_plan": price,
        "slices": slices,
        "status": "ACTIVE",
        "created_at": now.isoformat(),
        "filled_qty": 0,
        "remaining_qty": total_qty,
        "avg_fill_price": 0.0,
        "slippage_bps": 0.0,
    }
    _plans[symbol] = plan
    return plan


# --------------------------------------------------
#  Slice execution
# --------------------------------------------------
def execute_next_slice(symbol: str, current_price: float) -> dict:
    """
    Fire the next PENDING slice for *symbol*.

    Returns dict: {fired: bool, qty, price, slices_remaining, plan_complete}
    """
    plan = _plans.get(symbol)
    if plan is None or plan["status"] != "ACTIVE":
        return {"fired": False, "reason": "no_active_plan"}

    now = datetime.now(IST)
    for sl in plan["slices"]:
        if sl["status"] != "PENDING":
            continue
        # Check scheduled time
        sched = datetime.fromisoformat(sl["scheduled_at"])
        if now < sched:
            return {"fired": False, "reason": "waiting", "next_at": sl["scheduled_at"]}

        # Fill this slice
        sl["status"] = "FILLED"
        sl["filled_at"] = now.isoformat()
        sl["fill_price"] = current_price
        plan["filled_qty"] += sl["qty"]
        plan["remaining_qty"] -= sl["qty"]

        # Recalc avg fill price
        filled = [s for s in plan["slices"] if s["status"] == "FILLED"]
        total_cost = sum(s["qty"] * s["fill_price"] for s in filled)
        plan["avg_fill_price"] = round(total_cost / plan["filled_qty"], 2) if plan["filled_qty"] > 0 else 0

        # Slippage vs plan price
        if plan["price_at_plan"] > 0:
            plan["slippage_bps"] = round(
                (plan["avg_fill_price"] - plan["price_at_plan"]) / plan["price_at_plan"] * 10000, 2
            )

        # Check if plan complete
        remaining = sum(1 for s in plan["slices"] if s["status"] == "PENDING")
        if remaining == 0:
            plan["status"] = "COMPLETED"

        return {
            "fired": True,
            "qty": sl["qty"],
            "price": current_price,
            "slices_remaining": remaining,
            "plan_complete": remaining == 0,
            "avg_fill": plan["avg_fill_price"],
            "slippage_bps": plan["slippage_bps"],
        }

    # All slices filled
    plan["status"] = "COMPLETED"
    return {"fired": False, "reason": "all_filled", "plan_complete": True}


def cancel_plan(symbol: str) -> dict:
    """Cancel an active execution plan."""
    plan = _plans.get(symbol)
    if plan is None:
        return {"cancelled": False, "reason": "no_plan"}
    plan["status"] = "CANCELLED"
    for sl in plan["slices"]:
        if sl["status"] == "PENDING":
            sl["status"] = "CANCELLED"
    return {"cancelled": True, "filled_qty": plan["filled_qty"]}


# --------------------------------------------------
#  Status / state
# --------------------------------------------------
def get_executor_status() -> dict:
    """Dashboard-friendly summary of all execution plans."""
    active = {k: v for k, v in _plans.items() if v["status"] == "ACTIVE"}
    completed = {k: v for k, v in _plans.items() if v["status"] == "COMPLETED"}
    total_slip = []
    for p in completed.values():
        if p.get("slippage_bps") is not None:
            total_slip.append(p["slippage_bps"])

    return {
        "active_plans": len(active),
        "completed_plans": len(completed),
        "avg_slippage_bps": round(float(np.mean(total_slip)), 2) if total_slip else 0.0,
        "plans": {k: _plan_summary(v) for k, v in _plans.items()},
        "timestamp": datetime.now(IST).isoformat(),
    }


def _plan_summary(plan: dict) -> dict:
    """Compact summary for one plan."""
    return {
        "strategy": plan["strategy"],
        "total_qty": plan["total_qty"],
        "filled_qty": plan["filled_qty"],
        "remaining_qty": plan["remaining_qty"],
        "avg_fill_price": plan["avg_fill_price"],
        "slippage_bps": plan["slippage_bps"],
        "status": plan["status"],
        "n_slices": len(plan["slices"]),
        "slices_filled": sum(1 for s in plan["slices"] if s["status"] == "FILLED"),
    }


def save_executor_state(data: dict = None):
    """Persist execution state to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = data if data is not None else get_executor_status()
    with open(STATE_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_executor_state() -> dict:
    """Load saved execution state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
