"""Timing / overhead logging for aggregation (defense cost) and attacks (attack overhead).

Phases recorded per round (seconds, ``time.perf_counter``):
  train            local training of every client, tagged benign / attacker
  stealth          adaptive-stealth post-processing of the attackers' updates
  attacker_agg     aggregation performed by attacker clients
  defense_agg      ``agg_server.aggregate()`` of every benign client (the defense cost)
  agg_total        wall-clock of the whole aggregation phase of the round

Attack overhead of a round, derived in ``end_round``:
  attack_train_overhead_s = mean(attacker train) - mean(benign train)   (extra local cost of crafting the update)
  attack_total_overhead_s = attack_train_overhead_s * n_attackers + stealth + attacker_agg
When the attack is inactive, attacker training is a normal benign step and no overhead is reported.

One JSON line per round is appended to ``timing_log_path``; a SUMMARY line is logged at the end.
"""
import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("saga.timing")


def _mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else float("nan")


class TimingLogger:
    def __init__(self):
        self._lock = threading.Lock()
        self.path: Optional[str] = None
        self._ev: Dict[str, List[float]] = defaultdict(list)
        self.rounds: List[Dict[str, object]] = []

    def configure(self, path: Optional[str]) -> None:
        with self._lock:
            self.path = path
            self._ev.clear()
            self.rounds = []
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, phase: str, seconds: float, role: Optional[str] = None) -> None:
        key = f"{phase}:{role}" if role else phase
        with self._lock:
            self._ev[key].append(float(seconds))

    @contextmanager
    def measure(self, phase: str, role: Optional[str] = None):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(phase, time.perf_counter() - t0, role)

    def end_round(self, round_idx: int, attack_active: bool, **wall: float) -> Dict[str, object]:
        """Close the round: derive statistics, log one line, append one JSONL record."""
        with self._lock:
            ev = {k: list(v) for k, v in self._ev.items()}
            self._ev.clear()
        tb, ta = ev.get("train:benign", []), ev.get("train:attacker", [])
        da = ev.get("defense_agg", [])
        rec: Dict[str, object] = {
            "round": int(round_idx), "attack_active": bool(attack_active),
            "n_benign": len(tb), "n_attackers": len(ta),
            "train_benign_mean_s": _mean(tb), "train_attacker_mean_s": _mean(ta),
            "stealth_s": float(sum(ev.get("stealth", []))),
            "attacker_agg_s": float(sum(ev.get("attacker_agg", []))),
            "defense_agg_calls": len(da), "defense_agg_mean_s": _mean(da),
            "defense_agg_max_s": float(max(da)) if da else float("nan"),
            "defense_agg_sum_s": float(sum(da)),
            **{f"{k}_s": float(v) for k, v in wall.items()},
        }
        if attack_active and ta and tb:
            train_ovh = max(0.0, _mean(ta) - _mean(tb))
            rec["attack_train_overhead_s"] = train_ovh
            rec["attack_total_overhead_s"] = train_ovh * len(ta) + rec["stealth_s"] + rec["attacker_agg_s"]
        else:
            rec["attack_train_overhead_s"] = 0.0
            rec["attack_total_overhead_s"] = 0.0
        with self._lock:
            self.rounds.append(rec)
            if self.path:
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        logger.info("[timing] round=%d | defense agg: mean=%.3fs max=%.3fs sum=%.3fs (%d calls) | "
                    "attack overhead: train=+%.3fs total=%.3fs (stealth=%.3fs, atk-agg=%.3fs) | train benign=%.3fs",
                    round_idx, rec["defense_agg_mean_s"], rec["defense_agg_max_s"], rec["defense_agg_sum_s"],
                    len(da), rec["attack_train_overhead_s"], rec["attack_total_overhead_s"],
                    rec["stealth_s"], rec["attacker_agg_s"], rec["train_benign_mean_s"])
        return rec

    def log_summary(self) -> None:
        with self._lock:
            rs = list(self.rounds)
        if not rs:
            return
        act = [r for r in rs if r["attack_active"]]
        logger.info("[timing] SUMMARY over %d rounds: defense agg per-call mean=%.3fs, per-round sum=%.3fs | "
                    "attack overhead over %d attack rounds: mean total=%.3fs/round (train=+%.3fs)",
                    len(rs), np.nanmean([r["defense_agg_mean_s"] for r in rs]),
                    np.nanmean([r["defense_agg_sum_s"] for r in rs]), len(act),
                    _mean([r["attack_total_overhead_s"] for r in act]),
                    _mean([r["attack_train_overhead_s"] for r in act]))


timing_logger = TimingLogger()


def setup_timing_logging(config: Dict, results_dir: Optional[str] = None) -> None:
    """Console handler + JSONL path (config['timing_log_path'] or <results_dir>/timing_<attack>_<defense>.jsonl)."""
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    path = config.get("timing_log_path")
    if not path:
        base = results_dir or config.get("results_dir") or "experiments/statistics"
        path = os.path.join(base, f"timing_{config.get('attack', 'none')}_{config.get('defense', 'none')}.jsonl")
    timing_logger.configure(path)
