"""Common detection metrics (TPR / FPR) for every defense.

Ground truth: a sender is *malicious* when its id is in ``malicious_ids`` AND the attack is
active in that round; every other sender is benign.

* TPR (true malicious detection rate) = rejected malicious / malicious received
* FPR (false rejection rate)          = rejected benign    / benign received

A defense only has to call ``self.report_selection(accepted_positions)`` (positions in
``received_params``) from ``aggregate``; the client then calls ``DetectionLogger.log_round``.
Counts are accumulated per (defense, round) and per (defense, defender) so metrics can be
computed over any slice offline from the JSONL file; running totals are also logged.
"""
import json
import logging
import os
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("saga.detection")


def _rate(num: int, den: int) -> float:
    return float(num) / den if den > 0 else float("nan")


def compute_counts(sender_ids: Sequence[int], accepted_positions: Iterable[int],
                   malicious_ids: Iterable[int], attack_active: bool = True) -> Dict[str, object]:
    """Confusion counts for one aggregation. Positions index into ``sender_ids``."""
    accepted = set(int(i) for i in accepted_positions)
    mal = set(int(i) for i in malicious_ids) if attack_active else set()
    tp = fp = tn = fn = 0
    rejected_ids: List[int] = []
    for pos, sid in enumerate(sender_ids):
        is_mal = int(sid) in mal
        rejected = pos not in accepted
        if rejected:
            rejected_ids.append(int(sid))
        if is_mal and rejected:
            tp += 1
        elif is_mal:
            fn += 1
        elif rejected:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_malicious": tp + fn, "n_benign": fp + tn,
            "tpr": _rate(tp, tp + fn), "fpr": _rate(fp, fp + tn),
            "rejected_ids": rejected_ids}


class DetectionLogger:
    """Process-wide accumulator + JSONL writer (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.path: Optional[str] = None
        self.totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    def configure(self, path: Optional[str]) -> None:
        with self._lock:
            self.path = path
            if path:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log_round(self, defense: str, defender_id: int, round_idx: int,
                  sender_ids: Sequence[int], accepted_positions: Iterable[int],
                  malicious_ids: Iterable[int], attack_active: bool = True) -> Dict[str, object]:
        c = compute_counts(sender_ids, accepted_positions, malicious_ids, attack_active)
        with self._lock:
            tot = self.totals[defense]
            for k in ("tp", "fp", "tn", "fn"):
                tot[k] += c[k]
            cum_tpr = _rate(tot["tp"], tot["tp"] + tot["fn"])
            cum_fpr = _rate(tot["fp"], tot["fp"] + tot["tn"])
            rec = {"defense": defense, "round": int(round_idx), "defender": int(defender_id),
                   "attack_active": bool(attack_active), "senders": [int(s) for s in sender_ids],
                   **c, "cum_tpr": cum_tpr, "cum_fpr": cum_fpr}
            if self.path:
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
        logger.info("[detect] %s round=%d defender=%d TP=%d FN=%d FP=%d TN=%d | TPR=%.3f FPR=%.3f | cum TPR=%.3f FPR=%.3f",
                    defense, round_idx, defender_id, c["tp"], c["fn"], c["fp"], c["tn"],
                    c["tpr"], c["fpr"], cum_tpr, cum_fpr)
        return rec

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {d: {**t, "tpr": _rate(t["tp"], t["tp"] + t["fn"]),
                        "fpr": _rate(t["fp"], t["fp"] + t["tn"])} for d, t in self.totals.items()}

    def log_summary(self) -> None:
        for d, s in self.summary().items():
            logger.info("[detect] SUMMARY %s: TPR=%.4f FPR=%.4f (TP=%d FN=%d FP=%d TN=%d)",
                        d, s["tpr"], s["fpr"], s["tp"], s["fn"], s["fp"], s["tn"])


detection_logger = DetectionLogger()


def setup_detection_logging(config: Dict, results_dir: Optional[str] = None) -> None:
    """Attach a console handler and set the JSONL path (config['detection_log_path'] or results_dir)."""
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    path = config.get("detection_log_path")
    if not path:
        base = results_dir or config.get("results_dir") or "experiments/statistics"
        path = os.path.join(base, f"detection_{config.get('attack', 'none')}_{config.get('defense', 'none')}.jsonl")
    detection_logger.configure(path)
