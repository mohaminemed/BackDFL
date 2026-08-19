import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Tuple
import numpy as np

try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

from ..fl.baseserver import FedAvgAggregator


class SAGAServer(FedAvgAggregator):
    """
    SAGA-v2: Adaptive GuArd defense hardened against optimized-trigger backdoor
    attacks (A3FL, IBA) and colluding/sybil clients.

    Key changes vs. v1:
      (A) Directional (cosine) anomaly check added alongside norm/drift check,
          on both last-layer and full-model vectors. Norm-bounded-but-direction-
          anomalous updates (IBA's stealth objective) are now caught.
      (B) Cross-client clustering (HDBSCAN if available, else a robust
          MAD-based fallback) to catch colluding/sybil clients that look
          individually "normal" relative to their own history.
      (C) Cumulative-drift cap per client, closing the "slow boil" loophole in
          the persistent-baseline mechanism where an attacker ramps up
          poisoning strength in small per-round steps.
      (D) Fallback selection no longer prefers the *smallest*-anomaly clients
          (which is exactly what an optimized attack minimizes). Instead it
          falls back to the clustering majority, and if that's still
          insufficient, the round is skipped (global model unchanged) rather
          than force-selecting suspicious clients.
      (E) Post-selection robust aggregation: per-client update norm-clipping to
          the accepted-set median, trimmed averaging, and small Gaussian noise
          on the final aggregate (weak differential-privacy-style bound on any
          residual malicious contribution).
      (F) Adaptive threshold now uses an EMA instead of a monotonically
          shrinking min(), so it can track legitimate distribution shifts in
          both directions and isn't a fully predictable static target.
      (G) Light per-round randomization (which layers count as "last layers",
          small jitter on thresholds) to deny an adaptive/white-box attacker
          (A3FL) a fixed differentiable target across rounds.
      (H) Optional behavioral validation hook: if a testloader is available,
          the tentative aggregate is sanity-checked against a clean-accuracy
          drop before being committed; otherwise this step is skipped.
    """

    def __init__(
        self,
        model: nn.Module,
        testloader: nn.Module = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, testloader, device)
        self.config = config if config is not None else {}

        # --- Parameters (original) ---
        self.tau = float(self.config.get("saga_tau", 1.0))
        self.sigma = float(self.config.get("saga_sigma", 0.0))
        self.adaptive_threshold = float(self.config.get("adaptive_threshold", 1e9))
        self.rel_change_limit = float(self.config.get("rel_change_limit", 1.0))

        # --- Round bookkeeping ---
        self.current_round: int = int(self.config.get("round", 0))
        self.total_rounds: int = int(self.config.get("num_rounds", 100))
        self.prev_diff_norms: Dict[int, float] = {}

        # --- Persistent change tracking (per-client) ---
        self.change_baselines: Dict[int, Dict[str, float]] = {}
        self.max_persist_rounds = int(self.config.get("max_persist_rounds", 5))
        self.require_consecutive_ok = bool(self.config.get("require_consecutive_ok", True))
        self.consecutive_ok_needed = int(self.config.get("consecutive_ok_needed", 2))
        self._eps = float(self.config.get("eps", 1e-8))

        # --- Adaptive rel-change ---
        self.decay_floor = float(self.config.get("decay_floor", 0.5))

        # --- Weighted aggregation ---
        self.weighted = bool(self.config.get("saga_weighted", False))

        # ================= NEW: directional check =================
        self.cos_mad_tau = float(self.config.get("cos_mad_tau", 1.0))
        self.min_accept_cos_last = float(self.config.get("min_accept_cos_last", -1.0))  # set via MAD below
        self.min_accept_cos_full = float(self.config.get("min_accept_cos_full", -1.0))

        # ================= NEW: cumulative drift cap =================
        self.cumulative_drift: Dict[int, float] = {}
        self.cumulative_drift_cap = float(self.config.get("cumulative_drift_cap", 1e9))
        self.cumulative_cap_auto = bool(self.config.get("cumulative_cap_auto", True))
        self.cumulative_cap_mult = float(self.config.get("cumulative_cap_mult", 6.0))

        # ================= NEW: clustering =================
        self.use_clustering = bool(self.config.get("saga_use_clustering", True))
        self.cluster_min_size_frac = float(self.config.get("cluster_min_size_frac", 0.25))

        # ================= NEW: robust aggregation =================
        self.clip_to_median_norm = bool(self.config.get("clip_to_median_norm", True))
        self.dp_sigma = float(self.config.get("saga_dp_sigma", 0.0))  # 0 disables noise
        self.trim_frac = float(self.config.get("saga_trim_frac", 0.0))  # coordinate-wise trimmed mean fraction

        # ================= NEW: threshold EMA =================
        self.threshold_ema_beta = float(self.config.get("threshold_ema_beta", 0.9))

        # ================= NEW: randomization for anti-adaptivity =================
        self.randomize_layers = bool(self.config.get("saga_randomize_layers", True))
        self.jitter_frac = float(self.config.get("saga_jitter_frac", 0.05))
        self._rng = np.random.default_rng(int(self.config.get("saga_seed", 1234)))

        # ================= NEW: behavioral validation =================
        self.behavioral_check = bool(self.config.get("saga_behavioral_check", False))
        self.max_clean_acc_drop = float(self.config.get("max_clean_acc_drop", 0.10))
        self._last_clean_acc: Optional[float] = None

        # --- Drift logging for visualization ---
        self.log_full_l2 = []
        self.log_last_l2 = []
        self.log_full_cos = []
        self.log_last_cos = []
        self.log_selected = []       # NEW: per-round selected indices
        self.log_cluster_labels = [] # NEW: per-round cluster labels

        print(
            f"Initialized SAGAServer-v2 (tau={self.tau}, weighted={self.weighted}, "
            f"adaptive_threshold={self.adaptive_threshold}, clustering={self.use_clustering}, "
            f"dp_sigma={self.dp_sigma})"
        )

    # -------------------- Utilities --------------------
    @staticmethod
    def _flatten_state_dict_to_vector(
        state: Dict[str, torch.Tensor], keys_subset: Optional[List[str]] = None
    ) -> torch.Tensor:
        keys = sorted(list(state.keys()))
        if keys_subset is not None:
            keys = [k for k in keys if k in keys_subset]
        return torch.cat([state[k].detach().cpu().flatten() for k in keys], dim=0)

    def _get_last_layers(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """
        Return names of the last parameter layers. When randomize_layers is on,
        occasionally widen the scope (last 2-4 params) so the defense's
        "field of view" isn't a fixed, learnable target for an adaptive attacker.
        """
        layer_names = list(state_dict.keys())
        param_layers = [name for name in layer_names if "weight" in name or "bias" in name]
        n = 2
        if self.randomize_layers and len(param_layers) > 2:
            n = int(self._rng.integers(2, min(5, len(param_layers) + 1)))
        return param_layers[-n:]

    def _effective_threshold(self) -> float:
        gamma = float(self.current_round ) / max(1.0, float(self.total_rounds ))
        base = self.adaptive_threshold + (1.0 - gamma) * (abs(self.adaptive_threshold) * self.sigma)
        if self.jitter_frac > 0:
            jitter = 1.0 + float(self._rng.uniform(-self.jitter_frac, self.jitter_frac))
            base *= jitter
        return base

    def _update_rel_change_limit(self, rel_changes_current_round: List[float]) -> None:
        if len(rel_changes_current_round) == 0:
            return
        vals = sorted(rel_changes_current_round)
        n = len(vals)
        median = vals[n // 2] if (n % 2 == 1) else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        abs_dev = sorted([abs(v - median) for v in vals])
        mad = abs_dev[n // 2] if (n % 2 == 1) else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
        tau = float(self.config.get("mad_tau", 1.0))
        mad = max(mad, self._eps)
        rel_limit = float(median + tau * mad)

        gamma = float(self.current_round ) / max(1.0, float(self.total_rounds ))
        decay = (1 - gamma) * (1 - self.decay_floor) + self.decay_floor
        self.rel_change_limit = decay * rel_limit

    def _cosine(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8))

    @staticmethod
    def _median_mad(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-8
        return med, mad

    # -------------------- NEW: directional threshold from MAD --------------------
    def _cosine_thresholds(self, cos_vals: List[float]) -> float:
        """
        Robust lower-bound on acceptable cosine similarity: median - cos_mad_tau*MAD,
        clipped to [-1, 1). Lower cosine similarity = more directionally anomalous.
        """
        med, mad = self._median_mad(cos_vals)
        lower = med - self.cos_mad_tau * mad
        return float(np.clip(lower, -1.0, 0.999))

    # -------------------- NEW: cross-client clustering --------------------
    def _cluster_clients(self, vectors: List[torch.Tensor]) -> np.ndarray:
        """
        Cluster clients by cosine-normalized update direction to detect
        colluding / sybil groups submitting near-identical malicious updates.
        Returns an array of labels (majority-cluster label vs everyone else),
        with -1 meaning "not in the trusted majority cluster".
        """
        n = len(vectors)
        if n < 4:
            # too few clients to cluster meaningfully; trust everyone here
            return np.zeros(n, dtype=int)

        mat = torch.stack(vectors, dim=0)
        norms = torch.norm(mat, dim=1, keepdim=True) + 1e-8
        mat_normed = (mat / norms).numpy()

        min_cluster_size = max(2, int(self.cluster_min_size_frac * n))

        if _HAS_HDBSCAN:
            try:
                labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine").fit_predict(mat_normed)
                valid = labels[labels >= 0]
                if len(valid) == 0:
                    return np.zeros(n, dtype=int)  # degenerate: trust everyone, other checks still apply
                majority_label = np.bincount(valid).argmax()
                out = np.where(labels == majority_label, 0, -1)
                return out
            except Exception:
                pass  # fall through to MAD-based fallback

        # --- Fallback clustering without sklearn: distance-to-centroid MAD test ---
        centroid = mat_normed.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        cos_to_centroid = mat_normed @ centroid
        med, mad = self._median_mad(cos_to_centroid.tolist())
        lower = med - 2.0 * mad
        out = np.where(cos_to_centroid >= lower, 0, -1)
        return out

    # -------------------- NEW: safe robust aggregation --------------------
    def _robust_aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        selected_states: List[Dict[str, torch.Tensor]],
        selected_lens: List[int],
        selected_diff_norms: List[float],
    ) -> Dict[str, torch.Tensor]:
        first = selected_states[0]
        median_norm = float(np.median(selected_diff_norms)) if len(selected_diff_norms) > 0 else 0.0

        # 1) Norm-clip each client's delta to the median accepted drift
        clipped_states: List[Dict[str, torch.Tensor]] = []
        for st, dnorm in zip(selected_states, selected_diff_norms):
            if self.clip_to_median_norm and dnorm > self._eps:
                scale = min(1.0, median_norm / dnorm)
            else:
                scale = 1.0
            clipped = {}
            for k in first.keys():
                delta = st[k].detach().cpu().to(torch.float32) - global_state[k].detach().cpu().to(torch.float32)
                clipped[k] = global_state[k].detach().cpu().to(torch.float32) + delta * scale
            clipped_states.append(clipped)

        averaged: Dict[str, torch.Tensor] = {}
        if self.weighted:
            total = float(sum(selected_lens))
            for k in first.keys():
                acc = torch.zeros_like(first[k], dtype=torch.float32)
                for st, l in zip(clipped_states, selected_lens):
                    acc += st[k] * (float(l) / total)
                averaged[k] = acc
        else:
            for k in first.keys():
                stacked = torch.stack([st[k] for st in clipped_states], dim=0)
                if self.trim_frac > 0 and stacked.shape[0] > 2:
                    k_trim = int(self.trim_frac * stacked.shape[0])
                    if k_trim > 0:
                        sorted_stacked, _ = torch.sort(stacked, dim=0)
                        stacked = sorted_stacked[k_trim: stacked.shape[0] - k_trim]
                averaged[k] = torch.mean(stacked, dim=0)

        # 2) Weak-DP noise on the final aggregate
        if self.dp_sigma > 0:
            for k in averaged.keys():
                averaged[k] = averaged[k] + torch.randn_like(averaged[k]) * self.dp_sigma

        # cast back to original dtypes
        for k in averaged.keys():
            averaged[k] = averaged[k].type(first[k].dtype)

        return averaged

    # -------------------- NEW: optional behavioral validation --------------------
    def _behavioral_ok(self, candidate_state: Dict[str, torch.Tensor]) -> bool:
        """
        If a clean testloader is available, sanity-check the tentative aggregate
        against a clean-accuracy regression before committing it. This is the
        only check in the pipeline that looks at *behavior* rather than
        parameter-space statistics, which matters because A3FL/IBA optimize
        specifically to look benign in parameter space.
        """
        if not self.behavioral_check or self.testloader is None:
            return True
        try:
            backup = self.get_params()
            self.set_params({k: v.to(self.device) for k, v in candidate_state.items()})
            acc = self.evaluate(self.testloader) if hasattr(self, "evaluate") else None
            self.set_params(backup)  # restore; caller commits explicitly
            if acc is None:
                return True  # can't evaluate, don't block on it
            if self._last_clean_acc is not None and (self._last_clean_acc - acc) > self.max_clean_acc_drop:
                print(f"[SAGA][Behavioral] REJECT aggregate: clean acc dropped "
                      f"{self._last_clean_acc:.4f} -> {acc:.4f}")
                return False
            self._last_clean_acc = acc
            return True
        except Exception as e:
            print(f"[SAGA][Behavioral] check skipped due to error: {e}")
            return True

    # -------------------- Aggregation --------------------
    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        num_updates = len(self.received_params)
        if num_updates == 0:
            return self.get_params()

        self.current_round = current_round + 30
        global_state = self.get_params()

        last_layer_names = self._get_last_layers(global_state)
        ref_vec = self._flatten_state_dict_to_vector(global_state, keys_subset=last_layer_names)
        full_ref_vec = self._flatten_state_dict_to_vector(global_state)

        diff_norms: List[float] = []
        full_l2_round, last_l2_round = [], []
        full_cos_round, last_cos_round = [], []
        client_last_vecs: List[torch.Tensor] = []
        client_full_vecs: List[torch.Tensor] = []

        for i, st in enumerate(self.received_params):
            client_vec = self._flatten_state_dict_to_vector(st, keys_subset=last_layer_names)
            full_client_vec = self._flatten_state_dict_to_vector(st)

            last_l2 = torch.linalg.norm(ref_vec - client_vec)
            full_l2 = torch.linalg.norm(full_ref_vec - full_client_vec)

            last_cos = self._cosine(ref_vec, client_vec)
            full_cos = self._cosine(full_ref_vec, full_client_vec)

            diff_norms.append(float(last_l2))
            last_l2_round.append(float(last_l2))
            full_l2_round.append(float(full_l2))
            last_cos_round.append(last_cos)
            full_cos_round.append(full_cos)

            client_last_vecs.append(client_vec)
            client_full_vecs.append(full_client_vec)

            print(f"[SAGA][DriftLog] client{i} fullL2={float(full_l2):.4f}, lastL2={float(last_l2):.4f} "
                  f"| cos(full)={full_cos:.4f}, cos(last)={last_cos:.4f}")

        self.log_full_l2.append(full_l2_round)
        self.log_last_l2.append(last_l2_round)
        self.log_full_cos.append(full_cos_round)
        self.log_last_cos.append(last_cos_round)

        print(f"[SAGA][DriftLog] fullL2(avg)={np.mean(full_l2_round):.4f}, lastL2(avg)={np.mean(last_l2_round):.4f} "
              f"| cos(full)={np.mean(full_cos_round):.4f}, cos(last)={np.mean(last_cos_round):.4f}")

        # --- (F) EMA adaptive threshold (bidirectional, not monotonic) ---
        med, mad = self._median_mad(diff_norms)
        target_threshold = med + self.tau * mad
        if self.current_round == 0:
            self.adaptive_threshold = target_threshold
        else:
            self.adaptive_threshold = (
                self.threshold_ema_beta * self.adaptive_threshold
                + (1 - self.threshold_ema_beta) * target_threshold
            )
        effective_threshold = self._effective_threshold()

        # --- (A) Directional thresholds ---
        cos_last_floor = self._cosine_thresholds(last_cos_round)
        cos_full_floor = self._cosine_thresholds(full_cos_round)

        # --- Per-client relative changes (for persistence + fallback signal) ---
        rel_changes: List[float] = []
        for i, diff in enumerate(diff_norms):
            prev = self.prev_diff_norms.get(i)
            if prev is None:
                rel_est = abs(diff - med) / (min(diff, med) + self._eps)
                rel_changes.append(rel_est)
            else:
                rel = abs(diff - prev) / (min(prev, diff) + self._eps)
                rel_changes.append(rel)
        self._update_rel_change_limit(rel_changes)

        # --- (C) Auto-calibrate cumulative drift cap off current round stats ---
        if self.cumulative_cap_auto:
            self.cumulative_drift_cap = max(self.cumulative_drift_cap, self.cumulative_cap_mult * (med + mad))

        selected_indices: List[int] = []
        log_reasons: Dict[int, str] = {}

        for i, diff in enumerate(diff_norms):
            baseline_entry = self.change_baselines.get(i)
            prev = self.prev_diff_norms.get(i)

            # 1) Drift threshold check (norm)
            if diff > effective_threshold:
                log_reasons[i] = (
                    f"REJECT (drift={diff:.6f} > effective_threshold={effective_threshold:.6f})"
                )
                continue

            # 1b) NEW: Directional check (last-layer and full-model)
            if last_cos_round[i] < cos_last_floor or full_cos_round[i] < cos_full_floor:
                log_reasons[i] = (
                    f"REJECT (DIRECTIONAL: cos_last={last_cos_round[i]:.4f} (floor {cos_last_floor:.4f}), "
                    f"cos_full={full_cos_round[i]:.4f} (floor {cos_full_floor:.4f}))"
                )
                continue

            # 1c) NEW: Cumulative drift cap
            cum = self.cumulative_drift.get(i, 0.0)
            step = abs(diff - prev) if prev is not None else 0.0
            if cum + step > self.cumulative_drift_cap:
                log_reasons[i] = (
                    f"REJECT (CUMULATIVE_DRIFT: {cum + step:.6f} > cap={self.cumulative_drift_cap:.6f})"
                )
                # do not update cumulative_drift on rejection
                continue

            # 2) Persistent-change check (unchanged logic, now gated behind directional+cumulative checks)
            if baseline_entry is not None:
                baseline = float(baseline_entry["baseline"])
                age = int(self.current_round - baseline_entry["since_round"])
                rel_to_baseline = abs(diff - baseline) / (min(diff, baseline) + self._eps)
                eff_limit = self.rel_change_limit
                if rel_to_baseline > eff_limit:
                    log_reasons[i] = (
                        f"REJECT (PERSISTED_CHANGE: rel_to_baseline={rel_to_baseline:.6f} > limit={eff_limit:.6f})"
                    )
                    baseline_entry["consec_ok"] = 0
                    if age >= self.max_persist_rounds:
                        del self.change_baselines[i]
                        log_reasons[i] += f" | baseline aged-out after {age} rounds, released."
                    continue
                else:
                    if self.require_consecutive_ok:
                        baseline_entry["consec_ok"] = int(baseline_entry.get("consec_ok", 0)) + 1
                        if baseline_entry["consec_ok"] >= self.consecutive_ok_needed:
                            selected_indices.append(i)
                            log_reasons[i] = f"ACCEPT (persisted baseline matched, released)"
                            self.prev_diff_norms[i] = diff
                            self.cumulative_drift[i] = cum + step
                            del self.change_baselines[i]
                        else:
                            selected_indices.append(i)
                            log_reasons[i] = (
                                f"ACCEPT (persisted baseline matched; consec_ok="
                                f"{baseline_entry['consec_ok']}/{self.consecutive_ok_needed})"
                            )
                            self.prev_diff_norms[i] = diff
                            self.cumulative_drift[i] = cum + step
                        continue
                    else:
                        selected_indices.append(i)
                        log_reasons[i] = "ACCEPT (persisted baseline matched, released)"
                        self.prev_diff_norms[i] = diff
                        self.cumulative_drift[i] = cum + step
                        del self.change_baselines[i]
                        continue

            # 3) Normal relative-change check
            if prev is not None:
                rel_change = abs(diff - prev) / (min(prev, diff) + self._eps)
                eff_limit = self.rel_change_limit
                if rel_change > eff_limit:
                    direction = "JUMP" if diff > prev else "DROP"
                    self.change_baselines[i] = {
                        "baseline": float(prev), "since_round": int(self.current_round), "consec_ok": 0
                    }
                    log_reasons[i] = (
                        f"REJECT ({direction}: rel_change={rel_change:.6f} > limit={eff_limit:.6f}) "
                        f"-> persisted baseline={prev:.6f}"
                    )
                    continue
            else:
                log_reasons[i] = f"1ST Round prev=N/A, curr={diff:.6f}"

            # 4) Accept
            selected_indices.append(i)
            log_reasons[i] = f"ACCEPT (diff={diff:.6f}, prev={prev if prev is not None else 'N/A'})"
            self.prev_diff_norms[i] = float(diff)
            self.cumulative_drift[i] = cum + step

        # --- (B) Cross-client clustering: intersect with per-client survivors ---
        cluster_labels = np.zeros(len(diff_norms), dtype=int) - 1  # default: not evaluated
        if self.use_clustering and len(diff_norms) >= 4:
            cluster_labels = self._cluster_clients(client_last_vecs)
            trusted_cluster = set(np.where(cluster_labels == 0)[0].tolist())
            pre_cluster_selected = list(selected_indices)
            selected_indices = [i for i in selected_indices if i in trusted_cluster]
            dropped_by_cluster = set(pre_cluster_selected) - set(selected_indices)
            for i in dropped_by_cluster:
                log_reasons[i] = log_reasons.get(i, "") + " | REJECT (outside majority cluster)"
        self.log_cluster_labels.append(cluster_labels.tolist())

        # --- (D) Safe fallback: never favor lowest-anomaly clients ---
        min_k = max(2, int(0.3 * len(diff_norms)))
        if len(selected_indices) < min_k:
            print(f"[SAGA] Primary+cluster selection insufficient ({len(selected_indices)} < {min_k}).")
            if self.use_clustering and len(diff_norms) >= 4:
                trusted_cluster = set(np.where(cluster_labels == 0)[0].tolist())
                if len(trusted_cluster) >= min_k:
                    selected_indices = sorted(trusted_cluster)
                    print(f"[SAGA] Fallback: using majority cluster ({len(selected_indices)} clients).")
                else:
                    print("[SAGA] Fallback: majority cluster also insufficient. "
                          "Skipping round (keeping previous global model).")
                    self.received_params = []
                    self.received_lens = []
                    return {k: v.cpu().clone() for k, v in global_state.items()}
            else:
                print("[SAGA] Fallback: clustering unavailable/skipped and too few survivors. "
                      "Skipping round (keeping previous global model).")
                self.received_params = []
                self.received_lens = []
                return {k: v.cpu().clone() for k, v in global_state.items()}

        self.log_selected.append(list(selected_indices))

        # --- Robust aggregation over selected clients ---
        selected_states = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]
        selected_diff_norms = [diff_norms[i] for i in selected_indices]

        averaged = self._robust_aggregate(global_state, selected_states, selected_lens, selected_diff_norms)

        # --- (H) Optional behavioral validation before committing ---
        if not self._behavioral_ok(averaged):
            print("[SAGA] Behavioral check failed; keeping previous global model this round.")
            self.received_params = []
            self.received_lens = []
            return {k: v.cpu().clone() for k, v in global_state.items()}

        # --- Commit ---
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_params = []
        self.received_lens = []

        print(f"[SAGA] Selected {len(selected_indices)}/{len(diff_norms)} updates (round {current_round})")
        return {k: v.cpu().clone() for k, v in averaged.items()}