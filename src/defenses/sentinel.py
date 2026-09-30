import copy
import random
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator


class SentinelServer(FedAvgAggregator):
    """
    Sentinel defense (Feng et al., ECAI 2024), matching the same
    set_params / receive_update / aggregate(current_round) interface as
    `BalanceServer`.

    Sentinel is a three-stage hybrid pipeline (paper Algorithm 1):
      1. Similarity filtering: drop neighbor updates whose layer-averaged
         cosine similarity to the local model falls below `tau_S`.
      2. Bootstrap validation: score the surviving updates by their loss
         on a small, freshly-sampled local bootstrap set, averaged
         against each neighbor's own loss history via a damped
         loss-distance mapping (Algorithm 4) to get an aggregation
         weight w_j in [0, 1].
      3. Layer normalization: rescale each surviving update layer-wise so
         its norm never exceeds the local model's own norm (rho <= 1,
         like FLTrust), guarding against stealth/scaled attacks that
         pass stages 1-2.
      The local model itself always enters the final weighted average
      with w_i = 1 (its own loss-distance to itself is 0).

    Persistence of loss history. Like `ArgusServer`'s trust state, this
    class's `self.loss_history` persists across rounds because the
    caller (`BenignClient`) constructs `agg_server` once and reuses it.
    Only the per-round reception buffers are cleared in `aggregate()`.


    The paper: Algorithm 2 specifies a "row-wise then
    layer-wise" average cosine similarity, but doesn't define what a
    "row" is for tensors of different rank (conv kernels, FC weight
    matrices, biases). This implementation computes one cosine
    similarity per parameter tensor (flattened) and averages across
    tensors -- the paper's coarser "layer-wise" granularity, without
    guessing at the unspecified row convention.

    Similarity threshold. The paper leaves tau_S an unspecified constant,
    but a fixed value is a poor fit across a training run: cosine
    similarity between honest neighbors' models is naturally low in
    early rounds (models diverge from random initialization) and rises
    as training converges, so a static threshold either over-filters
    early or stops discriminating at all late. Three variants are
    available via `sentinel_similarity_variant`:
      - 'static': fixed `sentinel_tau_s`.
      - 'median': tau_S = median cosine similarity among this round's
        neighbors. Self-calibrating, but rejects ~50% of neighbors every
        round *by construction*, regardless of how many attackers are
        actually present. With a low attacker ratio (e.g. ~15%), this
        discards far more honest information than necessary, hurting
        convergence and per-node consistency (low min-accuracy nodes)
        without additional security benefit -- if you see this pattern
        (low/controlled ASR but poor, uneven convergence), switch to
        'mad'.
      - 'mad' (default): tau_S = median - k * MAD(similarities), a
        robust outlier test rather than a fixed rejection quota. If
        neighbors are tightly clustered (no attackers, or attackers not
        distinguishable this round), MAD is small and the threshold sits
        just under the cluster, so few or no neighbors are rejected. If
        one or two neighbors sit clearly below the honest cluster, only
        those get excluded. This targets actual outliers instead of
        enforcing a fixed rejection rate, and is the recommended default.

    Config keys:
      - sentinel_similarity_variant (str, default 'mad'): 'static',
        'median', or 'mad' (see above). 'adaptive' is accepted as a
        backward-compatible alias for 'median'.
      - sentinel_tau_s (float, default 0.5): similarity filtering
        threshold, used only when sentinel_similarity_variant == 'static'.
        Not given a numeric value in the paper -- treat this default as
        a starting point to tune, not a reproduced setting.
      - sentinel_mad_k (float, default 2.0): number of MADs below the
        median a similarity must fall to be treated as an outlier, used
        only when sentinel_similarity_variant == 'mad'. Lower k rejects
        more aggressively; higher k is more permissive.
      - sentinel_tau_l (float, default 0.0): loss-distance threshold
        below which a weight is zeroed (Algorithm 4, line 5).
      - sentinel_l_min (float, default 1e-3): minimum average loss used
        in the damping factor, for numerical stability near zero loss.
      - sentinel_kappa_max (float, default None = uncapped, matching the
        paper): upper bound on the damping factor kappa = 1/max(l_i, l_min).
        As training converges, l_i (the cumulative mean of the local
        node's own bootstrap loss) keeps shrinking toward l_min, so
        kappa keeps growing -- late in training it can reach 1/l_min,
        making the weight w = exp(-kappa * d_l) collapse to ~0 for even
        tiny loss gaps d_l that are within normal bootstrap-sampling
        noise, not evidence of an attack. This shows up as sudden,
        intermittent min-accuracy crashes late in a run with no attacker
        present. Setting e.g. sentinel_kappa_max = 50.0 bounds how
        aggressive the damping can get without changing early-round
        behavior (where l_i is still far from l_min).
      - sentinel_weight_mix [UNUSED] (float, default 0.0 = paper-faithful): blends
        the loss-distance weighted average with a plain uniform average
        over the *same* accepted set (own model + Stage-1/2 survivors).
        0.0 uses the weights w_j from Algorithm 4 as-is; 1.0 ignores them
        entirely and just averages everyone who passed filtering equally
        (Stage 1/2 still determine *membership* -- this only changes how
        much each admitted member counts once accepted). 
      - sentinel_bootstrap_min (int, default 300), sentinel_bootstrap_frac: bootstrap set size is
        max(sentinel_bootstrap_frac * |val_dataset|, sentinel_bootstrap_min),
        freshly resampled every round (paper Section 3, Step 2).
      - sentinel_val_dataset: Here we provide a holdout dataset (25% of the training set) for evaluation.
        bootstrap samples are drawn fresh each round via random indices). Required.    
        """

    def __init__(self, model: nn.Module, testloader: nn.Module = None,
                 device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        self.similarity_variant = str(self.config.get('sentinel_similarity_variant', 'mad'))
        self.tau_s = float(self.config.get('sentinel_tau_s', 0.5))
        self.mad_k = float(self.config.get('sentinel_mad_k', 2.0))
        self.tau_l = float(self.config.get('sentinel_tau_l', 0.0))
        self.l_min = float(self.config.get('sentinel_l_min', 1e-3))
        self.kappa_max = self.config.get('sentinel_kappa_max', None)
        self.loss_window = int(self.config.get('sentinel_loss_window', 0))
        self.bootstrap_min = int(self.config.get('sentinel_bootstrap_min', 300))
        self.bootstrap_frac = float(self.config.get('sentinel_bootstrap_frac', 1.0 / 10.0))
        self.val_dataset = self.config.get('sentinel_val_dataset', None)

        # Per-round buffers (reset in aggregate()).
        self.received_params: List[Dict[str, torch.Tensor]] = []
        self.received_lens: List[int] = []
        self.received_ids: List[int] = []

        # PERSISTENT loss history -- survives across rounds since this
        # object itself is only constructed once per client.
        self.loss_history: Dict[int, List[float]] = {}
        self.own_loss_history: List[float] = []

        print(f"Initialized SentinelServer (similarity_variant={self.similarity_variant}, "
              f"tau_s={self.tau_s}, mad_k={self.mad_k}, tau_l={self.tau_l}, l_min={self.l_min}, "
              f"bootstrap_min={self.bootstrap_min}, bootstrap_frac={self.bootstrap_frac})")

    # -------------------- reception (round-scoped) --------------------

    def receive_update(self, weights: Dict[str, torch.Tensor], num_samples: int,
                        sender_id: Optional[int] = None) -> None:
        """See class docstring for the sender_id / positional-fallback caveat."""
        sid = sender_id if sender_id is not None else len(self.received_params)
        self.received_params.append(weights)
        self.received_lens.append(num_samples)
        self.received_ids.append(sid)

    # -------------------- bootstrap sampling --------------------

    def _sample_bootstrap_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freshly sample a bootstrap batch from the local validation
        dataset (paper Section 3, Step 2: resampled every round)."""
        if self.val_dataset is None:
            raise ValueError("SentinelServer requires `sentinel_val_dataset` in config.")
        n = len(self.val_dataset)
        size = min(n, max(int(self.bootstrap_frac * n), self.bootstrap_min))
        print(f"[SENTINEL] Sampling bootstrap batch of size {size} from val dataset of size {n}")
        indices = random.sample(range(n), size)
        xs, ys = zip(*(self.val_dataset[i] for i in indices))
        return torch.stack(xs).to(self.device), torch.tensor(ys).to(self.device)

    # -------------------- Stage 1: similarity filtering --------------------

    @staticmethod
    def _cosine_similarity_layerwise(state_a: Dict[str, torch.Tensor],
                                      state_b: Dict[str, torch.Tensor]) -> float:
        """Average cosine similarity across parameter tensors (Algorithm 2;
        see class docstring for the row/layer granularity simplification)."""
        sims = []
        for key in state_a.keys():
            va, vb = state_a[key].flatten().float(), state_b[key].flatten().float()
            sim = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            sims.append(sim)
        return sum(sims) / len(sims) if sims else 0.0

    # -------------------- Stage 2: bootstrap validation --------------------

    def _compute_bootstrap_loss(self, state: Dict[str, torch.Tensor],
                                 batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """Algorithm 3: cross-entropy loss of `state` on the bootstrap batch."""
        x, y = batch
        probe_model = copy.deepcopy(self.model).to(self.device)
        probe_model.load_state_dict(state)
        probe_model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(probe_model(x), y).item()
        return loss

    def _map_loss_distance(self, own_history: List[float], other_history: List[float]) -> float:
        """Algorithm 4: damped decay of the loss distance into a weight in [0, 1].
        See class docstring for sentinel_kappa_max / sentinel_loss_window,
        which bound how harsh this damping can get late in training."""
        own_window = own_history[-self.loss_window:] if self.loss_window > 0 else own_history
        other_window = other_history[-self.loss_window:] if self.loss_window > 0 else other_history
        l_i = sum(own_window) / len(own_window) if own_window else self.l_min
        l_j = sum(other_window) / len(other_window) if other_window else self.l_min
        kappa = 1.0 / max(l_i, self.l_min)
        if self.kappa_max is not None:
            kappa = min(kappa, float(self.kappa_max))
        d_l = max(l_j - l_i, 0.0)
        w = float(torch.exp(torch.tensor(-kappa * d_l)))
        if w < self.tau_l:
            w = 0.0
        return w

    # -------------------- Stage 3: layer normalization --------------------

    @staticmethod
    def _normalize_model(local_state: Dict[str, torch.Tensor],
                          neighbor_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Algorithm 5: scale each layer of `neighbor_state` down (never up)
        so its norm never exceeds the corresponding local layer's norm."""
        normalized = {}
        for key in neighbor_state.keys():
            local_norm = torch.linalg.norm(local_state[key].float())
            neighbor_norm = torch.linalg.norm(neighbor_state[key].float())
            rho = min(1.0, float(local_norm / (neighbor_norm + 1e-12)))
            normalized[key] = (rho * neighbor_state[key].float()).to(neighbor_state[key].dtype)
        return normalized

    # -------------------- aggregation --------------------

    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        own_state = self.get_params()
        num_updates = len(self.received_params)

        if num_updates == 0:
            print("SentinelServer.aggregate(): warning - no updates to aggregate")
            return own_state

        # -- Own bootstrap loss for this round (reference for Stage 2) --
        bootstrap_batch = self._sample_bootstrap_batch()
        own_loss = self._compute_bootstrap_loss(own_state, bootstrap_batch)
        self.own_loss_history.append(own_loss)

        # -- Stage 1: similarity filtering --
        similarities = [
            (sid, recv_state, self._cosine_similarity_layerwise(own_state, recv_state))
            for recv_state, sid in zip(self.received_params, self.received_ids)
        ]
        sims_only = [s for _, _, s in similarities]

        variant = 'median' if self.similarity_variant == 'adaptive' else self.similarity_variant
        if variant == 'mad':
            median_sim = float(torch.median(torch.tensor(sims_only)))
            abs_dev = torch.tensor([abs(s - median_sim) for s in sims_only])
            mad = float(torch.median(abs_dev))
            effective_tau_s = median_sim - self.mad_k * mad
            print(f"[SENTINEL] MAD similarity threshold this round: {effective_tau_s:.4f} "
                  f"(median={median_sim:.4f}, mad={mad:.4f}, k={self.mad_k}, "
                  f"{len(sims_only)} neighbors)")
        elif variant == 'median':
            effective_tau_s = float(torch.median(torch.tensor(sims_only)))
            print(f"[SENTINEL] Median similarity threshold this round: {effective_tau_s:.4f} "
                  f"(median of {len(sims_only)} neighbor similarities)")
        else:
            effective_tau_s = self.tau_s

        surviving: List[Tuple[int, Dict[str, torch.Tensor]]] = [
            (sid, recv_state) for sid, recv_state, sim in similarities if sim >= effective_tau_s
        ]

        if not surviving:
            print("[SENTINEL] All neighbors filtered by similarity -> fallback to FedAvg.")
            result = super().aggregate()
            self.received_params, self.received_lens, self.received_ids = [], [], []
            return result

        # -- Stage 2: bootstrap validation --> per-neighbor weight --
        weighted: List[Tuple[Dict[str, torch.Tensor], float]] = [(own_state, 1.0)]  # local model, w_i = 1
        for sid, recv_state in surviving:
            loss = self._compute_bootstrap_loss(recv_state, bootstrap_batch)
            self.loss_history.setdefault(sid, []).append(loss)
            w = self._map_loss_distance(self.own_loss_history, self.loss_history[sid])
            if w > 0.0:
                weighted.append((recv_state, w))

        total_weight = sum(w for _, w in weighted)
        if total_weight <= 0.0:
            print("[SENTINEL] All neighbor weights zeroed -> fallback to FedAvg.")
            result = super().aggregate()
            self.received_params, self.received_lens, self.received_ids = [], [], []
            return result

        # -- Stage 3: layer normalization + weighted aggregation --
        averaged: Dict[str, torch.Tensor] = {}
        for key in own_state.keys():
            acc = torch.zeros_like(own_state[key], dtype=torch.float32)
            for state, w in weighted:
                normalized = self._normalize_model(own_state, state) if state is not own_state else state
                acc += w * normalized[key].float()
            averaged[key] = (acc / total_weight).to(own_state[key].dtype)

        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # Reset only the per-round buffers -- loss history above is untouched.
        self.received_params, self.received_lens, self.received_ids = [], [], []

        print(f"[SENTINEL] Aggregated {len(weighted) - 1}/{num_updates} neighbors "
              f"(+ own model) at round {current_round}, own bootstrap loss={own_loss:.4f}")

        return {k: v.cpu().clone() for k, v in averaged.items()}