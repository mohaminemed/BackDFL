# dfl_dual_server.py
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List, Tuple, Any

from ..fl.baseserver import FedAvgAggregator

# -------------------------
# Utility helpers
# -------------------------

def _flatten_state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    keys = sorted(state.keys())
    parts = [state[k].detach().cpu().flatten() for k in keys]
    return torch.cat(parts, dim=0)


def _flatten_to_numpy(state: Dict[str, torch.Tensor]) -> np.ndarray:
    return _flatten_state_dict_to_vector(state).numpy()


def _ed(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    a = _flatten_state_dict_to_vector(state_a)
    b = _flatten_state_dict_to_vector(state_b)
    return float(torch.linalg.norm(a - b).cpu())


def _cosine_similarity(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    a = _flatten_state_dict_to_vector(state_a)
    b = _flatten_state_dict_to_vector(state_b)
    an = float(torch.linalg.norm(a).cpu())
    bn = float(torch.linalg.norm(b).cpu())
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float((a @ b).cpu() / (an * bn))


# 1D Wasserstein (approx via CDF difference)
def _wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    if u.size == 0 or v.size == 0:
        return float(np.inf)
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)
    pts = np.unique(np.concatenate([u_sorted, v_sorted]))
    # CDF values
    iu = np.searchsorted(u_sorted, pts, side='right') / float(u_sorted.size)
    iv = np.searchsorted(v_sorted, pts, side='right') / float(v_sorted.size)
    deltas = np.diff(np.concatenate([pts, pts[-1:] + 0.0]))
    vals = np.abs(iu - iv) * deltas
    return float(np.sum(vals))


def compute_wasserstein_between_dummy_datasets(dset_a: Dict[int, np.ndarray], dset_b: Dict[int, np.ndarray]) -> float:
    total = 0.0
    labels = set(list(dset_a.keys()) + list(dset_b.keys()))
    for lab in labels:
        arr_a = dset_a.get(lab, None)
        arr_b = dset_b.get(lab, None)
        if arr_a is None and arr_b is None:
            continue
        if arr_a is None:
            arr_a = np.zeros_like(arr_b)
        if arr_b is None:
            arr_b = np.zeros_like(arr_a)
        arr_a = np.asarray(arr_a)
        arr_b = np.asarray(arr_b)
        if arr_a.ndim == 1:
            arr_a = arr_a[:, None]
        if arr_b.ndim == 1:
            arr_b = arr_b[:, None]
        n_features = max(arr_a.shape[1], arr_b.shape[1])
        if arr_a.shape[1] < n_features:
            arr_a = np.pad(arr_a, ((0, 0), (0, n_features - arr_a.shape[1])))
        if arr_b.shape[1] < n_features:
            arr_b = np.pad(arr_b, ((0, 0), (0, n_features - arr_b.shape[1])))
        for f in range(n_features):
            total += _wasserstein_1d(arr_a[:, f], arr_b[:, f])
    return float(total)


# -------------------------
# Model inversion for dummy dataset generation
# (privacy-aware: optimize features only; labels fixed/sampled)
# -------------------------
def model_inversion_generate_dummy_dataset(
    model: nn.Module,
    target_eq_grad: Dict[str, torch.Tensor],
    input_shape: Tuple[int, ...],
    num_samples_per_class: int = 5,
    classes: Optional[List[int]] = None,
    steps: int = 100,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
    s_t: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Gradient-matching model inversion (DLG-style): optimizes synthetic inputs `x`
    so that the gradient they induce on `model`'s parameters matches
    `target_eq_grad` (optionally scaled by `s_t`).

    Returns dict[label] -> np.array (n_samples x flattened_feature_dim)

    Notes on correctness:
      - `model`'s parameters are replaced with fresh leaf `nn.Parameter`s so that
        `torch.autograd.grad` can be called against them cleanly.
      - The per-step gradient w.r.t. model parameters is computed with
        `create_graph=True`. This is required so that the matching loss
        (built from those gradients) is itself differentiable w.r.t. `x` --
        without it, `obj.backward()` has no path back to `x` and the
        synthetic samples never actually update (a "double backprop" bug).
    """
    device = device or torch.device('cpu')

    # Copy model & move to device.
    model = copy.deepcopy(model).to(device)
    model.eval()

    # Ensure parameters are fresh leaf nn.Parameter objects (so autograd.grad
    # can be taken w.r.t. them directly, and so we don't mutate the caller's model).
    for name, p in model.named_parameters():
        new_p = nn.Parameter(p.detach().clone().to(device), requires_grad=True)
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], new_p)

    params = list(model.parameters())

    # Flatten target gradient vector (already scaled by caller if desired).
    tgt_vec = _flatten_state_dict_to_vector(target_eq_grad).to(device)

    feature_dim = int(np.prod(input_shape))
    classes = classes if classes is not None else list(range(10))
    outputs: Dict[int, List[np.ndarray]] = {lab: [] for lab in classes}

    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    # Inversion loop: optimize inputs x only.
    for lab in classes:
        for _ in range(num_samples_per_class):
            x = torch.randn((1, feature_dim), device=device, requires_grad=True) * 0.1
            opt = optim.Adam([x], lr=lr)
            for _step in range(steps):
                opt.zero_grad()
                inp = x.view((1,) + input_shape) if len(input_shape) > 1 else x
                logits = model(inp)
                label = torch.tensor([lab], device=device, dtype=torch.long)
                loss = loss_fn(logits, label)

                # Compute grads w.r.t. model parameters. create_graph=True keeps
                # this computation connected to `x` so that `obj.backward()`
                # below actually flows gradients into `x`.
                grads = torch.autograd.grad(loss, params, create_graph=True)
                grad_vec = torch.cat([g.flatten() for g in grads], dim=0)

                # Matching objective against the (possibly s_t-scaled) target gradient.
                diff = grad_vec - s_t * tgt_vec
                obj = torch.norm(diff) ** 2
                obj.backward()
                opt.step()

            outputs[lab].append(x.detach().cpu().numpy().reshape(-1))

    final = {lab: np.stack(arrs, axis=0) for lab, arrs in outputs.items()}
    return final


# -------------------------
# DFLDualServer
# -------------------------
class DFLDualServer(FedAvgAggregator):
    """
    Implementation of DFL-Dual as a subclass of FedAvgAggregator.

    Expected self.received_params: list of either:
      - plain state_dict (torch tensors), OR
      - dict with keys: 'model' (state_dict), optional 'E','B','local_size','meta'...

    self.received_lens is used (sample counts).

    Configuration (config dict):
      - dfldual_alpha (float) default 1.0
      - dfldual_C1 (float) default 1e6
      - dfldual_C2 (float) default 10.0
      - dummy_samples_per_class (int) default 5 (fast_mode) / 10 (else)
      - inversion_steps (int) default 100 (fast_mode) / 200 (else)
      - inversion_lr (float) default 0.05 (fast_mode) / 0.1 (else)
      - inversion_s (float) default 1.0
      - use_inversion (bool) default False
      - fast_mode (bool) default False -> reduces inversion cost defaults when True
      - classes (list) default 0..9
      - input_shape (tuple) default (1,28,28)
      - local_epochs (int) default 5   -- used when client's E not provided
      - batch_size (int) default 32    -- used when client's B not provided
      - weighted_aggregation (bool) default False -- weight by sample count
        instead of a plain arithmetic mean over selected clients
    """

    def __init__(self, model: nn.Module, testloader: Optional[Any] = None,
                 device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)
        cfg = config or {}

        # DFL-Dual hyperparams
        self.alpha = float(cfg.get('dfldual_alpha', 1.0))
        self.C1 = float(cfg.get('dfldual_C1', 1e6))
        self.C2 = float(cfg.get('dfldual_C2', 10.0))

        # inversion defaults (fast-mode tuned)
        self.fast_mode = bool(cfg.get('fast_mode', False))
        if self.fast_mode:
            self.dummy_samples_per_class = int(cfg.get('dummy_samples_per_class', 5))
            self.inversion_steps = int(cfg.get('inversion_steps', 100))
            self.inversion_lr = float(cfg.get('inversion_lr', 0.05))
        else:
            self.dummy_samples_per_class = int(cfg.get('dummy_samples_per_class', 10))
            self.inversion_steps = int(cfg.get('inversion_steps', 200))
            self.inversion_lr = float(cfg.get('inversion_lr', 0.1))

        self.inversion_s = float(cfg.get('inversion_s', 1.0))
        self.use_inversion = bool(cfg.get('use_inversion', False))
        self.classes = cfg.get('classes', list(range(10)))
        self.input_shape = tuple(cfg.get('input_shape', (1, 28, 28)))
        self.prev_global_state: Optional[Dict[str, torch.Tensor]] = None

        # fallback local epoch/batch if clients don't send them
        self.default_local_epochs = int(cfg.get('local_epochs', 5))
        self.default_batch_size = int(cfg.get('batch_size', 32))

        # aggregation strategy: plain mean (paper) vs sample-count weighted mean
        self.weighted_aggregation = bool(cfg.get('weighted_aggregation', False))

        print(f"Initialized DFLDualServer(alpha={self.alpha}, C1={self.C1}, "
              f"use_inversion={self.use_inversion}, fast_mode={self.fast_mode})")

    # helper to extract model state dict from received entry
    def _entry_model_state(self, entry: Any) -> Dict[str, torch.Tensor]:
        if isinstance(entry, dict) and 'model' in entry:
            return entry['model']
        elif isinstance(entry, dict) and len(entry) > 0 and all(isinstance(v, torch.Tensor) for v in entry.values()):
            # Possibly a raw state dict disguised as a dict wrapper
            return entry
        else:
            return entry

    def _compute_equivalent_gradient(self, entry: Any, round_idx: int) -> Dict[str, torch.Tensor]:
        """
        Compute equivalent gradient:
          g_i = (theta^{t+1/2}_i - theta^{t}_i) / (E * (|D_i| / B))
        If entry provides E/B/local_size use them, otherwise fall back to config defaults.
        prev_global_state (theta^t) is taken from server-stored copy self.prev_global_state,
        unless the entry itself carries a 'prev_model'.
        Returns a state-dict-shaped dict on CPU, scaled by s_t = inversion_s ** round_idx.
        """
        model_state = self._entry_model_state(entry)

        if isinstance(entry, dict) and entry.get('prev_model') is not None:
            prev_state = entry.get('prev_model')
        elif self.prev_global_state is not None:
            prev_state = self.prev_global_state
        else:
            prev_state = {k: torch.zeros_like(v) for k, v in model_state.items()}

        is_meta_dict = isinstance(entry, dict) and 'model' in entry
        E = float(entry.get('E', self.default_local_epochs)) if is_meta_dict else self.default_local_epochs
        B = float(entry.get('B', self.default_batch_size)) if is_meta_dict else self.default_batch_size
        local_size = float(entry.get('local_size', 1.0)) if is_meta_dict else 1.0

        denom = E * (local_size / max(1.0, B))
        if denom == 0.0:
            denom = 1.0

        s_t = (self.inversion_s ** round_idx) if self.inversion_s != 1.0 else 1.0

        eq = {}
        for k in model_state.keys():
            a = model_state[k].detach().cpu()
            b = prev_state.get(k, torch.zeros_like(a)).detach().cpu() if isinstance(prev_state, dict) else torch.zeros_like(a)
            eq[k] = ((a - b) / denom) * s_t
        return eq

    def _compute_dual_distance_matrix(self, models: List[Dict[str, torch.Tensor]],
                                       dummy_datasets: List[Optional[Dict[int, np.ndarray]]]) -> np.ndarray:
        n = len(models)
        D = np.zeros((n, n), dtype=float)
        flattened = [_flatten_to_numpy(m) for m in models]
        for i in range(n):
            for j in range(n):
                if i == j:
                    D[i, j] = 0.0
                    continue
                ed_ij = float(np.linalg.norm(flattened[i] - flattened[j]))
                wd_ij = 0.0
                if dummy_datasets and dummy_datasets[i] is not None and dummy_datasets[j] is not None:
                    wd_ij = compute_wasserstein_between_dummy_datasets(dummy_datasets[i], dummy_datasets[j])
                D[i, j] = min(wd_ij + self.alpha * ed_ij, self.C1)
        return D

    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        """
        Main aggregation call. This method:
         - stores prev_global_state (theta^t) before any changes
         - optionally generates dummy datasets per received model via inversion
         - computes D_ij dual distances, bootstrap two-stage selection, and aggregates selected models
        """
        # snapshot current global model as theta^t before aggregation
        global_state = self.get_params()
        self.prev_global_state = {k: v.detach().cpu().clone() for k, v in global_state.items()}

        num_updates = len(self.received_params)
        if num_updates == 0:
            print("DFLDualServer.aggregate(): warning - no updates")
            return global_state

        # extract model state dicts list in same order as received
        models: List[Dict[str, torch.Tensor]] = [self._entry_model_state(e) for e in self.received_params]

        # prepare dummy datasets (inversion) if requested
        dummy_datasets: List[Optional[Dict[int, np.ndarray]]] = [None] * len(models)
        if self.use_inversion:
            for idx, entry in enumerate(self.received_params):
                try:
                    # eq_grad is already scaled by s_t = inversion_s ** current_round,
                    # so we pass s_t=1.0 into the inversion call below to avoid
                    # applying that scaling a second time.
                    eq_grad = self._compute_equivalent_gradient(entry, current_round)
                    try:
                        model_copy = copy.deepcopy(self.model)
                        model_copy.load_state_dict(self._entry_model_state(entry))
                    except Exception as exc:
                        print(f"[DFL-Dual] Warning: cannot load entry model into server architecture "
                              f"for idx={idx}. Skipping inversion. Exc: {exc}")
                        dummy_datasets[idx] = None
                        continue
                    dd = model_inversion_generate_dummy_dataset(
                        model=model_copy,
                        target_eq_grad=eq_grad,
                        input_shape=self.input_shape,
                        num_samples_per_class=self.dummy_samples_per_class,
                        classes=self.classes,
                        steps=self.inversion_steps,
                        lr=self.inversion_lr,
                        device=self.device or torch.device('cpu'),
                        s_t=1.0,
                    )
                    dummy_datasets[idx] = dd
                except Exception as exc:
                    print(f"[DFL-Dual] Inversion failed for idx={idx}: {exc}")
                    dummy_datasets[idx] = None
        else:
            # skip inversion; WD will be 0
            dummy_datasets = [None] * len(models)

        # compute dual distance matrix D_ij
        D = self._compute_dual_distance_matrix(models, dummy_datasets)

        # compute cosine similarities w.r.t server global model
        cosines = [_cosine_similarity(global_state, m) for m in models]

        # Stage 1: for each j, cluster other clients (k != j) into two groups by
        # median D_kj and pick the more-trustworthy group M*_j
        n = len(models)
        selected_masks = np.zeros((n, n), dtype=bool)  # selected_masks[j,k] = True if k in M*_j
        for j in range(n):
            idxs = [k for k in range(n) if k != j]
            if len(idxs) == 0:
                selected_masks[j, j] = True
                continue
            dist_to_j = np.array([D[k, j] for k in idxs])
            med = float(np.median(dist_to_j))
            M1 = [idxs[i] for i, d in enumerate(dist_to_j) if d <= med]
            M2 = [idxs[i] for i, d in enumerate(dist_to_j) if d > med]

            def trust_of(group):
                if len(group) == 0:
                    return -1.0
                return float(np.mean([cosines[k] for k in group]))

            t1 = trust_of(M1)
            t2 = trust_of(M2)
            Mstar = M1 if (t1 > t2) else M2
            for k in Mstar:
                selected_masks[j, k] = True

        # q_j = sum_{k in M*_j} D_kj
        q_vals = np.zeros(n, dtype=float)
        for j in range(n):
            group_k = np.where(selected_masks[j])[0].tolist()
            if len(group_k) == 0:
                q_vals[j] = 1e-6
            else:
                q_vals[j] = float(np.sum(D[group_k, j]))

        q_i_est = float(np.median(q_vals)) + 1e-8
        r_vals = np.minimum(q_vals / q_i_est, self.C2)

        # Stage 2: cluster r_j into two groups by median and pick the group with higher trust
        med_r = float(np.median(r_vals))
        G1 = [j for j in range(n) if r_vals[j] <= med_r]
        G2 = [j for j in range(n) if r_vals[j] > med_r]

        def trust_of_group(group):
            if len(group) == 0:
                return -1.0
            return float(np.mean([cosines[k] for k in group]))

        tG1 = trust_of_group(G1)
        tG2 = trust_of_group(G2)
        selected_final = G1 if (tG1 > tG2) else G2

        # fallback: if empty, pick top half by cosine similarity
        if len(selected_final) == 0:
            order = np.argsort([-c for c in cosines])  # descending
            kpick = max(1, int(math.ceil(n / 2)))
            selected_final = order[:kpick].tolist()

        # Compose selected states and use received_lens for possible weighting
        selected_states = [models[i] for i in selected_final]
        selected_lens = [self.received_lens[i] if i < len(self.received_lens) else 1 for i in selected_final]

        if len(selected_states) == 0:
            print("[DFL-Dual] No selected states, returning global_state unchanged")
            self.received_params = []
            self.received_lens = []
            return global_state

        # Aggregate: arithmetic mean (paper) OR sample-count-weighted mean
        averaged: Dict[str, torch.Tensor] = {}
        first = selected_states[0]
        if self.weighted_aggregation:
            total = float(sum(selected_lens)) if sum(selected_lens) > 0 else float(len(selected_lens))
            for k in first.keys():
                acc = torch.zeros_like(first[k], dtype=torch.float32)
                for st, ln in zip(selected_states, selected_lens):
                    acc += st[k].detach().cpu() * (float(ln) / total)
                averaged[k] = acc.to(first[k].dtype)
        else:
            for k in first.keys():
                stacked = torch.stack([st[k].detach().cpu() for st in selected_states], dim=0)
                averaged[k] = torch.mean(stacked, dim=0).to(first[k].dtype)

        # Update server model and clear buffers
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_params = []
        self.received_lens = []

        print(f"[DFL-Dual] Aggregated {len(selected_final)}/{num_updates} clients at round {current_round}")
        return {k: v.cpu().clone() for k, v in averaged.items()}