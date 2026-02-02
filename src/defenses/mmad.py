import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple

from ..fl.baseserver import FedAvgAggregator


class MultiMetricsServer(FedAvgAggregator):
    """
    Multi-metrics adaptive defense (Huang et al., ICCV 2023) implementation.

    Key steps implemented (faithful to the paper):
      1. Compute per-client feature vector x = (L1, L2, Cosine) between client model
         and reference (server global) model.
      2. For each client i compute x'_i = (sum_j |x_i - x_j|) coordinate-wise.
      3. Form matrix X = [x'_1; x'_2; ...; x'_K] and compute covariance Sigma.
      4. Whiten: delta_i = sqrt(x'_i^T Sigma^{-1} x'_i).
      5. Rank clients by delta (low delta = more "benign"). Keep the lowest p fraction
         (config param `mmad_p`) and aggregate those with FedAvg weighting.

    Config keys (via `config`):
      - mmad_p (float): fraction p of clients to keep (default 0.5)
      - mmad_whiten (bool): whether to apply whitening (default True)
      - mmad_min_clients (int): minimum K to run whitening (default 4)
      - mmad_eps (float): regularization added to covariance diagonal for stability (default 1e-6)
      - mmad_metrics (list): which metrics to compute ['manhattan','euclidean','cosine']
      

    Provides two helpers for decentralized usage:
      - features_from_states(states, reference) -> Tensor(K, M)
      - accepts(received_state, neighborhood_states, reference, config)
        This lets a client reuse the same logic with its local neighborhood (note: the
        statistical quality depends on neighborhood size; paper assumes K>3).
    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        self.p = float(self.config.get('mmad_p', 0.5))
        self.whiten = bool(self.config.get('mmad_whiten', True))
        self.min_clients = int(self.config.get('mmad_min_clients', 4))
        self.eps = float(self.config.get('mmad_eps', 1e-6))
        self.metrics = list(self.config.get('mmad_metrics', ['manhattan', 'euclidean', 'cosine']))

        print(f"Initialized MultiMetricsServer(p={self.p}, whiten={self.whiten}, metrics={self.metrics})")

    # -------------------- utilities --------------------
    @staticmethod
    def _flatten_state(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [state[k].detach().cpu().flatten() for k in state]
        return torch.cat(parts, dim=0)

    def _compute_metrics(self, client_vecs: List[torch.Tensor], ref_vec: torch.Tensor) -> torch.Tensor:
        """Compute per-client feature values (K x M) where M = number of metrics."""
        K = len(client_vecs)
        feats: List[List[float]] = []
        try :
          for v in client_vecs:
            vals = []
            if 'manhattan' in self.metrics:
                vals.append(float(torch.linalg.norm(v - ref_vec, ord=1).item()))
            if 'euclidean' in self.metrics:
                vals.append(float(torch.linalg.norm(v - ref_vec, ord=2).item()))
            if 'cosine' in self.metrics:
                denom = (torch.linalg.norm(v) * torch.linalg.norm(ref_vec))
                if denom == 0:
                    vals.append(0.0)
                else:
                    vals.append(float(torch.dot(v, ref_vec).item() / denom.item()))
            feats.append(vals)
        except Exception as e:
                print(f'Compute metrics failed with {e}; falling back to L2 on xprime')    
        return torch.tensor(feats, dtype=torch.float64)  # shape (K, M)

    @staticmethod
    def _compute_xprime_matrix(X: torch.Tensor) -> torch.Tensor:
        # X: (K, M) features; x'_i = sum_{j != i} |x_i - x_j|
        K = X.shape[0]
        # broadcast subtract
        diffs = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))  # (K, K, M)
        # zero diagonal
        diffs[range(K), range(K), :] = 0.0
        xprime = diffs.sum(dim=1)  # (K, M)
        return xprime

    @staticmethod
    def _cov_inv(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # X: (K, M) where K samples, M features. compute covariance (M x M)
        K = X.shape[0]
        if K <= 1:
            raise ValueError('Need at least 2 samples to compute covariance')
        mu = torch.mean(X, dim=0, keepdim=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / float(max(1, K - 1))  # (M, M)
        # regularize
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype)
        # invert (use pinv for stability)
        try:
            inv = torch.linalg.inv(cov)
        except RuntimeError:
            inv = torch.linalg.pinv(cov)
        return inv

    # -------------------- decentralized helper --------------------
    @staticmethod
    def features_from_states(states: Sequence[Dict[str, torch.Tensor]], reference: Dict[str, torch.Tensor], metrics: Sequence[str] = ('manhattan','euclidean','cosine')) -> torch.Tensor:
        ref_vec = torch.cat([reference[k].detach().cpu().flatten() for k in reference])
        client_vecs = [torch.cat([s[k].detach().cpu().flatten() for k in s]) for s in states]
        # compute the same features as server method (use double precision for stability)
        feats = []
        for v in client_vecs:
            vals = []
            if 'manhattan' in metrics:
                vals.append(float(torch.linalg.norm(v - ref_vec, ord=1).item()))
            if 'euclidean' in metrics:
                vals.append(float(torch.linalg.norm(v - ref_vec, ord=2).item()))
            if 'cosine' in metrics:
                denom = (torch.linalg.norm(v) * torch.linalg.norm(ref_vec))
                vals.append(0.0 if denom == 0 else float(torch.dot(v, ref_vec).item() / denom.item()))
            feats.append(vals)
        return torch.tensor(feats, dtype=torch.float64)

    @staticmethod
    def accepts(received_state: Dict[str, torch.Tensor], neighborhood_states: Sequence[Dict[str, torch.Tensor]], reference: Dict[str, torch.Tensor], config: Optional[Dict] = None) -> bool:
        """
        Helper for DFL flows: given a neighborhood (including the received state and
        other neighbor states), compute the Multi-metrics scoring locally and return
        whether `received_state` is among the kept p-fraction (i.e., considered benign).

        Note: statistical quality depends on neighborhood size; paper presumes K>3.
        """
        cfg = config if config is not None else {}
        p = float(cfg.get('mmad_p', 0.5))
        whiten = bool(cfg.get('mmad_whiten', True))
        min_clients = int(cfg.get('mmad_min_clients', 4))
        eps = float(cfg.get('mmad_eps', 1e-6))
        metrics = list(cfg.get('mmad_metrics', ['manhattan','euclidean','cosine']))

        # build states list with received first
        states = list(neighborhood_states)
        # ensure received_state is included (if not present)
        if not any(all(torch.all(received_state[k] == s[k]) for k in received_state) for s in states):
            states = [received_state] + states
        K = len(states)
        if K < 2 or K <= min_clients - 1:
            # fallback: accept by default when not enough local samples
            return True

        feats = MultiMetricsServer.features_from_states(states, reference, metrics=metrics)
        xprime = MultiMetricsServer._compute_xprime_matrix(feats)

        if whiten:
            try:
                inv = MultiMetricsServer._cov_inv(xprime, eps=eps)
                deltas = torch.sqrt(torch.sum((xprime @ inv) * xprime, dim=1))
            except Exception:
                deltas = torch.norm(xprime, dim=1)
        else:
            deltas = torch.norm(xprime, dim=1)

        # index of received state (we ensured it's first if not originally present)
        recv_idx = 0
        K_keep = max(1, int(math.floor(p * K)))
        sorted_indices = torch.argsort(deltas)
        kept = sorted_indices[:K_keep].tolist()
        return recv_idx in kept

    # -------------------- server aggregation --------------------
    def aggregate(self) -> Dict[str, torch.Tensor]:
        K = len(self.received_params)
        if K == 0:
            print('MultiMetricsServer.aggregate(): no updates')
            return self.get_params()

        if K < 2:
            print('Too few clients: falling back to FedAvg')
            return super().aggregate()

        # flatten client vectors
        client_vecs = [self._flatten_state(p) for p in self.received_params]
        ref_vec = self._flatten_state(self.get_params())

        # compute metrics matrix (K x M)
        X = self._compute_metrics(client_vecs, ref_vec)  # double precision

        # compute x' matrix
        xprime = self._compute_xprime_matrix(X)  # (K, M)

        # apply whitening if enabled and enough clients
        if self.whiten and K >= self.min_clients:
            try:
                inv = self._cov_inv(xprime, eps=self.eps)
                deltas = torch.sqrt(torch.sum((xprime @ inv) * xprime, dim=1))
            except Exception as e:
                print(f'MultiMetricsServer: whitening failed with {e}; falling back to L2 on xprime')
                deltas = torch.norm(xprime, dim=1)
        else:
            deltas = torch.norm(xprime, dim=1)

        # lower delta => more benign. keep lowest p fraction
        K_keep = max(1, int(math.floor(self.p * K)))
        sorted_idx = torch.argsort(deltas)
        kept_idx = sorted_idx[:K_keep].tolist()

        if len(kept_idx) == 0:
            print('MultiMetricsServer: no clients kept, falling back to FedAvg')
            return super().aggregate()

        print(f'MultiMetricsServer kept indices: {kept_idx} (out of {K})')

        # aggregate selected clients with FedAvg weighting
        sel_states = [self.received_params[i] for i in kept_idx]
        sel_lens = [self.received_lens[i] for i in kept_idx]
        total = float(sum(sel_lens)) if sum(sel_lens) > 0 else float(len(sel_lens))

        first = sel_states[0]
        averaged: Dict[str, torch.Tensor] = {}
        for k in first.keys():
            acc = torch.zeros_like(first[k], dtype=torch.float32)
            for i, st in enumerate(sel_states):
                weight = float(sel_lens[i]) / total
                acc += st[k].detach().cpu() * weight
            averaged[k] = acc.type(first[k].dtype)

        # write back and clear
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_params = []
        self.received_lens = []

        return {k: v.cpu().clone() for k, v in averaged.items()}
