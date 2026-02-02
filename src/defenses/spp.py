import math
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Sequence

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator


class SPPServer(FedAvgAggregator):
    """
    Similarity of Partial Parameters (SPP) defense.

    Behavior / design notes :
      - After collecting all submissions, select J' parameter indices at random
        (without replacement) and compute similarity metrics on that subset.
      - Selection is performed AFTER collecting submissions so attackers
        cannot know which coordinates will be checked. The paper uses
        approximately J/2 in experiments.
      - A generic SPP wrapper that can be configured to use L2,
        Euclidean distance, cosine similarity, or combinations.
      - A static helper `accepts()` so the same logic can be
        reused in decentralized flows (client-side): clients may choose
        their own random subset after receiving neighbors' models.

    Config keys supported (via `config` dict):
      - spp_fraction (float, 0< f <=1): fraction of parameters J' = max(1, int(f * J)).
          Default: 0.5 (paper experimental choice ~J/2).
      - spp_fixed_count (int): optional override to specify exact J' instead
          of fraction. If provided, takes precedence.
      - spp_metric (str): 'cosine' | 'l2' | 'euclidean' | 'combined'
          Default: 'cosine'
      - spp_seed (int|None): seed for reproducible random subset selection.
          If None, selection will be nondeterministic per round.
      - spp_combined_weights (dict): when spp_metric == 'combined', a mapping
          e.g. {'cosine':0.6,'l2':0.4} to combine normalized metric scores.
      - spp_threshold (float|None): if provided, accept only models whose
          similarity (or combined score) >= spp_threshold. If None, behavior
          follows the "accept-if-nonnegative" or metric-specific rules; e.g.
          for cosine require > 0, for distance require <= some value -- in
          the generic wrapper we fall back to 'relative' comparison to the
          reference (see paper examples).

    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None, device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        self.fraction = float(self.config.get('spp_fraction', 0.5))
        self.fixed_count = self.config.get('spp_fixed_count', None)
        self.metric = str(self.config.get('spp_metric', 'cosine'))
        self.seed = self.config.get('spp_seed', 1234)
        self.threshold = None #self.config.get('spp_threshold', 0.0) # if None, use relative mode
        self.rel_threshold = None  # for relative acceptance mode
        self.combined_weights = self.config.get('spp_combined_weights', None)

        # round bookkeeping for deterministic selection if desired
        self.current_round = int(self.config.get('round', 0))

        print(f"Initialized SPPServer(fraction={self.fraction}, fixed_count={self.fixed_count}, metric={self.metric}, threshold={self.threshold},  seed={self.seed})")

    # -------------------- helpers --------------------
    @staticmethod
    def _flatten_state_dict(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [state[k].detach().cpu().flatten() for k in state]
        return torch.cat(parts, dim=0)

    def _choose_indices(self, J: int) -> List[int]:
        # determine J' from fraction or fixed_count
        if self.fixed_count is not None:
            jprime = min(max(1, int(self.fixed_count)), J)
        else:
            jprime = min(max(1, int(self.fraction * J)), J)

        rng = torch.Generator()
        if self.seed is not None:
            # mix round into seed for per-round deterministic variation
            rng.manual_seed(int(self.seed) + int(self.current_round))
        else:
            rng = None

        if rng is not None:
            indices = torch.randperm(J, generator=rng)[:jprime].tolist()
        else:
            indices = torch.randperm(J)[:jprime].tolist()

        return indices

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        # expects 1-d tensors
        denom = (torch.linalg.norm(a) * torch.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(torch.dot(a, b).item() / denom.item())

    @staticmethod
    def _l2_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        # We convert L2 distance into a similarity-like score by negative distance
        return -float(torch.linalg.norm(a - b).item())

    @staticmethod
    def _euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.linalg.norm(a - b).item())

    def _similarity_score(self, client_state, global_state, indices) -> float:
      """
      Compute the raw similarity score according to the configured metric.
      This uses the same logic as `accepts`, but returns the float score.
      """
      recv_vec = self._flatten_state_dict(client_state)
      ref_vec = self._flatten_state_dict(global_state)

      sub_recv = recv_vec[indices]
      sub_ref = ref_vec[indices]

      if self.metric == 'cosine':
        return SPPServer._cosine_similarity(sub_recv, sub_ref)
      elif self.metric == 'l2':
        return SPPServer._l2_similarity(sub_recv, sub_ref)
      elif self.metric == 'euclidean':
        return -SPPServer._euclidean_distance(sub_recv, sub_ref)
      elif self.metric == 'combined':
        if self.combined_weights is None:
            raise ValueError('combined_weights must be supplied for combined metric')
        cos = SPPServer._cosine_similarity(sub_recv, sub_ref)
        l2 = SPPServer._l2_similarity(sub_recv, sub_ref)
        eucl = -SPPServer._euclidean_distance(sub_recv, sub_ref)
        weighted = 0.0
        for k, w in self.combined_weights.items():
            if k == 'cosine':
                weighted += w * cos
            elif k == 'l2':
                weighted += w * l2
            elif k == 'euclidean':
                weighted += w * eucl
        return weighted
      else:
        raise ValueError(f'Unknown metric type: {self.metric}')

    def _relative_score(self, global_state, indices) -> float:
      """
      Relative acceptance mode for SPP when no explicit threshold is provided.
      This implements the 'fallback to relative comparison to the reference' behavior
      described in the SPP paper.

      Returns:
        mean score over all clients.
      """
      print("SPPServer: no explicit threshold — applying relative comparison to reference model.")

      selected_indices = []
      flat_global = self._flatten_state_dict(global_state)

      # Precompute scores for all clients to allow mean-based rules
      all_scores = []
      for client_state in self.received_params:
        score = self._similarity_score(client_state, global_state, indices)
        all_scores.append(score)

      return sum(all_scores) / len(all_scores)
 
    def accepts(self, received_state: Dict[str, torch.Tensor],
            reference_state: Dict[str, torch.Tensor],
            indices: Sequence[int],
            metric: Optional[str] = None,
            combined_weights: Optional[Dict] = None,
            threshold: Optional[float] = None) -> bool:
        """
        Static helper to evaluate a received model over a *given* subset `indices`.
        This is intended for use in decentralized flows where each client picks
        their own random subset after collecting neighbor models.

        - `indices` MUST be a sequence of integers indexing into the flattened
          parameter vector. Both states must be flattened with the same ordering.
        - `metric` can be 'cosine','l2','euclidean','combined'. For 'combined',
          `combined_weights` must be provided.
        - `threshold` if provided is interpreted as: accept if score >= threshold.

        Returns True if accepted.
        """
        recv_vec = self._flatten_state_dict(received_state)
        ref_vec = self._flatten_state_dict(reference_state)

        # select partial parameters
        sub_recv = recv_vec[indices]
        sub_ref = ref_vec[indices]

        if metric == 'cosine':
            score = self._cosine_similarity(sub_recv, sub_ref)
            # default rule: require positive cosine similarity
            return score >= float(self.rel_threshold)
        elif metric == 'l2':
            # we treat lower distance as better: convert to negative distance so higher is better
            score = SPPServer._l2_similarity(sub_recv, sub_ref)
            # default: accept if not larger than some implicit margin (here accept always)
            # but better to require score >= -inf -> always true; leave threshold to user
            return score >= float(self.rel_threshold)
        elif metric == 'euclidean':
            dist = SPPServer._euclidean_distance(sub_recv, sub_ref)
            # by default accept if distance is not extremely large — but we don't set
            # an arbitrary cutoff to remain faithful to the paper's general idea.
            return dist <= float(self.rel_threshold)
        elif metric == 'combined':
            if combined_weights is None:
                raise ValueError('combined_weights must be supplied for combined metric')
            # compute safely normalized sub-scores
            cos = SPPServer._cosine_similarity(sub_recv, sub_ref)
            if cos is None or math.isnan(cos):
               cos = 0.0
            #else :
             #   cos = 1.0 / (1.0 + cos)   

            l2 = SPPServer._l2_similarity(sub_recv, sub_ref)
            if l2 is None or math.isnan(l2):
              l2 = 0.0
            else :
              l2 = 1.0 / (1.0 + l2)  

            eucl = -SPPServer._euclidean_distance(sub_recv, sub_ref)
            if eucl is None or math.isnan(eucl):
              eucl = 0.0
            else :
                eucl = 1.0 / (1.0 + eucl)  
            # normalize to [0,1] heuristically if needed
            # simple min-max normalizations are not available here; rely on user thresholds
            weighted = 0.0
            for k, w in combined_weights.items():
                if k == 'cosine':
                    weighted += w * cos
                elif k == 'l2':
                    weighted += w * l2
                elif k == 'euclidean':
                    weighted += w * eucl
                else:
                    raise ValueError(f'Unknown sub-metric {k}')
            
            print(f"weighted: {weighted}")
            return weighted >= float(self.rel_threshold)
        else:
            raise ValueError(f'Unknown metric {metric}')
      

    # -------------------- server-side aggregation --------------------
    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_updates = len(self.received_params)
        if num_updates == 0:
            print('SPPServer.aggregate(): no updates to aggregate')
            return self.get_params()

        global_state = self.get_params()
        flat_global = self._flatten_state_dict(global_state)
        J = flat_global.numel()

        # pick indices AFTER collecting submissions
        indices = self._choose_indices(J)

        if self.threshold is None:
                # relative acceptance mode
                self.rel_threshold = self._relative_score(global_state, indices)
                print(f"SPPServer: relative threshold computed as {self.rel_threshold:.4f}")
        else:
                self.rel_threshold = self.threshold


        selected_indices: List[int] = []
        for idx, client_state in enumerate(self.received_params):
            try:
                accepted = self.accepts(client_state, global_state, indices,
                                             metric=self.metric, combined_weights=self.combined_weights,
                                             threshold=self.rel_threshold)
            except Exception as e:
                print(f'SPPServer: acceptance check failed for client {idx}: {e}. Rejecting client.')
                accepted = False
            if accepted:
                selected_indices.append(idx)

        if len(selected_indices) == 0:
              # still no accepted clients, skip aggregation
              print("SPPServer: no clients passed SPP strict check. Skipping aggregation this round.")
              return self.get_params()

        print(f'SPPServer selected client indices: {selected_indices} (checked {len(indices)} parameters)')

        # Aggregate selected clients with weighted (FedAvg-style) average by default
        selected_states = [self.received_params[i] for i in selected_indices]
        selected_lens = [self.received_lens[i] for i in selected_indices]
        total_samples = float(sum(selected_lens)) if sum(selected_lens) > 0 else float(len(selected_lens))

        averaged: Dict[str, torch.Tensor] = {}
        first = selected_states[0]
        for k in first.keys():
            acc = torch.zeros_like(first[k], dtype=torch.float32)
            for i, st in enumerate(selected_states):
                weight = float(selected_lens[i]) / total_samples
                acc += st[k].detach().cpu() * weight
            averaged[k] = acc.type(first[k].dtype)

        # write back, clear buffers
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})
        self.received_params = []
        self.received_lens = []

        return {k: v.cpu().clone() for k, v in averaged.items()}
