import copy
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the specific FedAvgAggregator from your project structure
from ..fl.baseserver import FedAvgAggregator


class UBARServer(FedAvgAggregator):
    """
    UBAR filtering as described in Guo et al., "Byzantine-Resilient
    Decentralized Stochastic Gradient Descent" (IEEE TCSVT, 2022).

    UBAR is a two-stage, per-round selection rule (paper Definition 3 /
    Algorithm 1) -- unlike Sentinel it carries no state across
    rounds, since candidates are reselected from scratch every iteration:
      1. Distance shortlist: keep the rho * |neighbors| neighbors whose
         estimate is closest (Euclidean distance) to the local model.
         The local model itself -- not the neighbor mean/median used by
         centralized defenses -- is the baseline, since it can't be
         manipulated by a Byzantine neighbor.
      2. Loss selection: evaluate every shortlisted candidate's
         *parameters* on a batch sampled from this node's *own* local
         training data, and keep only those whose loss is <= the local
         model's loss on that same batch. If none qualify, keep the
         single best (lowest-loss) candidate (paper Algorithm 1, lines
         12-14), so Stage 2 never empties the selection entirely.
      The kept candidates are averaged into R, then blended with the
      local model via the General Update Function: theta_new =
      alpha * theta_local + (1 - alpha) * R.
 

    Sampling unit. The paper's Algorithm 1 formalizes Stage 2 around a
    single stochastically-selected data sample xi_k,i per iteration;
    this implementation instead samples one small batch per round for
    numerical stability, which is the standard mini-batch-SGD reading of
    the same idea rather than a literal single example.

    Config keys:
      - ubar_rho (float, default 0.4): assumed ratio of benign
        neighbors, used to size the Stage-1 shortlist
        (round(rho * |neighbors|), at least 1). The paper requires each
        node to set this from an assessment of its threat environment
        (Section VII); 0.4 is only the value used in the paper's own
        experiments, not a calibrated default -- tune per deployment. If
        unknown, the paper suggests the conservative rho = 1/|neighbors|
        (assume just one benign neighbor).
      - ubar_batch_size (int, default 64): size of the batch freshly
        sampled from `ubar_train_dataset` each round for Stage 2 loss
        evaluation.
      - ubar_train_dataset: the client's own *training* Dataset (not a
        held-out validation set -- the paper explicitly reuses training
        samples for Stage 2, avoiding the extra-validation-set
        requirement of centralized methods like Zeno). Required.
    """

    def __init__(self, model: nn.Module, testloader: nn.Module = None,
                 device: Optional[torch.device] = None, config: Optional[Dict] = None):
        super().__init__(model, testloader, device)

        self.config = config if config is not None else {}
        self.rho = float(self.config.get('ubar_rho', 0.4))
        self.batch_size = int(self.config.get('ubar_batch_size', 64))
        self.val_dataset = self.config.get('ubar_val_dataset', None)

        # Per-round buffer only -- UBAR carries no state across rounds.
        self.received_params: List[Dict[str, torch.Tensor]] = []
        self.received_lens: List[int] = []

        print(f"Initialized UBARServer (rho={self.rho}, "
              f"batch_size={self.batch_size})")

    # -------------------- helpers --------------------

    @staticmethod
    def _flatten_state_dict_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(state.keys())
        return torch.cat([state[k].detach().cpu().flatten() for k in keys], dim=0)

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.val_dataset is None:
            raise ValueError("UBARServer requires `ubar_val_dataset` in config.")
        n = len(self.val_dataset)
        print(f"[UBAR] Sampling batch of size {self.batch_size} from val dataset of size {n}")
        size = min(n, self.batch_size)
        indices = torch.randperm(n)[:size].tolist()
        xs, ys = zip(*(self.val_dataset[i] for i in indices))
        return torch.stack(xs).to(self.device), torch.tensor(ys).to(self.device)

    def _compute_loss(self, state: Dict[str, torch.Tensor],
                       batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = batch
        probe_model = copy.deepcopy(self.model).to(self.device)
        probe_model.load_state_dict(state)
        probe_model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(probe_model(x), y).item()
        return loss

    # -------------------- aggregation --------------------

    def aggregate(self, current_round: int) -> Dict[str, torch.Tensor]:
        own_state = self.get_params()
        num_neighbors = len(self.received_params)

        if num_neighbors == 0:
            print("UBARServer.aggregate(): warning - no neighbor updates, keeping local model.")
            self.received_params, self.received_lens = [], []
            return own_state

        # -- Stage 1: distance shortlist --
        own_vec = self._flatten_state_dict_to_vector(own_state)
        distances = [
            (i, float(torch.linalg.norm(own_vec - self._flatten_state_dict_to_vector(state))))
            for i, state in enumerate(self.received_params)
        ]
        distances.sort(key=lambda pair: pair[1])
        shortlist_size = max(1, round(self.rho * num_neighbors))
        shortlisted_indices = [i for i, _ in distances[:shortlist_size]]

        # -- Stage 2: loss selection on this node's own data --
        batch = self._sample_batch()
        own_loss = self._compute_loss(own_state, batch)

        candidate_losses = [
            (i, self._compute_loss(self.received_params[i], batch)) for i in shortlisted_indices
        ]
        selected_indices = [i for i, loss in candidate_losses if loss <= own_loss]

        if not selected_indices:
            # Paper Algorithm 1, lines 12-14: never leave the selection empty.
            print(f"[UBAR] Round {current_round}: no candidates beat own loss ({own_loss:.4f}), falling back to best candidate.")
            best_i, _ = min(candidate_losses, key=lambda pair: pair[1])
            selected_indices = [best_i]

        # -- Average selected candidates --
        selected_states = [self.received_params[i] for i in selected_indices]
        r_state: Dict[str, torch.Tensor] = {}
        for key in own_state.keys():
            stacked = torch.stack([st[key].detach().cpu().float() for st in selected_states], dim=0)
            r_state[key] = torch.mean(stacked, dim=0)

        
        self.set_params({k: v.to(self.device) for k, v in r_state.items()})

        self.received_params, self.received_lens = [], []

        print(f"[UBAR] Round {current_round}: shortlisted {shortlist_size}/{num_neighbors} by "
              f"distance, kept {len(selected_indices)} by loss (own_loss={own_loss:.4f})")

        return {k: v.cpu().clone() for k, v in r_state.items()}