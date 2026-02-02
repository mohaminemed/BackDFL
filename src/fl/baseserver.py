from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import copy as cp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.base import BaseTrigger

class BaseServer(ABC):
    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def aggregate(self) -> Dict[str, torch.Tensor]:
        pass


class FedAvgAggregator(BaseServer):
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None):
        self.device = device if device is not None else torch.device("cpu")
        self.model = cp.deepcopy(model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.testloader = testloader  # keep reference (no deepcopy)
        self.received_params: List[Dict[str, torch.Tensor]] = []
        self.received_lens: List[int] = []

    def load_testdata(self, testloader):
        self.testloader = testloader

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        # load params onto server model (move to device)
        self.model.load_state_dict({k: v.to(self.device) for k, v in params.items()})

    def receive_update(self, params: Dict[str, torch.Tensor], length: int) -> None:
        # store params as CPU tensors to make aggregation stable
        params_cpu = {k: v.cpu().clone().float() for k, v in params.items()}
        self.received_params.append(params_cpu)
        self.received_lens.append(int(length))

    def evaluate(self, valloader=None) -> Dict[str, object]:
        valloader = valloader or self.testloader
        self.model.eval()
        if valloader is None:
            return {'num_samples': 0, 'metrics': {'loss': float('nan'), 'main_accuracy': float('nan')}}
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1
        return {'num_samples': total, 'metrics': {'loss': (loss_sum / iters) if iters else float('nan'), 'main_accuracy': (correct / total) if total else float('nan')}}

    def aggregate(self) -> Dict[str, torch.Tensor]:
        assert len(self.received_params) == len(self.received_lens) and len(self.received_params) > 0
        total_samples = sum(self.received_lens)

        averaged = {}
        # initialize zeros on cpu and accumulate weighted sums
        first = self.received_params[0]
        for k in first.keys():
            acc = torch.zeros_like(first[k], dtype=torch.float32)
            for i, client_params in enumerate(self.received_params):
                weight = float(self.received_lens[i]) / float(total_samples)
                acc += client_params[k] * weight
            averaged[k] = acc  # cpu tensors

        # load averaged params to server model
        self.set_params({k: v.to(self.device) for k, v in averaged.items()})

        # reset buffers
        self.received_params = []
        self.received_lens = []
        return {k: v.cpu().clone() for k, v in averaged.items()}

    def save_model(self, path: str) -> None:
        # save state_dict for portability
        torch.save(self.model.state_dict(), path)
