from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy as cp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def local_evaluate(self) -> Dict[str, Any]:
        pass
    @abstractmethod
    def aggregate_from_neighbors(self, neighbor_updates, defense_type="none", config=None):
        pass
class BenignClient(BaseClient):
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device is not None else torch.device("cpu")
        self.epochs_default = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        self._model = model.to(self.device)
        self.dataset_len = len(trainloader.dataset) if trainloader is not None else 0
        
        # Optimizer and scheduler will now be created on-demand
        self.optimizer = None
        self.scheduler = None
        self._create_optimizer() # Initial creation
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.agg_server = None  # persistent server instance for DFL
    

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def _create_optimizer(self) -> None:
        """Recreates the optimizer, binding it to the current model parameters."""
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

    def get_id(self) -> int:
        return self.id

    def num_samples(self) -> int:
        return self.dataset_len

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Load parameters and recreate the optimizer to reset its state."""
        self.model.load_state_dict(params)
        self.model.to(self.device)
        self._create_optimizer()
        
    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """Train locally and return metrics collected during training."""
        self.model.train()
        
        train_loss, correct, total = 0.0, 0, 0
        
        for _ in range(epochs or self.epochs_default):
            if self.trainloader is None: break
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        if self.scheduler:
            self.scheduler.step()

        num_batches = len(self.trainloader) if self.trainloader else 1
        avg_loss = train_loss / (num_batches * (epochs or self.epochs_default))
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
        
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        return result

    def local_evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on the local test set (or train set if no test set)."""
        self.model.eval()
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        # Use testloader if available, otherwise fallback to trainloader
        valloader = self.testloader or self.trainloader
        if valloader is None:
            return {'metrics': {'loss': float('nan'), 'accuracy': float('nan')}}

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1
                
        loss_avg = (loss_sum / iters) if iters > 0 else float('nan')
        acc = (correct / total) if total > 0 else float('nan')
        
        return {'client_id': self.get_id(), 'num_samples': total, 'metrics': {'loss': loss_avg, 'accuracy': acc}}
    
    def _init_agg_server(self, defense_type: str, config: Optional[Dict] = None):
      """
      Initialize the aggregation server (persistent) for the client based on defense_type.
      If already initialized, does nothing.
      """
      if self.agg_server is not None:
        return  # already initialized

      config = config or {}

      if defense_type == "none":
        from src.fl.baseserver import FedAvgAggregator
        self.agg_server = FedAvgAggregator(self.model, self.testloader, self.device)
      elif defense_type == "krum":
        from src.defenses.krum import MKrumServer
        self.agg_server = MKrumServer(self.model, self.testloader, self.device, config)
      elif defense_type == "trim":
        from src.defenses.trim import TrimmedMeanServer
        self.agg_server = TrimmedMeanServer(self.model, self.testloader, self.device, config)
      elif defense_type == "clip":
        from src.defenses.clip_dp import NormClippingServer
        self.agg_server = NormClippingServer(self.model, self.testloader, self.device, config)
      elif defense_type == "weakdp":
        from src.defenses.clip_dp import WeakDPServer
        self.agg_server = WeakDPServer(self.model, self.testloader, self.device, config)
      elif defense_type == "flame":
        from src.defenses.flame import FlameServer
        self.agg_server = FlameServer(self.model, self.testloader, self.device, config)
      elif defense_type == "deepsight":
        from src.defenses.deepsight import DeepSightServer
        self.agg_server = DeepSightServer(self.model, self.testloader, self.device, config)
      elif defense_type == "balance":
        from src.defenses.balance import BalanceServer
        self.agg_server = BalanceServer(self.model, self.testloader, self.device, config)
      elif defense_type == "spp":
        from src.defenses.spp import SPPServer
        self.agg_server = SPPServer(self.model, self.testloader, self.device, config)
      elif defense_type == "mmad":
        from src.defenses.mmad import MultiMetricsServer
        self.agg_server = MultiMetricsServer(self.model, self.testloader, self.device, config)
      elif defense_type == "abalance":
        from src.defenses.abalance import AdaptiveBalanceServer
        self.agg_server = AdaptiveBalanceServer(self.model, self.testloader, self.device, config)  
      elif defense_type == "scclip":
        from src.defenses.scclip import SCCLIPServer
        self.agg_server = SCCLIPServer(self.model, self.testloader, self.device, config) 
      elif defense_type == "dfldual" :
         from src.defenses.dfldual import DFLDualServer
         self.agg_server = DFLDualServer(self.model, self.testloader, self.device, config)     
      elif defense_type == "ubar":
        from src.defenses.ubar import UBARServer
        self.agg_server = UBARServer(self.model, self.testloader, self.device, config)  
      else:
        raise ValueError(f"Unknown defense type: {defense_type}")

    def aggregate_from_neighbors(self, neighbor_updates, defense_type="none", config=None, current_round=1):
       """
       Aggregate neighbor updates into a new global model for this client.
       Each neighbor update is expected as a tuple: (weights, num_samples, trigger_state).
       """
  
       if not neighbor_updates:
        # No neighbors sent updates, fallback to local model
        print(f"[Client {self.id}] No neighbor updates received. Keeping local model.")
        return self.model.state_dict()

       # --- Initialize a local per client defensive server  ---
       if self.agg_server is None:
          self._init_agg_server(defense_type, config)

       print(f"[CLIENT {self.id}] Started aggregation…")
       
       # --- Feed own updates to own server ---
       self.agg_server.set_params(self.get_params())
    
       if defense_type == "ubar":
          # UBAR: store own loss as local_loss
          local_eval = self.local_evaluate()
          self.agg_server.local_loss = local_eval.get("metrics", {}).get("loss", None)
       
       # --- Feed recieved neighbor updates ---
       valid_updates = 0
       for update in neighbor_updates:
          weights, num_samples, metrics, trigger_state = update
    
          # Standard FedAvg ingestion
          if weights is not None and num_samples is not None:
            self.agg_server.receive_update(weights, num_samples)
            valid_updates += 1

          # UBAR: need loss 
          if defense_type == "ubar" and metrics is not None :
              loss = metrics.get("loss", None)
              if loss is not None:
                  self.agg_server.receive_loss(float(loss))

       # --- No valid updates, fallback ---       
       if valid_updates == 0:
          print(f"[Client {self.id}] No valid neighbor updates. Keeping local model.")
          return self.model.state_dict()

       # --- Aggregate recieved updates ---
       if defense_type in ['balance', 'abalance', "dfldual", "ubar"]:
           agg_weights = self.agg_server.aggregate(current_round)  # returns a state_dict  
       else:    
           agg_weights = self.agg_server.aggregate()  # returns a state_dict

       for k in agg_weights.keys():
           agg_weights[k] = agg_weights[k].to(self.device)
       
       # --- Mixing with own update ---
       try:
         mix_ratio = config.get("mix_ratio", 0.5)
         own_state = self.model.state_dict()
         mixed_state = {}
         for k in own_state.keys():
           mixed_state[k] = (
            mix_ratio * own_state[k] +
            (1 - mix_ratio) * agg_weights[k]
          )
       except Exception as e:
          print(f"[CLIENT {self.id}] Exception in mixing: {e}")

       # --- Load the new mixed weights into the client model ---
       self.model.load_state_dict(mixed_state)
      
       return mixed_state
 
    

     