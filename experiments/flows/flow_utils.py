import copy
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

# --- Dataset adapters & models ---
from src.datasets.imagenet import TinyImageNetAdapter
from src.datasets.mnist import MNISTAdapter
from src.datasets.femnist import FEMNISTAdapter
from src.datasets.fashionmnist import FashionMNISTAdapter
from src.datasets.cifar10 import CIFAR10Adapter
from src.datasets.gtsrb import GTSRBAdapter
from src.datasets.har import HARAdapter


from src.models.mnist_cnn import LeNet5, EMNIST_CNN, MNISTNet, Fashion_CNN
from src.models.tiny_resnet18 import ResNet18_TinyImageNet
from src.models.cifar_resnet18 import CifarNetGN
from src.models.gtsrb_cnn import GTSRB_CNN
from src.models.har_mlp import HAR_MLP


# backdoor test loader for ASR evaluation
from src.datasets.backdoor import BackdoorDataset

# clients / attacks / triggers / selectors 
from src.fl.baseclient import BenignClient
from src.attacks.badnets_client import BadNetsClient
from src.attacks.scaling_client import ScalingAttackClient
from src.attacks.neurotoxin_client import NeurotoxinClient
from src.attacks.a3fl_client import A3FLClient
from src.attacks.dba_client import DBAClient
from src.attacks.iba_client import IBAClient
from src.attacks.selectors.randomselector import RandomSelector
from src.attacks.triggers.patch import PatchTrigger
from src.attacks.triggers.a3fl import A3FLTrigger
from src.attacks.triggers.distributed import DBATrigger
from src.attacks.triggers.iba import IBATrigger

from src.attacks.triggers.har_trigger import HARTrigger



from src.attacks.krum_client import KrumAttackClient
from src.attacks.trim_client import TrimAttackClient
from src.attacks.gauss_client import GaussianAttackClient
from src.attacks.label_flip_client import LabelFlipClient
from src.attacks.feature_client import FeatureAttackClient





def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def average_state_dicts(state_dicts, weights=None):
    """
    Weighted average of a list of state_dicts. Returns CPU tensors (float).
    """
    if not state_dicts:
        return {}
    if weights is None:
        weights = [1.0] * len(state_dicts)
    total_w = float(sum(weights))
    avg = {}
    keys = list(state_dicts[0].keys())
    for k in keys:
        acc = None
        for sd, w in zip(state_dicts, weights):
            t = sd[k].detach().cpu().float()
            acc = t * float(w) if acc is None else (acc + t * float(w))
        avg[k] = (acc / total_w).clone()
    return avg

def recreate_asr_test_loader(testloader, trigger, target_class, batch_size, seed=42):
    """
    Build a poisoned evaluation dataloader using the updated trigger/pattern.
    Works for both A3FL (pattern) and IBA (generator).
    """
    # --- Create poisoned evaluation dataset ---
    poisoned_dataset = BackdoorDataset(
        original_dataset=testloader.dataset,
        trigger_fn=trigger.apply,
        target_label=target_class,
        poison_fraction=1.0,
        seed=seed,
        poison_exclude_target=True
    )
    # --- Create ASR loader ---
    return DataLoader(
        poisoned_dataset,
        batch_size=batch_size,
        shuffle=True
    )

def evaluate_model_accuracy(model, test_loader, device):
    """Evaluate classification accuracy (0..1)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return (correct / total) if total > 0 else 0.0


def evaluate_asr(model, backdoor_loader: DataLoader, device: torch.device):
    if backdoor_loader is None: return 0.0
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in backdoor_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    asr = (correct / total) if total > 0 else 0.0
    return asr
  

def _make_adapter_and_model(config):
    """Return (adapter, model_cls, in_channels, image_size) based on config.dataset"""
    ds = config.get('dataset', 'mnist').lower()
    if ds == 'mnist':
        adapter = MNISTAdapter(root="data", download=True)
        #model_cls = lambda: LeNet5(num_classes=10)
        model_cls = lambda: MNISTNet() 
        in_channels = 1
        image_size = (28, 28)
    elif ds == 'fashionmnist':
        adapter = FashionMNISTAdapter(root="data", download=True)
        model_cls = lambda: Fashion_CNN(num_classes=10) 
        in_channels = 1
        image_size = (28, 28)

    elif ds == 'femnist':
        adapter = FEMNISTAdapter(root="data", train=True, download=True)
        #model_cls = lambda: EMNIST_CNN(num_classes=62)
        model_cls = lambda: LeNet5(num_classes=62)
        in_channels = 1
        image_size = (28, 28)
    elif ds == 'cifar10':
        adapter = CIFAR10Adapter(root="data", download=True)
        model_cls = lambda: CifarNetGN()
        in_channels = 3
        image_size = (32, 32)
    elif ds == 'gtsrb':
        adapter = GTSRBAdapter(root="data")
        model_cls = lambda: GTSRB_CNN(num_classes=43)
        in_channels = 3
        image_size = (32, 32)
    elif ds == 'har':
        adapter = HARAdapter(root="data", download=False)
        model_cls = lambda: HAR_MLP()
        in_channels = None
        image_size = None

    elif ds == 'tinyimagenet':
        adapter = TinyImageNetAdapter(root="data", download=True)
        model_cls = lambda: ResNet18_TinyImageNet(num_classes=200)
        in_channels = 3
        image_size = (64, 64)    

    else:
        raise NotImplementedError(f"Dataset '{ds}' not implemented in utils_fl.prepare_environment.")
    return adapter, model_cls, in_channels, image_size

import random
from typing import Dict, Iterable, Optional, List

def select_malicious(nodes: Iterable[int], num_malicious: int,
                            strategy: str = "first", seed: Optional[int] = None) -> List[int]:
    """Simple selector supporting only 'first' and 'random' strategies."""
    node_list = sorted(list(nodes))
    if num_malicious <= 0:
        return []
    if num_malicious > len(node_list):
        raise ValueError("num_malicious can't exceed number of nodes")
    if strategy == "first":
        return node_list[:num_malicious]
    elif strategy == "random":
        rng = random.Random(seed)
        return rng.sample(node_list, k=num_malicious)
    else:
        raise ValueError("Unsupported malicious selection strategy: {}".format(strategy))

def build_clients(client_loaders: Dict[int, object],
                  test_loader,
                  model_cls,
                  config: Dict,
                  selector,
                  trigger):
    """
    Minimal change: compute malicious_ids using 'first' or 'random' strategy and log them.
    config keys used:
      - num_malicious
      - malicious_strategy ('first' or 'random') default 'first'
      - malicious_seed (optional, used for 'random')
    """
    nodes = list(client_loaders.keys())
    num_malicious = config.get('num_malicious', 0)
    strategy = config.get('malicious_strategy', 'first')
    seed = config.get('seed', None)

    malicious_ids = select_malicious(nodes, num_malicious, strategy=strategy, seed=seed)

    # Log chosen malicious ids for reproducibility (print or logging)
    print(f"[build_clients] malicious_strategy={strategy}, seed={seed}, malicious_ids={malicious_ids}")

    clients = []
    for cid, loader in client_loaders.items():
        base_kwargs = {
            'id': cid,
            'trainloader': loader,
            'testloader': test_loader,
            'model': model_cls().to(config['device']),
            'lr': config['lr'],
            'weight_decay': 0.0,
            'epochs': config['local_epochs'],
            'device': config['device']
        }

        is_malicious = cid in malicious_ids

        if is_malicious:
            attack_type = config.get('attack', 'none')
            malicious_kwargs = {'selector': selector, 'target_class': config.get('target_class', 0)}

            if attack_type == 'badnets':
                client = BadNetsClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                             attack_start_round=config.get('attack_start_round', 0),
                                             attack_end_round=config.get('attack_end_round', -1),
                                             poison_fraction=config.get('poisoning_rate', 0.1),
                                             malicious_epochs=config.get('malicious_epochs', 10))
            elif attack_type == 'scaling':
                client = ScalingAttackClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                             attack_start_round=config.get('attack_start_round', 0),
                                             attack_end_round=config.get('attack_end_round', -1),
                                             scale_factor=config.get('scale_factor', 1.0),
                                             num_total_clients=config.get('num_clients'),
                                             num_malicious_clients=config.get('num_malicious'),
                                             poison_fraction=config.get('poisoning_rate', 0.1),
                                             malicious_epochs=config.get('malicious_epochs', 10))
            elif attack_type == 'neurotoxin':
                client = NeurotoxinClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                          attack_start_round=config.get('attack_start_round', 0),
                                          attack_end_round=config.get('attack_end_round', -1),
                                          mask_k_percent=config.get('mask_k_percent', 0.05),
                                          poison_fraction=config.get('poisoning_rate', 0.1),
                                          malicious_epochs=config.get('malicious_epochs', 10))
            elif attack_type == 'a3fl':
                client = A3FLClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                    attack_start_round=config.get('attack_start_round', 0),
                                    attack_end_round=config.get('attack_end_round', -1),
                                    poison_fraction=config.get('poisoning_rate', 0.1),
                                    malicious_epochs=config.get('malicious_epochs', 10))
            elif attack_type == 'iba':
                client = IBAClient(**base_kwargs, **malicious_kwargs, trigger=trigger,
                                   attack_start_round=config.get('attack_start_round', 0),
                                   attack_end_round=config.get('attack_end_round', -1),
                                   poison_fraction=config.get('poisoning_rate', 0.1),
                                   malicious_epochs=config.get('malicious_epochs', 10))
                                   
            elif attack_type == 'dba':
                # If trigger is a list indexed by malicious-order, map to the position in malicious_ids
                try:
                    pos = malicious_ids.index(cid)
                    client_trigger = trigger[pos] if isinstance(trigger, (list, tuple)) else trigger
                except ValueError:
                    client_trigger = trigger
                client = DBAClient(**base_kwargs, **malicious_kwargs, trigger=client_trigger,
                                   attack_start_round=config.get('attack_start_round', 0),
                                   attack_end_round=config.get('attack_end_round', -1),
                                   poison_fraction=config.get('poisoning_rate', 0.1))

            ### Classic untargeted poisoning attacks
            elif attack_type == 'krum':
                client = KrumAttackClient(**base_kwargs, epsilon = config.get('krum_epsilon', 0.01)) 
            elif attack_type == 'trim':
                client = TrimAttackClient(**base_kwargs, scale = config.get('trim_scale', 5.0))     
            elif attack_type == 'gauss':
                client = GaussianAttackClient(**base_kwargs, variance = config.get('gauss_variance', 200))   
                                  
            else:
                client = BenignClient(**base_kwargs)
        else:
            client = BenignClient(**base_kwargs)
        clients.append(client)
    return clients

def prepare_environment(config):
    """
    Prepare dataset adapters, client loaders, model factory, clients list, server model, and backdoor eval loader.
    Returns: env dict with keys:
      - 'clients', 'client_loaders', 'test_loader', 'model_cls', 'server_model', 'trigger', 'backdoor_loader'
    """
    device = config.get('device', torch.device('cpu'))

    adapter, model_cls, in_channels, image_size = _make_adapter_and_model(config)

    # --------------------------
    # 1️⃣ Build client loaders
    # --------------------------
    client_loaders = adapter.get_client_loaders(
        config['num_clients'],
        config["iid"],
        config['batch_size'],
        config['seed'],
        alpha=config.get('dirichlet_alpha', 0.5)
    )
    test_loader = adapter.get_test_loader(config['test_batch_size'])

    # --------------------------
    # 2️⃣ Print Data Distribution Stats
    # --------------------------
    print("\n[prepare_environment] 📊 Client Data Distribution Summary")
    client_sizes = {
        cid: len(loader.dataset) if hasattr(loader, 'dataset') else 0
        for cid, loader in client_loaders.items()
    }

    total_samples = sum(client_sizes.values())
    avg_samples = total_samples / len(client_sizes) if client_sizes else 0
    min_samples = min(client_sizes.values()) if client_sizes else 0
    max_samples = max(client_sizes.values()) if client_sizes else 0

    print(f"  Total samples: {total_samples}")
    print(f"  Clients: {len(client_sizes)}  |  Avg per client: {avg_samples:.2f}")
    print(f"  Min: {min_samples}, Max: {max_samples}")

    print("  Per-client sample counts:")
    for cid, n in sorted(client_sizes.items()):
        perc = (n / total_samples * 100) if total_samples > 0 else 0
        print(f"    Client {cid:02d}: {n:5d} samples ({perc:5.2f}%)")
    print("-" * 60)

    # Optional: print heterogeneity (std dev)
    import statistics
    if len(client_sizes) > 1:
        std_samples = statistics.stdev(client_sizes.values())
        print(f"  Standard deviation across clients: {std_samples:.2f}\n")

    # --------------------------
    # 3️⃣ Build trigger (for attacks)
    # --------------------------
    trigger = None
    attack = config.get('attack')

    # HAR dataset and Badnet attack
    if  config.get('dataset') == "har":
        if attack in ['badnets', 'neurotoxin', 'scaling'] :
             trigger = HARTrigger( trigger_features=[0, 10, 20],
                                      trigger_value=3.0)  

    elif config.get('attack', 'none') != 'none':
        trigger_pos = (image_size[0] - 4, image_size[1] - 4)
        if attack == 'a3fl':
                trigger = A3FLTrigger(
                    position=trigger_pos, size=(3, 3),
                    in_channels=in_channels, image_size=image_size,
                trigger_epochs=config.get('trigger_epochs', 5),
                trigger_lr=config.get('trigger_lr', 0.01)
            )
        elif attack == 'iba':
            if config.get('dataset') == 'femnist':
                from src.models.unet import FEMNISTAutoencoder
                unet_generator = FEMNISTAutoencoder(in_channel=1, out_channel=1)
            else:
                from src.models.unet import UNet
                unet_generator = UNet(in_channel=in_channels, out_channel=in_channels)
            trigger = IBATrigger(unet_model=unet_generator, 
                                 trigger_epochs=config.get('trigger_epochs', 5))
        elif attack == 'dba':
            shard_locations = [(0, 0), (2, 0), (0, 2), (2, 2)]
            trigger = [
                DBATrigger(client_id=i, shard_locations=shard_locations,
                           global_position=(image_size[0]-5, image_size[1]-5),
                           patch_size=(2, 2), color=(1.0,)*in_channels)
                for i in range(config.get('num_malicious', 0))
            ]
         
        else:
            trigger = PatchTrigger(position=trigger_pos, size=(3, 3), color=(1.0,)*in_channels)

    # --------------------------
    # 4️⃣ Selector and Clients
    # --------------------------
    selector = RandomSelector(config.get('poisoning_rate', 0.1))
    clients = build_clients(client_loaders, test_loader, model_cls, config, selector, trigger)

    # --------------------------
    # 5️⃣ Server model and backdoor loader
    # --------------------------
    server_model = model_cls().to(device)

    backdoor_loader_for_eval = None
    if config.get('attack', 'none') != 'none':
        eval_trigger_for_dba = PatchTrigger(
            position=(image_size[0]-5, image_size[1]-5),
            size=(4, 4), color=(1.0,)*in_channels
        ) if config.get('attack') == 'dba' else None

        initial_eval_trigger = eval_trigger_for_dba if config.get('attack') == 'dba' else trigger

        poisoned_dataset = BackdoorDataset(
            original_dataset=test_loader.dataset,
            trigger_fn=initial_eval_trigger.apply,
            target_label=config.get('target_class', 0),
            poison_fraction=1.0,
            seed=config.get('seed', 42),
            poison_exclude_target=True
        )
        backdoor_loader_for_eval = DataLoader(
            poisoned_dataset,
            batch_size=config['batch_size'],
            shuffle=True
            )    

    # --------------------------
    # 6️⃣ Return environment dictionary
    # --------------------------
    env = {
        'clients': clients,
        'client_loaders': client_loaders,
        'test_loader': test_loader,
        'model_cls': model_cls,
        'server_model': server_model,
        'trigger': trigger,
        'backdoor_loader': backdoor_loader_for_eval
    }
    return env

# --- Additional Helpers to save and load models ---
def save_clean_model(path: str, model: torch.nn.Module):
    """
    Atomically save only the model.state_dict() so any node with the same architecture
    (or compatible keys) can load it later.
    """
    tmp = path + ".tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)  # atomic on most OSes

def load_clean_model(path: str, model: torch.nn.Module, device: str = "cpu", strict: bool = True):
    """
    Load a saved state_dict into `model`. Use strict=False if you expect slight key mismatches
    (e.g., extra buffers or missing keys).
    Returns the loaded state_dict for inspection.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=strict)
    return state
