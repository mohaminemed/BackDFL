
import time
import torch
import os
import multiprocessing as mp
from src.fl.baseserver import FedAvgAggregator

from .topology import create_topology
from .flow_utils import evaluate_model_accuracy, average_state_dicts, evaluate_asr, recreate_asr_test_loader, save_clean_model, load_clean_model


def _get_prev_param_or_zero(prev_map, client_id, key, ref_tensor, device):
    """
    Return prev_map[client_id][key] moved to device, or a zero tensor of same shape.
    """
    prev_client_sd = prev_map.get(client_id)
    if not prev_client_sd:
        return torch.zeros_like(ref_tensor, device=device)
    prev_val = prev_client_sd.get(key)
    if prev_val is None:
        return torch.zeros_like(ref_tensor, device=device)
    return prev_val.to(device)

def _build_topology(env, config):
    """
    Wrapper that extracts topology configuration and calls create_topology
    with the correct argument structure:

        create_topology(n, k, graph_type="random", seed=None, **kwargs)

    All extended parameters (p, beta, m, rewiring_prob, etc.) must be passed
    inside **kwargs exactly as expected by the topology generator.
    """

    # ----------------------------------------------------------------------
    # If topology already exists in env → skip
    # ----------------------------------------------------------------------
    if env.get("neighbors") is not None:
        neighbors = env["neighbors"]
        print("Topology already loaded; skipping generation.")
        print("Topology (first 8 nodes):",
              {i: neighbors[i] for i in range(min(8, len(neighbors)))})
        return neighbors

    num_clients = config["num_clients"]

    # ----------------------------------------------------------------------
    # Parse config
    # ----------------------------------------------------------------------
    topo_cfg = config.get("topology", {})
    mode = topo_cfg.get("mode", "single")

    if mode != "single":
        raise ValueError(
            "Only topology.mode='single' is supported. "
            "Sweeps must be handled at experiment-harness level."
        )

    single = topo_cfg.get("single", {})

    # Required basic parameters (direct args)
    graph_type = single.get("name", "topology")
    graph_type = single.get("type", "ring")
    n_clients  = single.get("n", num_clients)
    k          = single.get("k", 4)
    seed_graph = single.get("seed", config.get("seed"))

    # Extended kwargs (passed directly to create_topology)
    #
    # IMPORTANT:
    # - These must match exactly the names expected by create_topology.
    # - The user may embed p, beta, rewiring_prob, m, etc. inside params{}.
    #
    params = single.get("params", {})

    # kwargs consumed by create_topology
    extra_kwargs = {}

    # If params contains topology generator parameters → unpack properly
    for key, val in params.items():
        extra_kwargs[key] = val

    # Auxiliary config (allowed to pass through **kwargs)
    if "spectral_gap" in single:
        extra_kwargs["spectral_gap"] = single["spectral_gap"]

    if "ensure_connected" in single:
        extra_kwargs["ensure_connected"] = single["ensure_connected"]

    if "max_additional_edge_attempts" in single:
        extra_kwargs["max_additional_edge_attempts"] = single["max_additional_edge_attempts"]

    # Instance annotation is an optional kwarg
    if single.get("annotate_instance_id", True):
        extra_kwargs["instance_id"] = 0

    # ----------------------------------------------------------------------
    # Override n_clients if num_clients is explicitly set
    # ----------------------------------------------------------------------
    if n_clients != num_clients:
        print(f"[Topology] Overriding num_clients {n_clients} → {num_clients}")
        env["num_clients"] = num_clients

    # ----------------------------------------------------------------------
    # Generate topology
    # ----------------------------------------------------------------------
    neighbors = create_topology(
        n=num_clients,
        k=k,
        graph_type=graph_type,
        seed=seed_graph,
        **extra_kwargs
    )

    env["neighbors"] = neighbors

    print(f"Topology generated ({graph_type}, n={num_clients}, k={k}, seed={seed_graph}, params={extra_kwargs}):")

    return neighbors


def _worker_wrapper(args):
    """
    Worker wrapper for parallel local training.

    Expected args:
        (client, model_params, config, current_round, extra_args)
    Returns:
        (client_id, weights, num_samples, trigger_state)
    """
    client, model_params, config, current_round, extra_args = args

    # Load provided model params into the client model
    if model_params is not None:
        try:
            client.set_params(model_params)
        except Exception:
            sd = client.model.state_dict()
            for k, v in model_params.items():
                sd[k] = v.clone()
            client.model.load_state_dict(sd)

    # Run local training
    update = client.local_train(config.get('local_epochs', 1), current_round, **extra_args)

    # Safely extract trigger state for different trigger types (IBA, A3FL)
    updated_trigger_state = None
    is_attack_round = (
        config.get('attack', 'none') != 'none' and
        config.get('attack_start_round', 0) <= current_round <= config.get('attack_end_round', 10**9)
    )
    if is_attack_round : 
        if client.__class__.__name__ == 'IBAClient':
            state_dict = client.trigger.generator.state_dict()
            updated_trigger_state = {k: v.cpu().clone() for k, v in state_dict.items()}
        elif client.__class__.__name__ == 'A3FLClient':
            updated_trigger_state = client.trigger.pattern.cpu().clone()

    return (client.id, update['weights'], update['num_samples'], update['metrics'], updated_trigger_state)

def run_decentralized_flow(env, config, logger):
    """
    Run a decentralized FL workflow.

    Args:
        env (dict): Environment containing clients and topology info
        config (dict): Experiment configuration
        logger (ResultsLogger): Logger for round results
    """
    clients = env['clients']
    device = config['device']
    num_clients = len(clients)
    neighbors = env.get('neighbors')

    # Build topology if not provided
    if neighbors is None:
        neighbors =_build_topology(env, config)
        print("Topology (first 8 nodes):", {i: neighbors[i] for i in range(min(8, len(neighbors)))})

    # Initialize multiprocessing pool
    num_workers = config.get('num_parallel_clients', min(8, num_clients))
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    pool = mp.Pool(processes=num_workers)

    # Track previous model params per client
    prev_model_params_per_client = {c.id: None for c in clients}

    total_rounds = int(config.get('num_rounds', 1))
    for round_idx in range(total_rounds):
        round_start = time.time()
        current_round = round_idx + 1
        
        print(f"\n--- Decentralized Round {current_round}/{total_rounds} | device {device} ---")

        # Prepare per-client args
        worker_args = []
        for client in clients:
            
            # Load a clean model for testing
            if current_round == 100 and config['dataset'] != "har" and config['attack'] == "a3fl":
                #model_params = load_clean_model(path=f"experiments/models/none_none_fashionmnist_decentralized_25.pth", model=client.model)
                #model_params = load_clean_model(path=f"experiments/models/none_none_cifar10_decentralized_50.pth", model=client.model)
                print(f"Client {client.id}: Loaded clean model for round 1")
            else:
                model_params = client.get_params() if hasattr(client, "get_params") else client.model.state_dict()

            extra_args = {}

            # Neurotoxin-specific: compute gradient delta
            if client.__class__.__name__ == 'NeurotoxinClient':
                
                agg_grad = {}
                for name, tensor in model_params.items():
                    prev_tensor = _get_prev_param_or_zero(prev_model_params_per_client, client.id, name, tensor, device)
                    agg_grad[name] = (tensor.to(device) - prev_tensor).clone()
                extra_args['prev_global_grad'] = agg_grad
                prev_model_params_per_client[client.id] = {k: v.detach().cpu().clone() for k, v in model_params.items()}

            worker_args.append((client, model_params, config, current_round, extra_args))

        # Parallel local training
        results = pool.map(_worker_wrapper, worker_args)
        updates_map = {cid: (weights, num_samples, metrics, updated_trigger_state) for (cid, weights, num_samples, metrics, updated_trigger_state) in results}
        # --- Neighbor aggregation ---
        for client in clients:
            neighs = neighbors.get(client.id, [])
            collected = [updates_map[nid] for nid in neighs if nid in updates_map]
            if len(collected) == 0:
                continue

            # Attacker Aggregation
            if client.__class__.__name__ in ('NeurotoxinClient', 'A3FLClient', 'BadNetsClient', 'DBAClient',
                'IBAClient', 'ScalingAttackClient') and current_round >= config.get('attack_start_round', 0):
                try:
                    print(f"Attacker: {client.id} strating malicious aggregaion")
                    client.aggregate_from_neighbors_attacker(
                        collected,
                        config=config,
                        prev_global_params_per_client=prev_model_params_per_client,
                        round_idx=current_round
                    )
                except AttributeError:
                    print(f"Attacker: {client.id} falling back to default FedAvg then mixing")
                    client.aggregate_from_neighbors(collected, defense_type='none', config=None)

            # Bengin Aggregation        
            else:
                try: 
                    client.aggregate_from_neighbors(collected, config.get('defense', 'none'), config, current_round)
                except Exception:
                    print(f"Client: {client.id} aggeragation failed")    
            
            # Propagate trigger state (if any)
            trigger_states = [c[3] for c in collected if c[3] is not None]
            if trigger_states:
                ts = trigger_states[0]
                if hasattr(client, 'trigger'): # IBA
                    if getattr(client.trigger, 'generator', None) is not None:
                        try:
                            client.trigger.generator.load_state_dict(ts)
                            print(f"Client {client.id}: Loaded trigger generator state.")
                        except Exception:
                            if hasattr(client.trigger, 'pattern'):
                                client.trigger.pattern = ts
                    elif getattr(client.trigger, 'pattern', None) is not None: # A3FL
                        client.trigger.pattern = ts
                        print(f"Client {client.id}: Loaded trigger pattern state.")

        # --- Evaluation ---
        main_accuracies = []
        asrs = []
        backdoor_loader = env.get('backdoor_loader')
        target_class = config.get('target_class', None)

        for client in clients:
            acc = 0.0
            try:
                metrics = client.local_evaluate()
                acc = metrics['metrics']['accuracy']
            except Exception:
                acc = evaluate_model_accuracy(client.model, client.testloader, device)

            if client.__class__.__name__ == 'BenignClient':
                main_accuracies.append(acc)
                if client.id == 0 and current_round in [1, 25, 50]:
                    save_clean_model(path=f"experiments/models/{config['experiment_name']}_{current_round}.pth", model=client.model)

            if backdoor_loader and config.get('attack', 'none') != 'none' and current_round >= config.get('attack_start_round', 0):
                asr_val = evaluate_asr(client.model, backdoor_loader, device)
                if client.__class__.__name__ == 'BenignClient':
                    asrs.append(asr_val)
                    print(f"ROUND {current_round} CLIENT {client.id}: ACC = {acc:.4f}, ASR = {asr_val:.4f}")
                else:
                    print(f"ROUND {current_round} ATTACKER {client.id}: ACC = {acc:.4f}, ASR = {asr_val:.4f}")    

                    

        main_acc = sum(main_accuracies) / len(main_accuracies) if main_accuracies else 0.0
        min_acc = min(main_accuracies) if main_accuracies else 0.0
        max_asr = max(asrs) if asrs else 0.0
        round_time = time.time() - round_start

        attack_flag = 1 if (config.get('attack', 'none') != 'none' and
                            config.get('num_malicious', 0) > 0 and
                            config.get('attack_start_round', 0) <= current_round <= config.get('attack_end_round', 10**9)) else 0

        print(f"Round {current_round}: Avg ACC = {main_acc:.4f}, Min ACC = {min_acc:.4f}, Max ASR = {max_asr:.4f}, Attack Active: {attack_flag}, Time: {round_time:.2f}s")
        logger.log_round(current_round, main_acc, min_acc, max_asr, attack_flag)

    pool.close()
    pool.join()
