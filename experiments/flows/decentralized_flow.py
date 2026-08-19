import time
import torch
import os
import multiprocessing as mp


from .topology import create_topology
from .flow_utils import evaluate_model_accuracy, evaluate_asr, recreate_asr_test_loader, save_clean_model, load_clean_model


def _get_prev_param_or_zero(prev_map, client_id, key, ref_tensor, device):
    prev_client_sd = prev_map.get(client_id)
    if not prev_client_sd:
        return torch.zeros_like(ref_tensor, device=device)
    prev_val = prev_client_sd.get(key)
    if prev_val is None:
        return torch.zeros_like(ref_tensor, device=device)
    return prev_val.to(device)

def _build_topology(env, config):
    if env.get("neighbors") is not None:
        neighbors = env["neighbors"]
        print("Topology already loaded; skipping generation.")
        return neighbors

    num_clients = config["num_clients"]
    topo_cfg = config.get("topology", {})
    mode = topo_cfg.get("mode", "single")
    single = topo_cfg.get("single", {})

    graph_type = single.get("type", "ring")
    n_clients  = single.get("n", num_clients)
    k          = single.get("k", 4)
    seed_graph = single.get("seed", config.get("seed"))
    params = single.get("params", {})
    extra_kwargs = {key: val for key, val in params.items()}

    if "spectral_gap" in single: extra_kwargs["spectral_gap"] = single["spectral_gap"]
    if "ensure_connected" in single: extra_kwargs["ensure_connected"] = single["ensure_connected"]
    if "max_additional_edge_attempts" in single: extra_kwargs["max_additional_edge_attempts"] = single["max_additional_edge_attempts"]
    if single.get("annotate_instance_id", True): extra_kwargs["instance_id"] = 0
    if n_clients != num_clients: env["num_clients"] = num_clients

    neighbors = create_topology(n=num_clients, k=k, graph_type=graph_type, seed=seed_graph, **extra_kwargs)
    env["neighbors"] = neighbors
    return neighbors


def _worker_wrapper_serial(args):
    """Worker wrapper for serial local training."""
    client, model_params, config, current_round, extra_args = args

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


def _worker_wrapper_parallel(args):
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
    clients = env['clients']
    device = config['device']

    neighbors = env.get('neighbors')

    if neighbors is None:
        neighbors =_build_topology(env, config)

    pool = None
    num_workers = config.get('num_parallel_clients', 1)
    if num_workers > 1:
        # Initialize multiprocessing pool
        try:
           mp.set_start_method('spawn', force=True)
        except RuntimeError:
           pass
        pool = mp.Pool(processes=num_workers)    

    prev_model_params_per_client = {c.id: None for c in clients}
    start_round = int(config.get('start_round', 1))
    total_rounds = int(config.get('num_rounds', 1))

    initial_pretrained_state = None
    pretrained_path = config.get('load_model_path', config.get('pretrained_path'))    
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Found pretrained model at {pretrained_path}. Loading into memory...")
        try:
            load_clean_model(path=pretrained_path, model=clients[0].model)
            initial_pretrained_state = {k: v.cpu().clone() for k, v in clients[0].model.state_dict().items()}
        except Exception:
            print(f"ERROR: Could not load pretrained model. Proceeding with random init.")

    backdoor_loader = env.get('backdoor_loader')
    asr_attacker_id = None  # cache key: which client's trigger this loader is bound to
    
    for current_round in range(start_round, total_rounds + 1):

        round_start = time.time()
        print(f"\n--- Decentralized Round {current_round}/{total_rounds} | device {device} ---")

        # 1. PREPARATION & TRAINING
       
        worker_args = []
        for client in clients:
            model_params = None 
            if current_round == start_round and initial_pretrained_state is not None:
                model_params = {k: v.clone() for k, v in initial_pretrained_state.items()}
            if model_params is None:
                model_params = client.get_params() if hasattr(client, "get_params") else client.model.state_dict()

            extra_args = {}
            if client.__class__.__name__ == 'NeurotoxinClient':
                agg_grad = {}
                for name, tensor in model_params.items():
                    prev_tensor = _get_prev_param_or_zero(prev_model_params_per_client, client.id, name, tensor, device)
                    agg_grad[name] = (tensor.to(device) - prev_tensor).clone()
                extra_args['prev_global_grad'] = agg_grad
                prev_model_params_per_client[client.id] = {k: v.detach().cpu().clone() for k, v in model_params.items()}

            
            worker_args.append((client, model_params, config, current_round, extra_args))

        t_train_start = time.time()
        if pool is None:
           # Serial Execution Loop
           print(f"Starting serial decentralized round {current_round}...")
           results = []
           for args in worker_args:
             res = _worker_wrapper_serial(args)
             results.append(res)

           updates_map = {cid: (weights, num_samples, metrics, updated_trigger_state) 
                                   for (cid, weights, num_samples, metrics, updated_trigger_state) in results} 
        else:     
           # Parallel local training
           print(f"Starting parallel decentralized round {current_round} with {num_workers} parallel clients...")
           results = pool.map(_worker_wrapper_parallel, worker_args)
           updates_map = {cid: (weights, num_samples, metrics, updated_trigger_state) for (cid, weights, num_samples, metrics, updated_trigger_state) in results}
              
        train_time = time.time() - t_train_start
        
        
        # 2. AGGREGATION
        t_agg_start = time.time()
        for client in clients:
            neighs = neighbors.get(client.id, [])
            collected = [updates_map[nid] for nid in neighs if nid in updates_map]
            if len(collected) == 0:
                continue

            if client.__class__.__name__ in ('NeurotoxinClient', 'A3FLClient', 'BadNetsClient', 'DBAClient', 'IBAClient', 'ScalingAttackClient') and current_round >= config.get('attack_start_round', 0):
                try:
                    client.aggregate_from_neighbors_attacker(collected, config=config, prev_global_params_per_client=prev_model_params_per_client, round_idx=current_round)
                except AttributeError:
                    client.aggregate_from_neighbors(collected, defense_type='none', config=None)
            else:
                try: 
                    client.aggregate_from_neighbors(collected, config.get('defense', 'none'), config, current_round)
                except Exception:
                    print(f"Client: {client.id} aggregation failed")    
            
            trigger_states = [c[3] for c in collected if c[3] is not None]
            if trigger_states:
                ts = trigger_states[0]
                if hasattr(client, 'trigger'): 
                    if getattr(client.trigger, 'generator', None) is not None:
                        try: client.trigger.generator.load_state_dict(ts)
                        except Exception: 
                             if hasattr(client.trigger, 'pattern'): client.trigger.pattern = ts
                    elif getattr(client.trigger, 'pattern', None) is not None: 
                        client.trigger.pattern = ts
        agg_time = time.time() - t_agg_start


        # 3. DYNAMIC TRIGGER & EVALUATION
        t_eval_start = time.time()
        
        is_attack_active = (config.get('attack', 'none') != 'none' 
                            and config.get('num_malicious', 0) > 0 
                            and config.get('attack_start_round', 100) <= current_round <= config.get('attack_end_round', 100))
        attack_flag_int = 1 if is_attack_active else 0

        if is_attack_active and config.get('attack') in ['a3fl', 'iba']:
            attacker = next((c for c in clients if c.__class__.__name__ in ['A3FLClient', 'IBAClient']), None)
            if attacker and (backdoor_loader is None or asr_attacker_id != attacker.id):
                # Build ONCE per attacker identity, not every round.
                # trigger.apply is read lazily at __getitem__, so later
                # updates to trigger.pattern / trigger.generator are
                # picked up automatically without rebuilding.
                backdoor_loader = recreate_asr_test_loader(
                    env['test_loader'], attacker.trigger, config.get('target_class', 0),
                    config.get('test_batch_size', 256), seed=config.get('seed', 42),
                    num_workers=config.get('asr_num_workers', 0),  # see note below
                    pin_memory=True
                )
                asr_attacker_id = attacker.id
                env['backdoor_loader'] = backdoor_loader

        main_accuracies = []
        asrs = []

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

            if backdoor_loader and current_round >= config.get('attack_start_round', 100):
                asr_val = evaluate_asr(client.model, backdoor_loader, device)
                if client.__class__.__name__ == 'BenignClient':
                    asrs.append(asr_val)
                    print(f"Round {current_round}: Client {client.id} ASR = {asr_val:.4f}")
                else:
                    print(f"Round {current_round}: Attacker {client.id} ASR = {asr_val:.4f}")    
        
        eval_time = time.time() - t_eval_start


        # --- Reporting ---
        main_acc = sum(main_accuracies) / len(main_accuracies) if main_accuracies else 0.0
        min_acc = min(main_accuracies) if main_accuracies else 0.0
        max_asr = max(asrs) if asrs else 0.0
        total_time = time.time() - round_start

        # PRINT TIMING BREAKDOWN
        print(f"⏱️  [Timing] Train: {train_time:.2f}s | Agg: {agg_time:.2f}s | Eval: {eval_time:.2f}s | Total: {total_time:.2f}s")
        print(f"Round {current_round}: Avg ACC = {main_acc:.4f}, Min ACC = {min_acc:.4f}, Max ASR = {max_asr:.4f}, Attack: {attack_flag_int}")
        logger.log_round(current_round, main_acc, min_acc, max_asr, attack_flag_int)

    if config.get("save_model", False):
        save_path = config.get("save_model_path", "experiments/pretrained_models/default.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 'clients' is a list. Find the first BenignClient safely, 
        # fallback to the very first client if no BenignClient exists.
        first_benign = next((c for c in clients if c.__class__.__name__ == 'BenignClient'), clients[0])
        
        # Save the state dictionary using PyTorch
        torch.save(first_benign.model.state_dict(), save_path)
        print(f"\n💾 [SAVED] Pretrained model saved successfully to: {save_path}")