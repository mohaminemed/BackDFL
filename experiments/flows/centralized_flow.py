
import time
import multiprocessing as mp
import torch

from src.fl.baseserver import FedAvgAggregator
from src.defenses.clip_dp import NormClippingServer, WeakDPServer
from src.defenses.deepsight import DeepSightServer
from src.defenses.flame import FlameServer
from src.defenses.krum import MKrumServer
from src.defenses.trim import TrimmedMeanServer
from src.defenses.balance import BalanceServer  
from src.defenses.spp import SPPServer
from src.defenses.mmad import MultiMetricsServer
import os


from .flow_utils import evaluate_model_accuracy, evaluate_asr, set_seed, save_clean_model, load_clean_model

def _client_worker(args):
    """
    Worker used for parallel client local training in centralized flow.
    args: (client, model_params, config, current_round, extra_args)
    """
    client, model_params, config, current_round, extra_args = args
    if model_params is not None:
        try:
            client.set_params(model_params)
        except Exception:
            # fallback
            client.model.load_state_dict(model_params)

    # pass prev_global_grad if provided in extra_args
    update = client.local_train(config.get('local_epochs', 1), current_round, **extra_args)

    updated_trigger_state = None
    # some attacks may return trigger_state in update; capture for global update if needed
    if update.get('trigger_state') is not None:
        updated_trigger_state = update['trigger_state']

    return (client.id, update['weights'], update['num_samples'], updated_trigger_state)


def run_centralized_flow(env, config, logger):
    """
    Centralized FL (FedAvg + optional defenses) loop.
    env must contain: 'server_model', 'clients', 'test_loader', 'backdoor_loader'
    """
    server_model = env['server_model']
    clients = env['clients']
    test_loader = env['test_loader']
    backdoor_loader = env.get('backdoor_loader')
    device = config.get('device', torch.device('cpu'))

    # create server with chosen defense
    defense_type = config.get('defense', 'none')
    if defense_type == 'none':
        server = FedAvgAggregator(server_model, test_loader, device)
    elif defense_type == 'krum':
        server = MKrumServer(server_model, test_loader, device, config)
    elif defense_type == 'trim':
        server = TrimmedMeanServer(server_model, test_loader, device, config)    
    elif defense_type == 'clip':
        server = NormClippingServer(server_model, test_loader, device, config)
    elif defense_type == 'weakdp':
        server = WeakDPServer(server_model, test_loader, device, config)    
    elif defense_type == 'flame':
        server = FlameServer(server_model, test_loader, device, config)
    elif defense_type == 'deepsight':
        server = DeepSightServer(server_model, test_loader, device, config)
    elif defense_type == 'balance':
        server = BalanceServer(server_model, test_loader, device, config) 
    elif defense_type == 'spp':
        server = SPPServer(server_model, test_loader, device, config) 
    elif defense_type == 'mmad':
        server = MultiMetricsServer(server_model, test_loader, device, config)       
    else:
        raise ValueError(f"Unknown defense: {defense_type}")

    num_rounds = int(config.get('num_rounds', 1))
    prev_model_params = None

    # optional parallel pool
    num_workers = config.get('num_parallel_clients', 1)
    pool = None
    if num_workers and num_workers > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        pool = mp.Pool(processes=num_workers)
    
    for round_idx in range(num_rounds):
        round_start = time.time()
        current_round = round_idx + 1
        
        print(f"\n--- Centralized Round {current_round}/{num_rounds} ---")

        model_params = server.get_params()

        # compute agg_grad for Neurotoxin clients if needed
        agg_grad = None
        if prev_model_params:
            agg_grad = {name: model_params[name].to(device) - prev_model_params[name].to(device) for name in prev_model_params}
        prev_model_params = {k: v.detach().cpu().clone() for k, v in model_params.items()}

        # prepare worker args
        worker_args = []
        for client in clients:
            extra_args = {}
            if isinstance(client, object) and client.__class__.__name__ == 'NeurotoxinClient':
                extra_args['prev_global_grad'] = agg_grad
            worker_args.append((client, model_params, config, current_round, extra_args))

        # run local training (parallel or sequential)
        if pool is not None:
            results = pool.map(_client_worker, worker_args)
        else:
            results = [_client_worker(a) for a in worker_args]

        # collect updates into server
        updated_trigger_state = None
        for cid, weights, num_samples, trigger_state in results:
            server.receive_update(weights, num_samples)
            if updated_trigger_state is None and trigger_state is not None:
                updated_trigger_state = trigger_state

        # handle trigger state for global eval if any (your original code did this)
        if updated_trigger_state is not None:
            # best-effort: if IBA or A3FL adjust backdoor_loader accordingly
            pass
        if defense_type in ['balance', 'asg', 'aspp', 'saga']:
            server.aggregate(current_round)
        else:
            server.aggregate()
        main_metrics = server.evaluate()
        main_acc = main_metrics['metrics']['main_accuracy'] if 'metrics' in main_metrics else evaluate_model_accuracy(server.model, test_loader, device)
        asr = evaluate_asr(server.model, backdoor_loader, device) if current_round >= config.get('attack_start_round', 0) else 0.0
        target_class = config.get('target_class', None)
      

        round_time = time.time() - round_start
        attack_flag = 1 if (config.get('attack', 'none') != 'none' and config.get('num_malicious', 0) > 0 and config.get('attack_start_round', 0) <= current_round <= config.get('attack_end_round', 0x7fffffff)) else 0
        print(f"Round {current_round}: Main Acc = {main_acc:.4f}, Backdoor ASR = {asr:.4f}, Attack Active={attack_flag}, took {round_time:.2f}s")

        #logger.log_round(current_round, main_acc, main_acc, asr, attack_flag)
        logger.log_round(current_round, main_acc, main_acc, asr, attack_flag)
                

    if pool is not None:
        pool.close()
        pool.join()
