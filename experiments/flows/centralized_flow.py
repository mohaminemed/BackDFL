import time
import copy
import multiprocessing as mp
import torch
import os

from src.fl.baseserver import FedAvgAggregator
from src.defenses.clip_dp import NormClippingServer, WeakDPServer
from src.defenses.deepsight import DeepSightServer
from src.defenses.flame import FlameServer
from src.defenses.krum import MKrumServer
from src.defenses.trim import TrimmedMeanServer
from src.defenses.balance import BalanceServer
from src.defenses.spp import SPPServer
from src.defenses.mmad import MultiMetricsServer


from .flow_utils import evaluate_model_accuracy, evaluate_asr, recreate_asr_test_loader, load_clean_model


def _select_a3fl_leader(a3fl_attackers, current_round, attack_start_round, hold_rounds):
    """Sticky round-robin leader selection (deterministic, reproducible)."""
    n = len(a3fl_attackers)
    rounds_since_start = max(0, current_round - attack_start_round)
    leader_slot = (rounds_since_start // hold_rounds) % n
    return a3fl_attackers[leader_slot]


def _push_pattern_to_attackers(clients, pattern):
    """Overwrite trigger.pattern of every A3FL attacker (main-process copies)."""
    for c in clients:
        if c.__class__.__name__ == 'A3FLClient':
            c.trigger.pattern = pattern.clone()


def _sync_a3fl_triggers(clients, config, current_round, env):
    """
    Enforce ONE shared A3FL trigger across all compromised clients.

    - The leader (sticky round-robin) runs Phase-1 trigger optimization.
    - FIX: the leader now also STARTS from the current shared pattern (before,
      it restarted from its own stale/initial pattern, so the trigger never
      accumulated progress across rounds/leaders).
    - Non-leaders skip Phase 1 and poison with the shared pattern.

    Returns the leader id (or None when no A3FL attack is active).
    """
    a3fl_attackers = [c for c in clients if c.__class__.__name__ == 'A3FLClient']
    if not a3fl_attackers:
        return None

    attack_start_round = config.get('attack_start_round', 0)
    attack_end_round = config.get('attack_end_round', 10 ** 9)
    is_a3fl_attack_round = (
        config.get('attack', 'none') == 'a3fl' and
        attack_start_round <= current_round <= attack_end_round
    )
    if not is_a3fl_attack_round:
        for atk in a3fl_attackers:
            atk.skip_trigger_training = False
        return None

    hold_rounds = max(1, int(config.get('a3fl_leader_hold_rounds', 10)))
    leader = _select_a3fl_leader(a3fl_attackers, current_round, attack_start_round, hold_rounds)
    shared_pattern = env.get('shared_a3fl_pattern')

    for atk in a3fl_attackers:
        atk.skip_trigger_training = (atk.id != leader.id)
        if shared_pattern is not None:
            atk.trigger.pattern = shared_pattern.clone()   # leader included

    return leader.id


def _client_worker(args):
    """Worker for local training. NOTE: with mp.Pool (spawn) the client is a
    pickled COPY, so state changed here (trigger.pattern, model) is NOT visible
    in the main process -- it must be returned explicitly (trigger_state)."""
    client, model_params, config, current_round, extra_args = args
    if model_params is not None:
        try:
            client.set_params(model_params)
        except Exception:
            client.model.load_state_dict(model_params)

    update = client.local_train(config.get('local_epochs', 1), current_round, **extra_args)

    updated_trigger_state = None
    if update.get('trigger_state') is not None:
        updated_trigger_state = update['trigger_state']
    elif client.__class__.__name__ == 'A3FLClient':
        updated_trigger_state = client.trigger.pattern.cpu().clone()
    elif client.__class__.__name__ == 'IBAClient':
        try:
            state_dict = client.trigger.generator.state_dict()
            updated_trigger_state = {k: v.cpu().clone() for k, v in state_dict.items()}
        except Exception:
            pass

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
    asr_attacker_id = None          # IBA: which attacker the cached loader is bound to
    eval_trigger = None             # A3FL: dedicated trigger used ONLY for evaluation
    device = config.get('device', torch.device('cpu'))

    env.setdefault('shared_a3fl_pattern', None)

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

    num_workers = config.get('num_parallel_clients', 1)
    pool = None
    if num_workers and num_workers > 1:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        pool = mp.Pool(processes=num_workers)

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
            print("ERROR: Could not load pretrained model. Proceeding with random init.")
    model_params = None
    for current_round in range(start_round, total_rounds + 1):

        round_start = time.time()
        print(f"\n--- Centralized Round {current_round}/{num_rounds} ---")

        a3fl_leader_id = _sync_a3fl_triggers(clients, config, current_round, env)

        if current_round == start_round and initial_pretrained_state is not None:
         model_params = {k: v.clone() for k, v in initial_pretrained_state.items()}
        else:
         model_params = {k: v.detach().clone() for k, v in server.get_params().items()}

        agg_grad = None
        if prev_model_params:
            agg_grad = {name: model_params[name].to(device) - prev_model_params[name].to(device) for name in prev_model_params}
        prev_model_params = {k: v.detach().cpu().clone() for k, v in model_params.items()}

        worker_args = []
        for client in clients:
            extra_args = {}
            if client.__class__.__name__ == 'NeurotoxinClient':
                extra_args['prev_global_grad'] = agg_grad
            worker_args.append((client, model_params, config, current_round, extra_args))

        if pool is not None:
            results = pool.map(_client_worker, worker_args)
        else:
            results = [_client_worker(a) for a in worker_args]

        updated_trigger_state = None
        for cid, weights, num_samples, trigger_state in results:
            server.receive_update(weights, num_samples)
            if updated_trigger_state is None and trigger_state is not None:
                updated_trigger_state = trigger_state
            if a3fl_leader_id is not None and cid == a3fl_leader_id and trigger_state is not None:
                env['shared_a3fl_pattern'] = trigger_state.clone() if hasattr(trigger_state, 'clone') else trigger_state

        is_attack_active = (config.get('attack', 'none') != 'none'
                            and config.get('num_malicious', 0) > 0
                            and config.get('attack_start_round', 100) <= current_round <= config.get('attack_end_round', 100))
        attack_flag_int = 1 if is_attack_active else 0
        attack_type = config.get('attack')

        # ---------------- Evaluation trigger / backdoor loader ----------------
        if is_attack_active and attack_type == 'a3fl':
            # The main-process attacker.trigger is never updated when the
            # pool is used (training happens on pickled copies), so the old
            # loader -- built once from attacker.trigger -- kept evaluating the
            # untrained 0.5 gray patch (Linf 0.500) instead of the trained,
            # shared trigger. Evaluate with a dedicated trigger carrying the
            # latest shared pattern, and rebuild the loader every round since
            # the pattern changes every round (BackdoorDataset may cache).
            shared = env.get('shared_a3fl_pattern')
            if shared is not None:
                _push_pattern_to_attackers(clients, shared)   # keep main-side copies consistent
                if eval_trigger is None:
                    ref = next(c for c in clients if c.__class__.__name__ == 'A3FLClient')
                    eval_trigger = copy.deepcopy(ref.trigger)
                eval_trigger.pattern = shared.clone()
                backdoor_loader = recreate_asr_test_loader(
                    env['test_loader'], eval_trigger, config.get('target_class', 0),
                    config.get('test_batch_size', 256), seed=config.get('seed', 42),
                    num_workers=0,   # must be 0: workers would hold a stale trigger copy
                    pin_memory=True
                )
                env['backdoor_loader'] = backdoor_loader
                print(f"Round {current_round}: eval trigger = shared A3FL pattern "
                      f"(leader {a3fl_leader_id}, Linf norm {eval_trigger.pattern.abs().max().item():.3f})")
        elif is_attack_active and attack_type == 'iba':
            attacker = next((c for c in clients if c.__class__.__name__ in ['IBAClient', 'IBAGeneratorOnlyClient']), None)
            if attacker and (backdoor_loader is None or asr_attacker_id != attacker.id):
                # NOTE: same applies to the IBA generator; if the
                # pool is used, load `updated_trigger_state` into the eval
                # generator each round as well.
                backdoor_loader = recreate_asr_test_loader(
                    env['test_loader'], attacker.trigger, config.get('target_class', 0),
                    config.get('test_batch_size', 256), seed=config.get('seed', 42),
                    num_workers=config.get('asr_num_workers', 0),
                    pin_memory=True
                )
                asr_attacker_id = attacker.id
                env['backdoor_loader'] = backdoor_loader
                print(f"Round {current_round}: Built backdoor_loader for attacker {attacker.id} ({attacker.__class__.__name__})")

        can_eval_asr = backdoor_loader is not None and current_round >= config.get('attack_start_round', 100)

        if backdoor_loader is not None:
           attacker_classes = {'A3FLClient', 'IBAClient', 'IBAGeneratorOnlyClient', 'NeurotoxinClient'}
           mal = {c.id for c in clients if c.__class__.__name__ in attacker_classes}
           probe = copy.deepcopy(server.model)
           for name, keep in [("attackers-only", lambda i: i in mal),
                       ("benign-only", lambda i: i not in mal)]:
              sds = [w for cid, w, _, _ in results if keep(cid)]
              if not sds:
                continue
              avg = {k: torch.stack([sd[k].float() for sd in sds]).mean(0).to(sds[0][k].dtype)
                  for k in sds[0]}
              probe.load_state_dict(avg)
              print(f"Round {current_round}: {name} avg ASR = {evaluate_asr(probe, backdoor_loader, device):.4f}")     

        #if can_eval_asr:
        #    for client in clients:
        #        asr_val = evaluate_asr(client.model, backdoor_loader, device)
        #        tag = "Client" if client.__class__.__name__ == 'BenignClient' else "Attacker"
        #        print(f"Round {current_round}: {tag} {client.id} ASR = {asr_val:.4f}")

        server.aggregate()
        main_metrics = server.evaluate()
        main_acc = main_metrics['metrics']['main_accuracy'] if 'metrics' in main_metrics else evaluate_model_accuracy(server.model, test_loader, device)
        asr = evaluate_asr(server.model, backdoor_loader, device) if (backdoor_loader is not None and current_round >= config.get('attack_start_round', 0)) else 0.0

        round_time = time.time() - round_start
        print(f"Round {current_round}: Main Acc = {main_acc:.4f}, Backdoor ASR = {asr:.4f}, Attack Active={attack_flag_int}, took {round_time:.2f}s")

        logger.log_round(current_round, main_acc, main_acc, asr, attack_flag_int)

    if pool is not None:
        pool.close()
        pool.join()