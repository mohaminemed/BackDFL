import copy
import random
import time
import torch
import os
import multiprocessing as mp
import pickle
import statistics


from .topology import create_topology
from .flow_utils import evaluate_model_accuracy, evaluate_asr, recreate_asr_test_loader, load_clean_model
from src.attacks.aggregation.mixing import AttackerAggregationMixin
from src.defenses.detection_metrics import detection_logger, setup_detection_logging
from src.timing_metrics import timing_logger, setup_timing_logging


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


def _select_a3fl_leader(a3fl_attackers, current_round, attack_start_round, hold_rounds, fixed_idx=None):
    """
    Sticky round-robin leader selection (same as the centralized flow): the
    same leader is held for `hold_rounds` consecutive attack rounds, then
    rotates. FIX: the previous version hard-coded `idx = 1` (debug leftover),
    so the leader never rotated. Set config['a3fl_fixed_leader_idx'] to get a
    fixed leader on purpose.
    """
    n = len(a3fl_attackers)
    if fixed_idx is not None:
        return a3fl_attackers[int(fixed_idx) % n]
    rounds_since_start = max(0, current_round - attack_start_round)
    return a3fl_attackers[(rounds_since_start // hold_rounds) % n]


def _push_pattern_to_attackers(clients, pattern):
    """Overwrite trigger.pattern of every A3FL attacker (main-process copies)."""
    for c in clients:
        if c.__class__.__name__ == 'A3FLClient':
            c.trigger.pattern = pattern.clone()


def _sync_a3fl_triggers(clients, config, current_round, env):
    """
    Enforce a single shared A3FL trigger across all compromised clients.

    - The leader runs Phase-1 trigger optimization; every other attacker
      skips it and poisons with the shared pattern.
    - FIX: the leader now also STARTS from the current shared pattern (before,
      it restarted from its own stale pattern, so progress never accumulated
      across rounds/leaders).

    Non-leaders use the pattern broadcast at the end of the PREVIOUS round
    (one round of staleness with parallel execution).

    Returns the leader id (or None when no A3FL attack is active this round).
    """
    a3fl_attackers = [c for c in clients if c.__class__.__name__ == 'A3FLClient']
    if not a3fl_attackers:
        return None

    warmup = int(config.get('a3fl_trigger_warmup_rounds', 5))
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
    opt_start = attack_start_round - warmup
    is_a3fl_opt_round = (config.get('attack', 'none') == 'a3fl'
                     and opt_start <= current_round <= attack_end_round)
    if not is_a3fl_opt_round:
        return None
    
    hold_rounds = max(1, int(config.get('a3fl_leader_hold_rounds', 10)))
    leader = _select_a3fl_leader(a3fl_attackers, current_round, attack_start_round,
                                 hold_rounds, config.get('a3fl_fixed_leader_idx'))
    shared_pattern = env.get('shared_a3fl_pattern')

    for atk in a3fl_attackers:
        atk.skip_trigger_training = (atk.id != leader.id)
        if shared_pattern is not None:
            atk.trigger.pattern = shared_pattern.clone()   # leader included

    return leader.id


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

    t_local = time.perf_counter()
    update = client.local_train(config.get('local_epochs', 1), current_round, **extra_args)
    train_s = time.perf_counter() - t_local

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

    return (client.id, update['weights'], update['num_samples'], update['metrics'], updated_trigger_state, train_s)


def _worker_wrapper_parallel(args):
    """
    Worker wrapper for parallel local training. With mp.Pool (spawn) the client
    is a pickled COPY: anything changed here (trigger.pattern, ...) is only
    visible to the main process through the returned trigger state.
    """
    client, model_params, config, current_round, extra_args = args

    if model_params is not None:
        try:
            client.set_params(model_params)
        except Exception:
            sd = client.model.state_dict()
            for k, v in model_params.items():
                sd[k] = v.clone()
            client.model.load_state_dict(sd)

    t_local = time.perf_counter()
    update = client.local_train(config.get('local_epochs', 1), current_round, **extra_args)
    train_s = time.perf_counter() - t_local

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

    return (client.id, update['weights'], update['num_samples'], update['metrics'], updated_trigger_state, train_s)

def run_decentralized_flow(env, config, logger):
    clients = env['clients']
    device = config['device']

    neighbors = env.get('neighbors')

    if neighbors is None:
        neighbors =_build_topology(env, config)

    pool = None
    num_workers = config.get('num_parallel_clients', 1)
    if num_workers > 1:
        try:
           mp.set_start_method('spawn', force=True)
        except RuntimeError:
           pass
        pool = mp.Pool(processes=num_workers)

    prev_model_params_per_client = {c.id: None for c in clients}

    setup_detection_logging(config)
    setup_timing_logging(config)
    start_round = int(config.get('start_round', 1))
    total_rounds = int(config.get('num_rounds', 1))

    env.setdefault('shared_a3fl_pattern', None)

    initial_pretrained_state = None
    pretrained_path = config.get('load_model_path', config.get('pretrained_path'))

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Found pretrained model at {pretrained_path}. Loading into memory...")
        try:
            load_clean_model(path=pretrained_path, model=clients[0].model)
            initial_pretrained_state = {k: v.cpu().clone() for k, v in clients[0].model.state_dict().items()}
        except Exception:
            print(f"ERROR: Could not load pretrained model. Proceeding with random init.")

    # Storage for Statistics-space analysis
    save_statistics = config.get('save_statistics', False)
    statistic_analysis_path = config.get('statistic_analysis_path', None)
    statistic_analysis_round = int(config.get('statistic_analysis_round', total_rounds))


    if save_statistics and statistic_analysis_path:
      os.makedirs(os.path.dirname(statistic_analysis_path) or '.', exist_ok=True)

      statistic_analysis_data = {
        'client_updates': {},
        'malicious_ids': [],
        'global_model': None,
    }

    backdoor_loader = env.get('backdoor_loader')
    asr_attacker_id = None   # IBA: which attacker the cached loader is bound to
    eval_trigger = None      # A3FL: dedicated trigger used ONLY for evaluation

    for current_round in range(start_round, total_rounds + 1):

        round_start = time.time()
        print(f"\n--- Decentralized Round {current_round}/{total_rounds} | device {device} ---")

        # 0. A3FL: pick leader + broadcast shared pattern BEFORE local training.
        a3fl_leader_id = _sync_a3fl_triggers(clients, config, current_round, env)

        # 1. PREPARATION & TRAINING
        worker_args = []
        for client in clients:
          
            if current_round == start_round and initial_pretrained_state is not None:
                model_params = {k: v.clone() for k, v in initial_pretrained_state.items()}
            else:
                model_params = client.get_params() if hasattr(client, "get_params") else client.model.state_dict()

            extra_args = {}
            if client.__class__.__name__ == 'NeurotoxinClient':
               agg_grad = {}

               if config.get("target_neighbor") :
                   prev_client_id = client.id
               else:
                   prev_client_id = neighbors.get(client.id, [client.id])[0]

               for name, tensor in model_params.items():
                  prev_tensor = _get_prev_param_or_zero(
                    prev_model_params_per_client,
                    prev_client_id,
                    name,
                    tensor,
                    device
                  )
                  agg_grad[name] = (
                    tensor.to(device) - prev_tensor
                  ).clone()

               extra_args['prev_global_grad'] = agg_grad

               prev_model_params_per_client[client.id] = {
                 k: v.detach().cpu().clone()
                 for k, v in model_params.items()
               }

            worker_args.append((client, model_params, config, current_round, extra_args))

        t_train_start = time.time()
        if pool is None:
           print(f"Starting serial decentralized round {current_round}...")
           results = []
           for args in worker_args:
             res = _worker_wrapper_serial(args)
             results.append(res)

           updates_map = {cid: (weights, num_samples, metrics, updated_trigger_state)
                                   for (cid, weights, num_samples, metrics, updated_trigger_state, _) in results}
        else:
           print(f"Starting parallel decentralized round {current_round} with {num_workers} parallel clients...")
           results = pool.map(_worker_wrapper_parallel, worker_args)
           updates_map = {cid: (weights, num_samples, metrics, updated_trigger_state) for (cid, weights, num_samples, metrics, updated_trigger_state, _) in results}

        train_time = time.time() - t_train_start
        attack_active_round = (
            config.get('attack', 'none') != 'none' and
            config.get('attack_start_round', 0) <= current_round <= config.get('attack_end_round', 10**9)
        )
        for res, cl in zip(results, clients):
            is_benign = cl.__class__.__name__ == 'BenignClient'
            timing_logger.record('train', res[5], 'benign' if is_benign else 'attacker')

        # A3FL: store the leader's freshly-trained pattern and mirror it onto
        # the main-process attacker copies (they are never updated by the pool
        # workers).
        if a3fl_leader_id is not None and a3fl_leader_id in updates_map:
            leader_trigger_state = updates_map[a3fl_leader_id][3]
            if leader_trigger_state is not None:
                env['shared_a3fl_pattern'] = leader_trigger_state.clone()
                _push_pattern_to_attackers(clients, env['shared_a3fl_pattern'])

        # NET CHURN
        net_churn_rho = config.get("net_churn_rho", 0.0)

        client_ids = [client.id for client in clients if client.__class__.__name__ == 'BenignClient']
        mal_client_ids = [client.id for client in clients if client.__class__.__name__ != 'BenignClient']

        if 'debug_attacker_ids' not in config:
           config['debug_attacker_ids'] = mal_client_ids
           print(f"Added malicious client IDs to config for detection metrics: {mal_client_ids}")

        if net_churn_rho > 0:
           rng = random.Random(config.get("seed", 42) + current_round)
           n_churn = int(len(client_ids) * net_churn_rho)
           churned_ids = set(rng.sample(client_ids, n_churn))
        else:
           churned_ids = set()

        active_client_ids = set(client_ids) - churned_ids

        print(
          f"[Round {current_round}] Active clients: "
          f"{len(active_client_ids)}/{len(client_ids)} "
          f"({len(active_client_ids) / len(client_ids):.1%})"
        )

        # 2. AGGREGATION
        t_agg_start = time.time()
        for client in clients:

            if client.id in churned_ids:
               continue

            neighs = neighbors.get(client.id, [])
            collected = [(nid, updates_map[nid]) for nid in neighs if nid in updates_map and (nid in active_client_ids or nid in mal_client_ids)]

            if len(collected) == 0:
                continue

            neighbor_updates = [c[1] for c in collected]

            if isinstance(client, AttackerAggregationMixin) and current_round >= config.get('attack_start_round', 0):
                try:
                    with timing_logger.measure('attacker_agg'):
                        client.aggregate_from_neighbors_attacker(neighbor_updates, config=config, prev_global_params_per_client=prev_model_params_per_client, round_idx=current_round)
                except AttributeError:
                    client.aggregate_from_neighbors(collected, defense_type='none', config=None)
            else:
                try:
                     client.aggregate_from_neighbors(collected, config.get('defense', 'none'), config, current_round)
                except Exception:
                    print(f"Client: {client.id} aggregation failed")

            # Process trigger states.
            # A3FL attackers are skipped -- their trigger is the single
            # orchestrated shared pattern (env['shared_a3fl_pattern']); letting
            # them adopt "the first neighbor's" pattern overwrote it with
            # whichever (possibly stale, non-leader) pattern came first.
            if client.__class__.__name__ != 'A3FLClient':
                trigger_states = [c[3] for c in neighbor_updates if c[3] is not None]
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

        # ============================================================
        # SAVE MODEL SNAPSHOT FOR STATISTICS ANALYSIS
        # ============================================================
        if save_statistics and current_round == statistic_analysis_round:

            print(
                f"[Statistics Analysis] Saving model snapshot "
                f"for round {current_round} -> {statistic_analysis_path}"
            )

            client_updates = {}

            for client in clients:
                try:
                    state_dict = client.model.state_dict()

                    client_updates[client.id] = {
                        k: v.detach().cpu().clone()
                        for k, v in state_dict.items()
                    }

                except Exception as e:
                    print(
                        f"[Statistics Analysis] Could not save client "
                        f"{client.id}: {e}"
                    )

            malicious_ids = [
                client.id
                for client in clients
                if client.__class__.__name__ != 'BenignClient'
            ]

            if initial_pretrained_state is not None:
                global_model = {
                    k: v.detach().cpu().clone()
                    for k, v in initial_pretrained_state.items()
                }
            else:
                reference_client = clients[0]

                global_model = {
                    k: v.detach().cpu().clone()
                    for k, v in reference_client.model.state_dict().items()
                }

            statistic_analysis_data = {
                'client_updates': client_updates,
                'malicious_ids': malicious_ids,
                'global_model': global_model,
            }
            with open(statistic_analysis_path, 'wb') as f:
                pickle.dump(statistic_analysis_data, f)

            print(
                f"[Statistics Analysis] Saved {len(client_updates)} clients "
                f"and {len(malicious_ids)} malicious clients."
            )


        # 3. DYNAMIC TRIGGER & EVALUATION
        t_eval_start = time.time()

        is_attack_active = (config.get('attack', 'none') != 'none'
                            and config.get('num_malicious', 0) > 0
                            and config.get('attack_start_round', 100) <= current_round <= config.get('attack_end_round', 100))
        attack_flag_int = 1 if is_attack_active else 0
        attack_type = config.get('attack')

        if is_attack_active and attack_type == 'a3fl':
            # Evaluate with a dedicated trigger carrying the latest shared
            # pattern. The old loader was bound once to the first attacker's
            # main-process trigger, whose pattern is not what was trained
            # (pool copies) and was also overwritten by neighbor propagation.
            # Rebuilt every round: the pattern changes each round and
            # BackdoorDataset caches ("Caching complete").
            shared = env.get('shared_a3fl_pattern')
            if shared is not None:
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
                # NOTE: same pool caveat applies to the IBA generator.
                backdoor_loader = recreate_asr_test_loader(
                    env['test_loader'], attacker.trigger, config.get('target_class', 0),
                    config.get('test_batch_size', 256), seed=config.get('seed', 42),
                    num_workers=0,
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

            if backdoor_loader is not None and current_round >= config.get('attack_start_round', 100):
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
        mean_asr = sum(asrs) / len(asrs) if asrs else 0.0
        median_asr = statistics.median(asrs) if asrs else 0.0
        contam_thr = config.get('contamination_threshold', 0.5)
        contaminated_frac = sum(a > contam_thr for a in asrs) / len(asrs) if asrs else 0.0
        total_time = time.time() - round_start

        print(f"⏱️  [Timing] Train: {train_time:.2f}s | Agg: {agg_time:.2f}s | Eval: {eval_time:.2f}s | Total: {total_time:.2f}s")
        print(f"Round {current_round}: Avg ACC = {main_acc:.4f}, Min ACC = {min_acc:.4f}, Max ASR = {max_asr:.4f}, Mean ASR = {mean_asr:.4f}, Contaminated = {contaminated_frac:.2f}, Attack: {attack_flag_int}")
        logger.log_round(current_round, main_acc, min_acc, max_asr, attack_flag_int,
                         mean_asr, median_asr, contaminated_frac)
        timing_logger.end_round(current_round, attack_active_round, train_wall=train_time, agg_wall=agg_time,
                                eval_wall=eval_time, round_wall=total_time)

        if config.get("save_model", True) and current_round in config.get("save_model_rounds", [total_rounds]):
            save_path = config.get("save_model_path", "experiments/pretrained_models/default.pth")
            save_path_round = save_path.replace(".pth", f"_round{current_round}.pth")
            os.makedirs(os.path.dirname(save_path_round), exist_ok=True)

            first_benign = next((c for c in clients if c.__class__.__name__ == 'BenignClient'), clients[0])

            torch.save(first_benign.model.state_dict(), save_path_round)
            print(f"\n💾 [SAVED] Pretrained model saved successfully to: {save_path_round}")

    detection_logger.log_summary()
    timing_logger.log_summary()
