#!/usr/bin/env python3
"""
run_parallel.py — main launcher for centralized or decentralized FL workflows.
Usage:
PYTHONPATH=. python experiments/run_parallel.py --config experiments/configs/base_config.yaml
"""

import argparse
import yaml
import torch
from src.utils import ResultsLogger
from experiments.flows.centralized_flow import run_centralized_flow
from experiments.flows.decentralized_flow import run_decentralized_flow
from experiments.flows.flow_utils import set_seed, prepare_environment

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))

    # -------------------------
    # Respect device from YAML
    # -------------------------
    if "device" in config and config["device"] is not None:
        # String from YAML, e.g. "cuda:1"
        device_str = config["device"]
        print(f"[INFO] Using device from YAML: {device_str}")
        config["device"] = torch.device(device_str)
    else:
        # Fallback to automatic detection
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] YAML had no device, using auto device: {auto_device}")
        config["device"] = torch.device(auto_device)

    logger = ResultsLogger(config['experiment_name'], config)
    print("\n--- Configuration ---")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("---------------------")

    env = prepare_environment(config)

    if config.get('flow', 'centralized') == 'decentralized':
        run_decentralized_flow(env, config, logger)
    else:
        run_centralized_flow(env, config, logger)

    print("\n--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
