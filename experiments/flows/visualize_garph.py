import yaml
import networkx as nx
import matplotlib.pyplot as plt


from .topology import create_topology, compute_spectral_gap_from_adj, approximate_parameter_for_target_spectral_gap
from .flow_utils import select_malicious


# ---------------------------------------------------------------------
# Configuration Section 
# ---------------------------------------------------------------------


# Toggle: use predefined topology or regenerate one
USE_FIXED_TOPOLOGY = True

# Parameters for generated topology (ignored if USE_FIXED_TOPOLOGY=True)
GEN_N = 20
GEN_K = 7
GEN_GRAPH_TYPE = "erdos_renyi"
GEN_SEED = 42
GEN_P = 0.08

# Malicious node selection strategy
NUM_MALICIOUS = 7
MAL_STRATEGY = "random"  # "first" or "random"
MAL_SEED = 42


# ---------------------------------------------------------------------
# Visualization Helper
# ---------------------------------------------------------------------
def visualize_topology(topology, malicious_ids, topo_name):
    """Draw and save the topology with malicious nodes highlighted in red."""
    G = nx.Graph()
    for node, neighbors in topology.items():
        for nbr in neighbors:
            G.add_edge(node, nbr)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    node_colors = ["red" if n in malicious_ids else "skyblue" for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    plt.tight_layout()
    plt.axis("off")
    plt.savefig(f"experiments/graphs/dfl_topology_{topo_name}.pdf")


# ---------------------------------------------------------------------
# Build Topology Function
# ---------------------------------------------------------------------
def _build_topology(config):
    """
    Wrapper that extracts topology configuration and calls create_topology
    with the correct argument structure:

        create_topology(n, k, graph_type="random", seed=None, **kwargs)

    All extended parameters (p, beta, m, rewiring_prob, etc.) must be passed
    inside **kwargs exactly as expected by the topology generator.
    """

  
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
    k          = single.get("k", 7)
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

    print(f"Topology generated ({graph_type}, n={num_clients}, k={k}, seed={seed_graph}, params={extra_kwargs}):")
    print("Topology (first 8 nodes):", {i: neighbors[i] for i in range(min(8, len(neighbors)))})

    return neighbors


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Choose topology
    config_path = "experiments/configs/base_template.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    topology = _build_topology(config)

    # Choose malicious nodes (using your imported function)
    malicious_ids = select_malicious(
        topology.keys(),
        NUM_MALICIOUS,
        strategy=MAL_STRATEGY,
        seed=MAL_SEED
    )

    visualize_topology(topology, malicious_ids, topo_name=config.get("topology", {}).get("single", {}).get("name", "default"))
