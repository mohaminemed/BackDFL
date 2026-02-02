# experiments/flows/topology.py
import random
import math
from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False

import numpy as np


def adjacency_dict_to_nx(adj: Dict[int, List[int]]) -> "nx.Graph":
    """Convert adjacency dict to NetworkX Graph (if nx available)."""
    if not _HAS_NX:
        raise RuntimeError("networkx is required for this conversion")
    G = nx.Graph()
    for u, nbrs in adj.items():
        for v in nbrs:
            G.add_edge(u, v)
    # ensure all nodes exist
    G.add_nodes_from(range(len(adj)))
    return G


def nx_to_adjacency_dict(G: "nx.Graph") -> Dict[int, List[int]]:
    """Convert NetworkX Graph to adjacency dict (sorted lists)."""
    return {i: sorted(list(map(int, G.neighbors(i)))) for i in G.nodes()}


def compute_spectral_gap_from_adj(adj: Dict[int, List[int]]) -> float:
    """
    Compute algebraic connectivity (the second-smallest eigenvalue of the Laplacian).
    Returns spectral gap >= 0. For a disconnected graph it will be 0.
    """
    n = len(adj)
    # build Laplacian matrix
    L = np.zeros((n, n), dtype=float)
    for i, nbrs in adj.items():
        L[i, i] = len(nbrs)
        for j in nbrs:
            L[i, j] -= 1.0
    # eigenvalues (symmetric matrix -> real)
    eigs = np.linalg.eigvalsh(L)  # numerically stable for symmetric
    eigs_sorted = np.sort(eigs)
    if n == 1:
        return 0.0
    # algebraic connectivity is the 2nd smallest eigenvalue (index 1)
    return float(eigs_sorted[1])


def create_topology(n: int,
                    k: int,
                    graph_type: str = "random",
                    seed: Optional[int] = None,
                    **kwargs) -> Dict[int, List[int]]:
    """
    Extended topology generator.
    graph_type: "random" | "ring" | "random_regular" | "line" | "cycle" |
                "erdos_renyi" | "watts_strogatz" | "barabasi_albert"
    kwargs:
      - p (float): for erdos_renyi
      - beta (float) or rewiring_prob (float): for watts_strogatz
      - m (int): for barabasi_albert (number of edges to attach)
      - attempts / max_tries: params for search routines (not used here)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if k >= n - 1:
        return {i: [j for j in range(n) if j != i] for i in range(n)}

    neighbors = {i: set() for i in range(n)}

    if graph_type == "ring":
        half = k // 2
        for i in range(n):
            for d in range(1, half + 1):
                neighbors[i].add((i + d) % n)
                neighbors[i].add((i - d) % n)
        if k % 2 == 1:
            for i in range(n):
                neighbors[i].add((i + half + 1) % n)

    elif graph_type == "line":
        # path graph with node i connected to i-1 and i+1 (degree <= 2). k is ignored (except to force higher deg)
        for i in range(n):
            if i - 1 >= 0:
                neighbors[i].add(i - 1)
            if i + 1 < n:
                neighbors[i].add(i + 1)
        # if k > 2, fallback to random extensions
        if k > 2:
            _fill_random_to_k(neighbors, n, k)

    elif graph_type == "cycle":
        for i in range(n):
            neighbors[i].add((i + 1) % n)
            neighbors[i].add((i - 1) % n)

    elif graph_type == "random_regular" and _HAS_NX:
        # may raise if impossible; leave caller to choose feasible k
        G = nx.random_regular_graph(k, n, seed=seed)
        for i in range(n):
            neighbors[i] = set(int(j) for j in G.neighbors(i))

    elif graph_type == "erdos_renyi":
        p = float(kwargs.get("p", k / max(1, n - 1)))
        # sample edges with prob p and then ensure symmetry and try to reach approximate degree k
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        # Optionally add random edges to reach exact k where possible
        _fill_random_to_k(neighbors, n, k)

    elif graph_type == "watts_strogatz" and _HAS_NX:
        # use networkx implementation to get small-world properties
        beta = float(kwargs.get("beta", kwargs.get("rewiring_prob", 0.1)))
        # ensure k is even for nx.watts_strogatz_graph
        k_even = k if k % 2 == 0 else k + 1
        G = nx.watts_strogatz_graph(n, k_even, beta, seed=seed)
        for i in range(n):
            neighbors[i] = set(int(j) for j in G.neighbors(i))
        # if original k was odd, remove one arbitrary neighbor to restore k (deterministic)
        if k % 2 == 1:
            for i in range(n):
                if len(neighbors[i]) > k:
                    neighbors[i].pop()

    elif graph_type == "barabasi_albert" and _HAS_NX:
        m = int(kwargs.get("m", max(1, min(k, n - 1))))
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        for i in range(n):
            neighbors[i] = set(int(j) for j in G.neighbors(i))
        _fill_random_to_k(neighbors, n, k)

    else:
        # default: symmetric random greedy fill (your original algorithm)
        for i in range(n):
            while len(neighbors[i]) < k:
                cand = random.randrange(n)
                if cand == i:
                    continue
                neighbors[i].add(cand)
                neighbors[cand].add(i)
                # if cand already full the add will be no-op but loop continues

    # convert sets to sorted lists
    return {i: sorted(list(neighbors[i])) for i in range(n)}


def _fill_random_to_k(neighbors: Dict[int, set], n: int, k: int):
    """Helper: randomly add symmetric edges until each node has degree >= k (best-effort)."""
    nodes = list(range(n))
    # try many times but prevent infinite loop
    max_iters = n * max(10, k * 5)
    it = 0
    while any(len(neighbors[i]) < k for i in nodes) and it < max_iters:
        a = random.randrange(n)
        b = random.randrange(n)
        if a == b:
            it += 1
            continue
        if len(neighbors[a]) >= k or len(neighbors[b]) >= k:
            it += 1
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
        it += 1
    # after attempts, accept degrees as-is (could be < k if impossible)


def approximate_parameter_for_target_spectral_gap(n: int,
                                                 k: int,
                                                 graph_type: str,
                                                 target_gap: float,
                                                 seed: Optional[int] = None,
                                                 param_bounds: Tuple[float, float] = (0.0, 1.0),
                                                 samples: int = 20,
                                                 tries_per_sample: int = 3,
                                                 verbosity: int = 0) -> Tuple[float, Dict[int, List[int]]]:
    """
    Approximate a graph parameter (for ER: p, WS: beta) that yields an algebraic connectivity
    close to target_gap. Returns (best_param, adjacency_dict).
    Method: sample parameters uniformly in bounds, for each sample generate 'tries_per_sample' graphs
    and take the one whose spectral gap is closest to target_gap. This is heuristic and not guaranteed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    best_param = None
    best_adj = None
    best_err = float("inf")

    for s in range(samples):
        param = random.uniform(param_bounds[0], param_bounds[1])
        for t in range(tries_per_sample):
            if graph_type == "erdos_renyi":
                adj = create_topology(n, k, graph_type="erdos_renyi", seed=random.randint(0, 2**31 - 1), p=param)
            elif graph_type == "watts_strogatz":
                # For WS we need networkx (otherwise fallback to naive ring+rewire is complicated)
                if not _HAS_NX:
                    raise RuntimeError("networkx is required to approximate Watts-Strogatz")
                adj = create_topology(n, k, graph_type="watts_strogatz", seed=random.randint(0, 2**31 - 1), beta=param)
            else:
                # unsupported graph_type for param search
                raise ValueError(f"approximate_parameter_for_target_spectral_gap unsupported for {graph_type}")

            gap = compute_spectral_gap_from_adj(adj)
            err = abs(gap - target_gap)
            if err < best_err:
                best_err = err
                best_param = param
                best_adj = adj
        if verbosity > 0:
            print(f"sample {s}: param={param:.4f}, best_err={best_err:.6f}")

    return best_param, best_adj


# Example usage helper
def usage_examples():
    """
    Prints usage example and experiment plan.
    """
    print("Example: create a 20-node ER graph with nominal degree ~6:")
    adj = create_topology(20, 7, graph_type="erdos_renyi", seed=42, p=0.08)
    #adj = create_topology(20, 7, graph_type="random_regular", seed=42, p=0.08)
    print("Topology (first 8 nodes):", {i: adj[i] for i in range(min(8, len(adj)))})
    gap = compute_spectral_gap_from_adj(adj)
    print(f"Algebraic connectivity (spectral gap): {gap:.6f}")

    print("\nTo search for a parameter that yields spectral gap ~0.5 (heuristic):")
    param, adj2 = approximate_parameter_for_target_spectral_gap(20, 7, "erdos_renyi", target_gap=0.5, samples=40, tries_per_sample=4, seed=1)
    print(f"best p ~ {param:.4f}, gap={compute_spectral_gap_from_adj(adj2):.6f}")



if __name__ == "__main__":
    # small self-test
    usage_examples()
