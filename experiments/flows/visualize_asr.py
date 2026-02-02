import yaml
import networkx as nx
import matplotlib.pyplot as plt


from .topology import create_topology, compute_spectral_gap_from_adj, approximate_parameter_for_target_spectral_gap
from .flow_utils import select_malicious


# ---------------------------------------------------------------------
# Configuration Section (change these only)
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
NUM_MALICIOUS = 3
MAL_STRATEGY = "random"  # "first" or "random"
defense = "abalance"  # For visualization filename
MAL_SEED = 42

# KRUM Final ASR at round 48 (client_id -> ASR), Clustered 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8558, 1: 0.8898, 2: 0.8101,
    3: 0.0301, 4: 0.3059, 5: 0.3530,
    6: 0.0334, 7: 0.0308, 8: 0.5444,
    9: 0.0310, 10: 0.3290, 11: 0.0324,
    12: 0.5005, 13: 0.3434, 14: 0.4548,
    15: 0.0256, 16: 0.0264,
    17: 0.4633,
    18: 0.3416,
    19: 0.3310
}
# KRUM Final ASR at round 48 (client_id -> ASR), Random 3 attackers: 0,3,8
FINAL_ASR = {
    0: 0.8804,
    1: 0.0348,
    2: 0.0379,
    3: 0.8253,
    4: 0.0328,
    5: 0.0390,
    6: 0.0306,
    7: 0.0335,
    8: 0.7819,
    9: 0.0365,
    10: 0.0238,
    11: 0.0359,
    12: 0.0325,
    13: 0.0413,
    14: 0.0272,
    15: 0.0273,
    16: 0.0287,
    17: 0.0376,
    18: 0.0419,
    19: 0.0307
}
# ABALANCE Final ASR at round 50 (client_id -> ASR), Random 3 attackers: 0,3,8
FINAL_ASR = {
    0: 0.0489,
    1: 0.0318,
    2: 0.0395,
    3: 0.0724,
    4: 0.0348,
    5: 0.0420,
    6: 0.0334,
    7: 0.0321,
    8: 0.0416,
    9: 0.0329,
    10: 0.0290,
    11: 0.0333,
    12: 0.0305,
    13: 0.0368,
    14: 0.0330,
    15: 0.0347,
    16: 0.0295,
    17: 0.0376,
    18: 0.0358,
    19: 0.0373
}
# Final ABALANCE ASR at round 50 (client_id -> ASR), Clustered 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8171,
    1: 0.8626,
    2: 0.7993,
    3: 0.0332,
    4: 0.0348,
    5: 0.0504,
    6: 0.0302,
    7: 0.0301,
    8: 0.0728,
    9: 0.0272,
    10: 0.0275,
    11: 0.0363,
    12: 0.0362,
    13: 0.0403,
    14: 0.0326,
    15: 0.0342,
    16: 0.0305,
    17: 0.0401,
    18: 0.0452,
    19: 0.0375
}

# Final ASR at round 50 (client_id -> ASR), Deepsight, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8709,
    1: 0.8298,
    2: 0.8593,
    3: 0.3831,
    4: 0.3198,
    5: 0.4607,
    6: 0.3031,
    7: 0.4283,
    8: 0.3090,
    9: 0.2694,
    10: 0.3048,
    11: 0.3055,
    12: 0.3770,
    13: 0.3739,
    14: 0.3607,
    15: 0.2437,
    16: 0.3263,
    17: 0.3801,
    18: 0.4059,
    19: 0.2945
}

# Final ASR at round 50 (client_id -> ASR), Deepsight, Random 3 attackers: 0,3,8
FINAL_ASR = {
    0: 0.2785,
    1: 0.2309,
    2: 0.2514,
    3: 0.3097,
    4: 0.2633,
    5: 0.2866,
    6: 0.2698,
    7: 0.2196,
    8: 0.2890,
    9: 0.2483,
    10: 0.2179,
    11: 0.2553,
    12: 0.2627,
    13: 0.2493,
    14: 0.0617,
    15: 0.2289,
    16: 0.0598,
    17: 0.2579,
    18: 0.2694,
    19: 0.2650
}

# Final ASR at round 50 (client_id -> ASR), BALANCE, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.9120,
    1: 0.8555,
    2: 0.8353,
    3: 0.0356,
    4: 0.0704,
    5: 0.0512,
    6: 0.0464,
    7: 0.3257,
    8: 0.0889,
    9: 0.0331,
    10: 0.0543,
    11: 0.0393,
    12: 0.0522,
    13: 0.0512,
    14: 0.0485,
    15: 0.0243,
    16: 0.0304,
    17: 0.0665,
    18: 0.3384,
    19: 0.0522
}

# Final ASR at round 50 (client_id -> ASR), MMAD, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8591,
    1: 0.8456,
    2: 0.8451,
    3: 0.0357,
    4: 0.0366,
    5: 0.0394,
    6: 0.0337,
    7: 0.0333,
    8: 0.0353,
    9: 0.0341,
    10: 0.0367,
    11: 0.0408,
    12: 0.0364,
    13: 0.0333,
    14: 0.0332,
    15: 0.0352,
    16: 0.0341,
    17: 0.0399,
    18: 0.0349,
    19: 0.0359
}

# Final ASR at round 50 (client_id -> ASR), DFL-DUAL First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8437,
    1: 0.8952,
    2: 0.8437,
    3: 0.2264,
    4: 0.4885,
    5: 0.5900,
    6: 0.3566,
    7: 0.5441,
    8: 0.4385,
    9: 0.0620,
    10: 0.4301,
    11: 0.2210,
    12: 0.4926,
    13: 0.4561,
    14: 0.5179,
    15: 0.0614,
    16: 0.0546,
    17: 0.4256,
    18: 0.5784,
    19: 0.4430
}

# Final ASR at round 50 (client_id -> ASR), SCCLIP, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.9031,
    1: 0.9020,
    2: 0.8454,
    3: 0.4454,
    4: 0.4279,
    5: 0.5806,
    6: 0.3776,
    7: 0.5367,
    8: 0.4560,
    9: 0.4010,
    10: 0.4287,
    11: 0.3990,
    12: 0.5425,
    13: 0.5105,
    14: 0.4156,
    15: 0.3593,
    16: 0.4890,
    17: 0.4371,
    18: 0.5470,
    19: 0.4414
}

# Final ASR at round 50 (client_id -> ASR), UBAR, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8980,
    1: 0.9063,
    2: 0.8992,
    3: 0.0454,
    4: 0.2574,
    5: 0.3272,
    6: 0.0626,
    7: 0.3485,
    8: 0.2901,
    9: 0.0418,
    10: 0.3767,
    11: 0.0413,
    12: 0.4519,
    13: 0.2768,
    14: 0.4652,
    15: 0.0359,
    16: 0.0335,
    17: 0.2821,
    18: 0.3414,
    19: 0.4021
}

# Final ASR at round 50 (client_id -> ASR), FLAME, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8253,
    1: 0.8854,
    2: 0.7835,
    3: 0.0337,
    4: 0.0386,
    5: 0.0441,
    6: 0.0328,
    7: 0.0340,
    8: 0.0310,
    9: 0.0383,
    10: 0.0314,
    11: 0.0359,
    12: 0.0305,
    13: 0.0378,
    14: 0.0339,
    15: 0.0322,
    16: 0.0312,
    17: 0.0378,
    18: 0.0364,
    19: 0.0339
}
# Final ASR at round 50 (client_id -> ASR), FLAME, First 3 attackers: 0,1,2
FINAL_ASR = {
    0: 0.8253,
    1: 0.8854,
    2: 0.7835,
    3: 0.0337,
    4: 0.0386,
    5: 0.0441,
    6: 0.0328,
    7: 0.0340,
    8: 0.0310,
    9: 0.0383,
    10: 0.0314,
    11: 0.0359,
    12: 0.0305,
    13: 0.0378,
    14: 0.0339,
    15: 0.0322,
    16: 0.0312,
    17: 0.0378,
    18: 0.0364,
    19: 0.0339
}

# Final ASR at round 46 (client_id -> ASR), FLAME, Watts-Strogatz, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8522,
    1: 0.5291,
    2: 0.5494,
    3: 0.8652,
    4: 0.1074,
    5: 0.3287,
    6: 0.0352,
    7: 0.0390,
    8: 0.8746,
    9: 0.0337,
    10: 0.0281,
    11: 0.0365,
    12: 0.0309,
    13: 0.0321,
    14: 0.0309,
    15: 0.0317,
    16: 0.0297,
    17: 0.0402,
    18: 0.0359,
    19: 0.0370
}

# Final ASR at round 50 (client_id -> ASR), ABALANCE, Watts-Strogatz, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8648,
    1: 0.0857,
    2: 0.4025,
    3: 0.8238,
    4: 0.0510,
    5: 0.2472,
    6: 0.0513,
    7: 0.2469,
    8: 0.8810,
    9: 0.0343,
    10: 0.0299,
    11: 0.3577,
    12: 0.0329,
    13: 0.0340,
    14: 0.0307,
    15: 0.0315,
    16: 0.0301,
    17: 0.0364,
    18: 0.0339,
    19: 0.4625
}

# Final ASR at round 50 (client_id -> ASR), ABALANCE, Random regular, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8116,
    1: 0.0329,
    2: 0.0458,
    3: 0.7949,
    4: 0.0347,
    5: 0.0462,
    6: 0.0320,
    7: 0.0295,
    8: 0.8441,
    9: 0.0310,
    10: 0.0264,
    11: 0.0437,
    12: 0.0314,
    13: 0.0417,
    14: 0.0332,
    15: 0.0389,
    16: 0.0291,
    17: 0.0470,
    18: 0.0411,
    19: 0.0380
}

# Final ASR at round 50 (client_id -> ASR), FLAME, Random regular, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8163,
    1: 0.0341,
    2: 0.0387,
    3: 0.8556,
    4: 0.0390,
    5: 0.0527,
    6: 0.0342,
    7: 0.0313,
    8: 0.8389,
    9: 0.0353,
    10: 0.0295,
    11: 0.0386,
    12: 0.0361,
    13: 0.0451,
    14: 0.0287,
    15: 0.0340,
    16: 0.0309,
    17: 0.0435,
    18: 0.0380,
    19: 0.0376
}



################# Additionaal graphs with metrics #################

# ASR at round 46 (client_id -> ASR), ABALANCE, watts 0.2, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8825,
    1: 0.0505,
    2: 0.0419,
    3: 0.8190,
    4: 0.0344,
    5: 0.0398,
    6: 0.0361,
    7: 0.0321,
    8: 0.8783,
    9: 0.0325,
    10: 0.0314,
    11: 0.0644,
    12: 0.0337,
    13: 0.0357,
    14: 0.0344,
    15: 0.0306,
    16: 0.0321,
    17: 0.0430,
    18: 0.0353,
    19: 0.3897
}

# ASR at round 31 (client_id -> ASR), FLAME, watts 0.2, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8682,
    1: 0.0444,
    2: 0.3871,
    3: 0.8649,
    4: 0.0690,
    5: 0.0943,
    6: 0.0631,
    7: 0.0302,
    8: 0.8831,
    9: 0.0327,
    10: 0.0262,
    11: 0.0346,
    12: 0.0329,
    13: 0.0334,
    14: 0.0324,
    15: 0.0333,
    16: 0.0310,
    17: 0.0439,
    18: 0.0381,
    19: 0.0441
}

# ASR at round 32 (client_id -> ASR), FLAME, watts 0.3, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8473,
    1: 0.0454,
    2: 0.0424,
    3: 0.8340,
    4: 0.0355,
    5: 0.0404,
    6: 0.0356,
    7: 0.0362,
    8: 0.8854,
    9: 0.0342,
    10: 0.0255,
    11: 0.0312,
    12: 0.0261,
    13: 0.3218,
    14: 0.0280,
    15: 0.0370,
    16: 0.0281,
    17: 0.0449,
    18: 0.0439,
    19: 0.0354
}


# ASR at round 38 (client_id -> ASR), ABALANCE, watts 0.3, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8219,
    1: 0.0433,
    2: 0.0544,
    3: 0.8347,
    4: 0.0332,
    5: 0.0634,
    6: 0.0466,
    7: 0.2196,
    8: 0.8698,
    9: 0.0411,
    10: 0.0322,
    11: 0.0378,
    12: 0.0346,
    13: 0.0391,
    14: 0.0318,
    15: 0.0392,
    16: 0.0352,
    17: 0.0494,
    18: 0.0418,
    19: 0.0451
}


# ASR at round 50 (client_id -> ASR), ABALANCE, full graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8668,
    1: 0.0371,
    2: 0.0369,
    3: 0.8230,
    4: 0.0327,
    5: 0.0427,
    6: 0.0328,
    7: 0.0309,
    8: 0.8927,
    9: 0.0324,
    10: 0.0287,
    11: 0.0363,
    12: 0.0329,
    13: 0.0340,
    14: 0.0338,
    15: 0.0310,
    16: 0.0314,
    17: 0.0382,
    18: 0.0354,
    19: 0.0384
}

# ASR at round 49 (client_id -> ASR), FLAME, full graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8582,
    1: 0.0362,
    2: 0.0407,
    3: 0.7888,
    4: 0.0333,
    5: 0.0386,
    6: 0.0326,
    7: 0.0319,
    8: 0.8510,
    9: 0.0329,
    10: 0.0291,
    11: 0.0359,
    12: 0.0322,
    13: 0.0365,
    14: 0.0361,
    15: 0.0327,
    16: 0.0301,
    17: 0.0401,
    18: 0.0363,
    19: 0.0399
}

# ASR at round 50 (client_id -> ASR), ABALANCE, Erdos graph 0.16, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8795,
    1: 0.0390,
    2: 0.0406,
    3: 0.8515,
    4: 0.3215,
    5: 0.0447,
    6: 0.0358,
    7: 0.3973,
    8: 0.8837,
    9: 0.0410,
    10: 0.0351,
    11: 0.0386,
    12: 0.0307,
    13: 0.0373,
    14: 0.0355,
    15: 0.3899,
    16: 0.0327,
    17: 0.0417,
    18: 0.5243,
    19: 0.0363
}

# ASR at round 50 (client_id -> ASR), FLAME, Erdos graph, 0.16 Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8076,
    1: 0.0411,
    2: 0.0398,
    3: 0.8689,
    4: 0.0329,
    5: 0.0425,
    6: 0.0337,
    7: 0.0312,
    8: 0.9010,
    9: 0.0364,
    10: 0.0334,
    11: 0.0368,
    12: 0.0324,
    13: 0.0367,
    14: 0.0355,
    15: 0.0358,
    16: 0.0339,
    17: 0.0410,
    18: 0.0352,
    19: 0.0375
}

# ASR at round 50 (client_id -> ASR), ABALANCE, Erdos p=0.24, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8286,
    1: 0.0367,
    2: 0.0410,
    3: 0.8721,
    4: 0.0292,
    5: 0.0420,
    6: 0.0325,
    7: 0.0282,
    8: 0.9130,
    9: 0.0340,
    10: 0.0328,
    11: 0.0385,
    12: 0.0375,
    13: 0.0378,
    14: 0.0308,
    15: 0.0355,
    16: 0.0380,
    17: 0.0407,
    18: 0.0363,
    19: 0.0453
}

# ASR at round 50 (client_id -> ASR), FLAME, Erdos p=0.24, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8895,
    1: 0.0392,
    2: 0.0472,
    3: 0.8382,
    4: 0.0359,
    5: 0.0461,
    6: 0.0343,
    7: 0.0394,
    8: 0.8839,
    9: 0.0389,
    10: 0.0336,
    11: 0.0443,
    12: 0.0350,
    13: 0.0393,
    14: 0.0321,
    15: 0.0363,
    16: 0.0316,
    17: 0.0451,
    18: 0.0378,
    19: 0.0401
}

# ASR at round 50 (client_id -> ASR), FLAME, Cycle graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8988,
    1: 0.7230,
    2: 0.6269,
    3: 0.8112,
    4: 0.6170,
    5: 0.4186,
    6: 0.3701,
    7: 0.6785,
    8: 0.9725,
    9: 0.6428,
    10: 0.0651,
    11: 0.0288,
    12: 0.0301,
    13: 0.0315,
    14: 0.0318,
    15: 0.0240,
    16: 0.0326,
    17: 0.0543,
    18: 0.2897,
    19: 0.5427
}

# ASR at round 50 (client_id -> ASR), ABALANCE, Cycle graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.9264,
    1: 0.7777,
    2: 0.7298,
    3: 0.8987,
    4: 0.6010,
    5: 0.3431,
    6: 0.3241,
    7: 0.7352,
    8: 0.9580,
    9: 0.5868,
    10: 0.0732,
    11: 0.0454,
    12: 0.0379,
    13: 0.0363,
    14: 0.0352,
    15: 0.0298,
    16: 0.0298,
    17: 0.0522,
    18: 0.3667,
    19: 0.6594
}

# ASR at round 50 (client_id -> ASR), FLAME, Ring graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.9244,
    1: 0.4128,
    2: 0.3373,
    3: 0.8360,
    4: 0.3434,
    5: 0.4321,
    6: 0.3740,
    7: 0.0709,
    8: 0.9372,
    9: 0.0367,
    10: 0.0265,
    11: 0.0351,
    12: 0.0330,
    13: 0.0312,
    14: 0.0363,
    15: 0.0356,
    16: 0.0350,
    17: 0.2637,
    18: 0.4396,
    19: 0.4032
}

# ASR at round 49 (client_id -> ASR), ABALANCE, Ring graph, Attackers: 0,3,8
FINAL_ASR = {
    0: 0.8870,
    1: 0.0413,
    2: 0.0430,
    3: 0.8340,
    4: 0.0511,
    5: 0.0497,
    6: 0.0349,
    7: 0.0368,
    8: 0.9513,
    9: 0.2884,
    10: 0.0279,
    11: 0.0320,
    12: 0.0317,
    13: 0.0297,
    14: 0.0367,
    15: 0.0291,
    16: 0.0294,
    17: 0.0488,
    18: 0.0387,
    19: 0.0417
}


# ---------------------------------------------------------------------
# Visualization Helper
# ---------------------------------------------------------------------
def visualize_topology(topology, malicious_ids, topo_name):
    G = nx.Graph()
    for node, neighbors in topology.items():
        for nbr in neighbors:
            G.add_edge(node, nbr)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # ASR values in node order
    asr_vals = [FINAL_ASR.get(n, 0.0) for n in G.nodes()]

    # Draw nodes colored by ASR
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=asr_vals,
        node_size=850,
        cmap="Reds",
        vmin=0,
        vmax=1
    )

    # Mark attackers with black edge
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=malicious_ids,
        node_size=850,
        node_shape="s",
        edgecolors="black",
        linewidths=2,
        node_color=[FINAL_ASR[n] for n in malicious_ids],
        cmap="Reds",
        vmin=0,
        vmax=1
    )

    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=11)

    # Optional: annotate ASR numerically
    for n in G.nodes():
        plt.text(pos[n][0], pos[n][1]-0.08, f"{FINAL_ASR[n]:.2f}",
                 fontsize=10, ha="center")

    # Colorbar
    cbar = plt.colorbar(nodes)
    cbar.set_label("Final ASR", fontsize=16)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"experiments/graphs/dfl_asr_topology_{topo_name}_{MAL_STRATEGY}_{defense}.pdf")

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
        #env["num_clients"] = num_clients

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
    print("Topology (first 10 nodes):", {i: neighbors[i] for i in range(min(10, len(neighbors)))})

    return neighbors


# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Choose topology
    config_path = "experiments/configs/base_template_vis_topo.yml"
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
