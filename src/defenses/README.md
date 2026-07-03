# Defense Implementations

This repository provides implementations of state-of-the-art defense mechanisms for **Decentralized Federated Learning (DFL)** and **Federated Learning (FL)** against Byzantine and backdoor attacks.


---

# 1. DFL Byzantine-Robust Aggregation

These methods are designed to operate in decentralized settings where each participant acts as its own defensive aggregator.

## UBAR

**Reference:** Guo et al., 2021

UBAR performs robust neighbor selection by:

- Computing pairwise distances between received models.
- Selecting the closest neighbors.
- Removing updates that significantly increase the local validation loss.
- Aggregating the remaining updates using **Trimmed Mean**.

---

## SCCLIP

**Reference:** He et al., 2022

SCCLIP limits the magnitude of incoming updates by clipping every received model so that its norm does not exceed the norm of the client's own update.


---

## BALANCE

**Reference:** Fang et al., 2024

BALANCE performs a model acceptance test before aggregation.

A received model is accepted only if its Euclidean distance from the local model remains below a predefined exponentially decaying threshold:

\[
d_{ij} \le \tau
\]



---

## Adaptive BALANCE (ABALANCE)

**Proposed Extension**

ABALANCE extends BALANCE by replacing the fixed threshold with an adaptive one.

Main improvements:

- Adaptive threshold based on the **median** of received distances.
- Robust dispersion estimation using **Median Absolute Deviation (MAD)**.
- Temporal consistency by constraining threshold changes using the previous communication round.
- More tolerant during early training on highly non-IID data.
- More resistant to stealthy backdoor attacks.

---

## DFL-Dual

**Reference:** Sun et al., 2024

DFL-Dual jointly analyzes updates in:

- **Model space**
- **Data space**

The defense:

1. Computes Euclidean distances between local models.
2. Reconstructs lightweight dummy datasets using model inversion.
3. Measures Wasserstein distances between reconstructed datasets.
4. Combines both metrics into a unified similarity score.
5. Performs two-stage clustering to identify trusted participants before aggregation.

---

# 2. FL Backdoor Defenses Adapted to DFL

The following centralized FL defenses have been adapted for decentralized aggregation.

---

## DeepSight

**Reference:** Rieger et al., 2022

DeepSight is a data-free clustering defense that analyzes client updates using multiple feature spaces, including:

- NEUP
- Decision Difference (DDif)
- Cosine similarity

Consensus clustering is then used to identify malicious updates before robust aggregation.

---

## FLAME

**Reference:** Nguyen et al., 2022

FLAME detects malicious updates through clustering of last-layer model parameters.

The defense then:

- Removes suspicious clusters
- Clips update norms
- Injects adaptive Gaussian noise
- Aggregates the remaining updates using FedAvg

---

## Similarity of Partial Parameters (SPP)

**Reference:** Wang et al., 2025

SPP randomly samples subsets of model parameters and compares received updates against the local reference model using similarity metrics such as:

- Cosine similarity
- Euclidean distance

Only sufficiently similar updates participate in aggregation.

---

## Multi-Metrics Adaptive Defense (MMAD)

**Reference:** Huang et al., 2023

MMAD combines multiple distance metrics:

- L1 distance
- L2 distance
- Cosine distance

These features are transformed into covariance-aware anomaly scores that rank participants according to their likelihood of being malicious.

---

## Norm Clipping / Weak Differential Privacy

**Reference:** Sun et al., 2019

Norm Clipping limits the magnitude of every received update before aggregation.

Weak-DP further injects low-variance Gaussian noise after aggregation to smooth the influence of adversarial updates while providing lightweight privacy protection.

---

## Multi-Krum

**Reference:** Blanchard et al., 2017

Multi-Krum computes pairwise distances among received updates and selects the most mutually consistent subset.

Only these selected updates are aggregated.

---

## Trimmed Mean

**Reference:** Yin et al., 2018

Trimmed Mean performs coordinate-wise robust aggregation by:

1. Sorting parameter values.
2. Removing the largest and smallest values.
3. Averaging the remaining parameters.

This reduces the influence of Byzantine updates.

## Coordinate-wise Median

**Reference:** Yin et al., 2018

The Coordinate-wise Median defense performs robust aggregation by computing the median of each model parameter independently across all received updates.

For every parameter coordinate:

1. Collect the corresponding values from all received client updates.
2. Compute the coordinate-wise median.
3. Assemble the aggregated model from the resulting median parameters.
