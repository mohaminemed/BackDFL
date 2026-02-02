import math
import torch
import copy as cp
import numpy as np
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_weighted_average(updates: list[dict], weights=None) -> dict:
    """
    computes the average update from a list of updates.
    Args:
        updates (list[dict]): the list of updates to average.
        weights (list[float]): the weight of each update during averaging.
    Returns:
        dict: the weighted average of updates.
    """
    if weights is None:
        weights = [1] * len(updates)
    sum_weights = np.sum(weights)
    avg = cp.deepcopy(updates[0])
    for key in avg.keys():
        avg[key] = sum(param[key] * (weight / sum_weights) for param, weight in zip(updates, weights))
    return avg

def compute_diff(w1, w2):
    """
    computes the difference between two updates (grad). the difference is computed between weights and biases
    other keys are ignored.
    Args:
        w1 (dict): the first model update
        w2 (dict): the second model update
    Returns:
        dict: w1 - w2
    """
    res = cp.deepcopy(w1)
    for key in res.keys():
        if "weight" in key or "bias" in key:
            res[key] = w1[key] - w2[key]
        else:
            res[key].zero_()    
    return res

def get_direction(updates, ref_update, alpha=0.999, print_stats=False):
    """
    returns the change of direction between the average of a list of updates (potential next global update)
    and a reference model (previous global model)
    """
    avg = compute_weighted_average(updates)
    diff = compute_diff(avg, ref_update) # avg - ref_update
    directions = cp.deepcopy(diff)

    thresholds = {key: np.percentile(np.abs(diff[key].cpu().detach().numpy()), (1 - alpha) * 100) for key in diff.keys()}
    num_zeros, num_negatives, num_positives = 0, 0, 0
    for ky in diff.keys():
        grad_array = diff[ky].cpu().detach().numpy()
        direction_array = np.sign(np.where(np.abs(grad_array) > thresholds[ky], grad_array, 0))
        directions[ky] = torch.from_numpy(direction_array).to(device)

        num_zeros += np.sum(direction_array == 0)
        num_negatives += np.sum(direction_array == -1)
        num_positives += np.sum(direction_array == 1)

    if print_stats:
        total = num_zeros + num_negatives + num_positives
        print(f"* Stat of Directions: {num_negatives / total * 100:.2f}% -1, {num_zeros / total * 100:.2f}% 0, {num_positives / total * 100:.2f}% 1")

    return directions   

def compute_distance(w0, w1) -> float:
    """ 
    computes the ditance between two updates. 
    Args:
        w0 (dict): first update.
        w1 (dict): second update.
    Returns
        float: ||w0 - w1||²
    """
    dist = 0.0
    for key in w1.keys():
        if "weight" in key or "bias" in key:
            dist += np.linalg.norm(w0[key].cpu().detach().numpy() - w1[key].cpu().detach().numpy()) ** 2
    return dist

def compute_l2_norm(update: dict)-> float:
	"""
	Compute the L2 norm of an update.
	Args:
        update (dict): A dictionary representing an update (e.g., model state_dict).
    Returns:
        float: The L2 norm of the update.
    """
	norm_sq = 0.0
	for key in update.keys():
		norm_sq += torch.sum(update[key]**2).item()
	l2_norm = math.sqrt(norm_sq)
	return l2_norm

def compute_weighted_average(updates: list[dict], weights=None) -> dict:
    """
    computes the average update from a list of updates.
    Args:
        updates (list[dict]): the list of updates to average.
        weights (list[float]): the weight of each update during averaging.
    Returns:
        dict: the weighted average of updates.
    """
    if weights is None:
        weights = [1] * len(updates)
    sum_weights = np.sum(weights)
    avg = cp.deepcopy(updates[0])
    for key in avg.keys():
        avg[key] = sum(param[key] * (weight / sum_weights) for param, weight in zip(updates, weights))
    return avg

class NoiseDataset(torch.utils.data.Dataset):
    """Dataset that generates random noise inputs."""

    def __init__(self, size: Tuple[int, int, int], num_samples: int):
        self.size = size
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Return just the noise tensor without a label
        noise = torch.rand(self.size)
        return noise
