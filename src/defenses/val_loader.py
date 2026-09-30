import random
from collections import defaultdict
from typing import Optional

from torch.utils.data import DataLoader, Subset


def build_val_loader(trainloader: Optional[DataLoader], samples_per_class: int = 1,
                            num_classes: Optional[int] = None, seed: Optional[int] = None
                            ) -> Optional[DataLoader]:
    """
    Build a small, class-balanced validation loader from a client's own
    training data.

    The whole selected set is returned as a single batch (so
    `next(iter(loader))` in ArgusServer._argus_val_batch gets every
    selected sample at once, matching the paper's usage), not split
    across multiple mini-batches.

    Args:
        trainloader: the client's own DataLoader (its .dataset is scanned).
        samples_per_class: cap per class (paper default: 1).
        num_classes: if known, scanning stops early once every class in
            [0, num_classes) has reached the cap, avoiding a full pass
            over a large local dataset. If None, the full dataset is
            scanned once (fine for typical per-client FL/DFL shard sizes).
        seed: optional seed for the scan order, so the same client
            doesn't always pick the same physical samples every call.

    Returns:
        A DataLoader over the selected subset (single batch), or None if
        `trainloader` has no usable dataset.
    """
    if trainloader is None or getattr(trainloader, 'dataset', None) is None:
        return None

    dataset = trainloader.dataset
    n = len(dataset)
    if n == 0:
        return None

    order = list(range(n))
    if seed is not None:
        random.Random(seed).shuffle(order)

    class_to_indices = defaultdict(list)
    seen_classes = set()
    for idx in order:
        _, label = dataset[idx]
        label = int(label)
        if len(class_to_indices[label]) < samples_per_class:
            class_to_indices[label].append(idx)
            seen_classes.add(label)
        if num_classes is not None and len(seen_classes) >= num_classes and \
                all(len(v) >= samples_per_class for v in class_to_indices.values()):
            break

    selected_indices = [idx for indices in class_to_indices.values() for idx in indices]
    if not selected_indices:
        return None

    subset = Subset(dataset, selected_indices)
    # Single batch containing every selected sample -- see docstring.
    return DataLoader(subset, batch_size=len(selected_indices), shuffle=False, num_workers=0)