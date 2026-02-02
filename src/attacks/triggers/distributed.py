import torch
import torch.nn as nn
from .base import BaseTrigger

class DBATrigger(BaseTrigger):
    """
    Corrected implementation of a distributed trigger shard for the DBA attack.
    This version correctly handles the 'color' argument and does not pass it
    to the BaseTrigger parent class.
    """
    def __init__(
        self,
        client_id: int,
        shard_locations: list,
        global_position: tuple,
        patch_size: tuple, #  (16,16), 
        color= (1.0,), #(1.5, 1.5, 1.5),  # brighter (if using RGB channels) 
        alpha=1.0
    ):
        num_shards = len(shard_locations)
        if num_shards == 0:
            raise ValueError("The shard_locations list cannot be empty.")

        # Assign a shard to this client in a round-robin fashion
        shard_index = client_id % num_shards
        x_offset, y_offset = shard_locations[shard_index]

        # Calculate the absolute on-image position for this client's patch
        local_position = (global_position[0] + x_offset, global_position[1] + y_offset)
        
        # Create the pattern from the color
        pattern = torch.tensor(color).view(-1, 1, 1)

        # Call the parent __init__ with only the arguments it expects
        super().__init__(position=local_position, size=patch_size, pattern=pattern, alpha=alpha)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Applies this client's specific trigger shard to a single image."""
        if image.dim() != 3:
            raise ValueError(f"The `apply` method expects a single 3D image tensor, but got shape {image.shape}")

        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        pattern = self.pattern.to(image.device)

        y_start = min(y_pos, img_h)
        y_end = min(y_pos + patch_h, img_h)
        x_start = min(x_pos, img_w)
        x_end = min(x_pos + patch_w, img_w)

        region = poisoned_image[:, y_start:y_end, x_start:x_end]
        blend_pattern = self.alpha * pattern + (1.0 - self.alpha) * region
        poisoned_image[:, y_start:y_end, x_start:x_end] = blend_pattern

        return torch.clamp(poisoned_image, 0, 1)
