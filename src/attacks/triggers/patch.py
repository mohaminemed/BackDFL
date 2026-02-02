from .base import BaseTrigger
import torch


class PatchTrigger(BaseTrigger):
    def __init__(self, position=(28, 28), size=(3, 3), color=(1.0, 0.0, 0.0)):
        """
        Initializes the patch trigger.

        Args:
            position (tuple): Top-left (x, y) coordinates.
            size (tuple): (width, height) of the patch.
            color (tuple): The (R, G, B) color, assumed to be in [0.0, 1.0] range.
        """
        # Reshape color to (C, 1, 1) for broadcasting across the patch spatial dimensions
        self.pattern = torch.tensor(color).view(-1, 1, 1)
        super().__init__(position, size, self.pattern, alpha=1.0)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the patch to a PyTorch tensor.

        Args:
            image (torch.Tensor): A clean image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: The image with the trigger embedded.
        """
        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        # Move the pattern to the same device as the image tensor
        pattern = self.pattern.to(image.device)

        # Define and clip boundaries
        y_start = torch.clamp(torch.tensor(y_pos), 0, img_h)
        y_end = torch.clamp(torch.tensor(y_pos + patch_h), 0, img_h)
        x_start = torch.clamp(torch.tensor(x_pos), 0, img_w)
        x_end = torch.clamp(torch.tensor(x_pos + patch_w), 0, img_w)

        # Select the region of interest (ROI)
        region = poisoned_image[:, y_start:y_end, x_start:x_end]

        # Blend the patch. The pattern (C, 1, 1) will broadcast over the region (C, H, W).
        poisoned_image[:, y_start:y_end, x_start:x_end] = \
            self.alpha * pattern + (1.0 - self.alpha) * region

        return poisoned_image