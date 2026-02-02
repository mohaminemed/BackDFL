import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .base import BaseTrigger
from ...models.unet import UNet 

class IBATrigger(BaseTrigger):
    """
    Implements the generative trigger from "IBA" (Irreversible Backdoor Attacks),
    aligned with the official repository.

    This trigger uses a U-Net model to generate an input-specific perturbation.
    The generator is trained with a combined loss to be both effective and stealthy.
    """
    def __init__(self, unet_model: UNet, trigger_epochs: int = 5, alpha: float = 0.2, lambda_noise: float = 0.01):
        """
        Initializes the IBA generative trigger.

        Args:
            unet_model (UNet): A U-Net model instance to use as the generator.
            alpha (float): Scaling factor for the perturbation's magnitude.
            lambda_noise (float): Weight for the noise regularization term in the loss.
        """
        self.generator = unet_model
        self.generator.eval()
        self.lambda_noise = lambda_noise
        self.trigger_epochs = trigger_epochs
        self.alpha = alpha
        # Position and size are not used for this full-image trigger.
        super().__init__(position=(0, 0), size=(0, 0), pattern=self.generator, alpha=alpha)

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def train_generator(self, classifier_model, dataloader: DataLoader, target_class: int, 
                        learning_rate: float = 1e-3):
        """
        Trains the U-Net generator to fool a given classifier model while keeping
        the perturbation small.
        """
        device = next(classifier_model.parameters()).device
        self.generator.to(device)
        
        classifier_model.eval()
        self._freeze_model(classifier_model)

        self.generator.train()
        
        optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        print(f"--- Training IBA Generator for {self.trigger_epochs} epochs ---")
        for epoch in range(self.trigger_epochs):
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                
                # Generate perturbation
                perturbation = self.generator(inputs)
                poisoned_inputs = torch.clamp(inputs + self.alpha * perturbation, 0.0, 1.0)
                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)
                
                # Get classifier's output
                outputs = classifier_model(poisoned_inputs)
                
                # --- Combined Loss Calculation (Aligned with official repo) ---
                # 1. Adversarial Loss (to make the attack work)
                l_adv = loss_fn(outputs, poisoned_labels)
                
                # 2. Noise Regularization Loss (to keep the trigger stealthy)
                l_noise = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1).mean()

                loss = l_adv + self.lambda_noise * l_noise
                # --- End Combined Loss ---

                loss.backward()
                optimizer.step()
            
        self._unfreeze_model(classifier_model)
        self.generator.eval()
        print("--- IBA Generator training finished ---")

    @torch.no_grad()
    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the generative trigger to a single clean image.
        """
        device = image.device
        self.generator.to(device)
        self.generator.eval()
        
        # Add a batch dimension, generate perturbation, and remove batch dimension
        perturbation = self.generator(image.unsqueeze(0)).squeeze(0)
        
        poisoned_image = torch.clamp(image + self.alpha * perturbation, 0.0, 1.0)
        return poisoned_image