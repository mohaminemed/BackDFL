import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from .base import BaseTrigger

class A3FLTrigger(BaseTrigger):
    """
    A re-implementation of the A3FL trigger, aligned with the official GitHub repository.
    The trigger is a learnable pattern applied via a mask, optimized using an
    adversarial training scheme.
    """
    def __init__(self, position=(2, 2), size=(5, 5), alpha=1.0, in_channels: int = 3, image_size=(28, 28),
                 trigger_epochs: int = 10, trigger_lr: float = 0.01, lambda_balance: float = 0.1,
                 adv_epochs: int = 100, adv_lr: float = 0.01):
        """
        Initializes the A3FL trigger.
        """
        # The trigger is a full-size tensor, initialized to 0.5 as in the repo.
        # The patch is applied using a separate mask.
        initial_pattern = torch.ones(in_channels, image_size[0], image_size[1]) * 0.5
        
        # The mask defines the patch area.
        mask = torch.zeros(in_channels, image_size[0], image_size[1])
        mask[:, position[1]:position[1]+size[1], position[0]:position[0]+size[0]] = 1.0
        
        self.mask = mask
        # The pattern itself is what gets optimized.
        self.pattern = initial_pattern

        # The BaseTrigger pattern is the combination of the trigger and mask
        # but we will handle application manually.
        super().__init__(position, size, self.pattern, alpha)
        
        # Store optimization hyperparameters
        self.trigger_epochs = trigger_epochs
        self.trigger_lr = trigger_lr
        self.lambda_balance = lambda_balance
        self.adv_epochs = adv_epochs
        self.adv_lr = adv_lr

    def _freeze_model(self, model):
        for param in model.parameters(): param.requires_grad = False

    def _unfreeze_model(self, model):
        for param in model.parameters(): param.requires_grad = True

    def _apply_trigger_to_batch(self, images: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        """Applies the trigger pattern to a batch of images using the mask."""
        pattern = pattern.to(images.device)
        mask = self.mask.to(images.device)
        return torch.clamp(images * (1 - mask) + pattern * mask, 0.0, 1.0)

    def _get_adversarial_model(self, classifier_model, dataloader, pattern):
        """Creates the hardened adversarial model."""
        adv_model = copy.deepcopy(classifier_model)
        self._unfreeze_model(adv_model)
        adv_model.train()
        
        device = next(adv_model.parameters()).device
        optimizer = optim.SGD(adv_model.parameters(), lr=self.adv_lr, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.adv_epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                poisoned_inputs = self._apply_trigger_to_batch(inputs, pattern)
                optimizer.zero_grad()
                outputs = adv_model(poisoned_inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
        self._freeze_model(adv_model)
        adv_model.eval()
        return adv_model

    def _get_model_similarity(self, model_a: nn.Module, model_b: nn.Module) -> torch.Tensor:
        """Computes cosine similarity between two models' parameters."""
        params_a = torch.cat([p.view(-1) for p in model_a.parameters()])
        params_b = torch.cat([p.view(-1) for p in model_b.parameters()])
        return F.cosine_similarity(params_a, params_b.to(params_a.device), dim=0, eps=1e-8)

    def train_trigger(self, classifier_model, dataloader: DataLoader, target_class: int):
        """Optimizes the trigger patch using the A3FL adversarial process."""
        device = next(classifier_model.parameters()).device
        pattern = self.pattern.clone().detach().to(device).requires_grad_(True)
        
        print("--- Training A3FL Trigger (aligned with official repo) ---")
        adversarial_model = self._get_adversarial_model(classifier_model, dataloader, pattern)
        
        self._freeze_model(classifier_model)
        classifier_model.eval()

        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(self.trigger_epochs):
            similarity = self._get_model_similarity(classifier_model, adversarial_model)
            dynamic_lambda = self.lambda_balance * similarity
            
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                poisoned_inputs = self._apply_trigger_to_batch(inputs, pattern)
                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)

                outputs_orig = classifier_model(poisoned_inputs)
                backdoor_loss = loss_fn(outputs_orig, poisoned_labels)
                
                outputs_adv = adversarial_model(poisoned_inputs)
                adaptation_loss = loss_fn(outputs_adv, poisoned_labels)
                
                total_loss_batch = backdoor_loss + dynamic_lambda * adaptation_loss
                
                if pattern.grad is not None:
                    pattern.grad.zero_()

                total_loss_batch.backward()
                
                with torch.no_grad():
                    # PGD step using the sign of the gradient, as in the official repo
                    pattern.sub_(self.trigger_lr * pattern.grad.sign())
                    pattern.clamp_(0, 1) # Project back to valid range

        self._unfreeze_model(classifier_model)
        
        # Update the class's pattern with the optimized result
        self.pattern = pattern.detach().cpu()
        print("--- A3FL Trigger training finished ---")

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Applies the optimized trigger to a single image."""
        if image.dim() != 3:
            raise ValueError(f"The `apply` method expects a single 3D image tensor, but got shape {image.shape}")

        pattern = self.pattern.to(image.device)
        mask = self.mask.to(image.device)
        
        return torch.clamp(image * (1 - mask) + pattern * mask, 0, 1)

