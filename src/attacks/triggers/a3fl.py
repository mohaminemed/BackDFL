import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from .base import BaseTrigger


class A3FLTrigger(BaseTrigger):
    """
    A re-implementation of the A3FL trigger, aligned with Algorithm 1 of the paper
    (online bi-level optimization), rather than with a separately-pretrained
    adversarial model.

    Key change from the previous version: theta' (the adversarial/"unlearned"
    model) is initialized ONCE from the current classifier and then updated with
    a single SGD step per batch, interleaved with the trigger's PGD steps on that
    same batch -- exactly as in Algorithm 1, lines 1-11. Previously, theta' was
    trained for `adv_epochs` full passes over the trigger dataloader BEFORE any
    trigger optimization happened, which let it fully unlearn the (stale, 0.5-
    initialized) pattern and made the adaptation term dominate the loss.

    """

    def __init__(self, position=(2, 2), size=(5, 5), alpha=1.0, in_channels: int = 3, image_size=(28, 28),
                 trigger_epochs: int = 5, trigger_pgd_steps: int = 1, trigger_lr: float = 0.01,
                 lambda_balance: float = 0.1, adv_lr: float = 0.01):
        """
        Initializes the A3FL trigger.

        trigger_epochs: number of passes over the (small) trigger dataloader --
            this is the "K" in Algorithm 1's outer `for j = 1 to K` loop, i.e.
            how many batches of local data drive the online trigger/theta'
            co-optimization. Defaults to 1, matching roughly one local pass.
        trigger_pgd_steps: number of PGD steps taken on the trigger pattern per
            batch before the single theta' SGD step (Algorithm 1's K_trigger,
            lines 4-8). Defaults to 1; raise it if the trigger needs more steps
            to converge per batch, per the paper's "until convergence" PGD.
        trigger_lr: PGD step size alpha_1 for the trigger pattern.
        lambda_balance: lambda_0, scaled by cosine similarity between theta_t
            and theta'_t to get the dynamic lambda used in the trigger loss.
        adv_lr: SGD learning rate alpha_2 for theta' (momentum 0.9, weight
            decay 5e-4, as in the paper).
        """
        # The trigger is a full-size tensor, initialized to 0.5 as in the repo.
        # The patch is applied using a separate mask.
        initial_pattern = torch.ones(in_channels, image_size[0], image_size[1]) * 0.5

        # The mask defines the patch area.
        mask = torch.zeros(in_channels, image_size[0], image_size[1])
        mask[:, position[1]:position[1] + size[1], position[0]:position[0] + size[0]] = 1.0

        self.mask = mask
        # The pattern itself is what gets optimized.
        self.pattern = initial_pattern

        # The BaseTrigger pattern is the combination of the trigger and mask
        # but we will handle application manually.
        super().__init__(position, size, self.pattern, alpha)

        # Store optimization hyperparameters
        self.trigger_epochs = trigger_epochs
        self.trigger_pgd_steps = trigger_pgd_steps
        self.trigger_lr = trigger_lr
        self.lambda_balance = lambda_balance
        self.adv_lr = adv_lr

    def _freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = True


    def _get_model_similarity(self, model_a, model_b):
        with torch.no_grad():
           pa = torch.cat([p.detach().reshape(-1) for p in model_a.parameters()])
           pb = torch.cat([p.detach().reshape(-1) for p in model_b.parameters()]).to(pa.device)
           return F.cosine_similarity(pa, pb, dim=0, eps=1e-8)        

    def _apply_trigger_to_batch(self, images: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        """Applies the trigger pattern to a batch of images using the mask."""
        pattern = pattern.to(images.device)
        mask = self.mask.to(images.device)
        return torch.clamp(images * (1 - mask) + pattern * mask, 0.0, 1.0)

    def train_trigger(self, classifier_model, dataloader: DataLoader, target_class: int):
        """
        Optimizes the trigger patch and the adversarial model theta' jointly and
        online, following Algorithm 1 of the paper:

            theta'_t = theta_t                                   # line 1
            for j = 1 .. K:                                       # line 2
                sample batch B                                    # line 3
                for k = 1 .. K_trigger:                            # line 4
                    L = Lce(x+delta, y~; theta_t)
                        + lambda * Lce(x+delta, y~; theta'_t)      # line 6
                    delta <- delta - alpha_1 * sign(grad_delta L)  # line 7 (PGD)
                theta'_t <- theta'_t - alpha_2 * grad_theta
                            Lce(x+delta, y; theta'_t)              # line 10 (1 SGD step, true labels)
        """
        device = next(classifier_model.parameters()).device
        pattern = self.pattern.clone().detach().to(device).requires_grad_(True)

        print(f"--- Training A3FL Trigger (online Algorithm 1): "
              f"epochs={self.trigger_epochs}, pgd_steps/batch={self.trigger_pgd_steps} ---")

        # theta_t is only used for forward passes here; it is not modified.
        self._freeze_model(classifier_model)
        classifier_model.eval()

        # theta'_t initialized once from the current global/local model (line 1).
        adv_model = copy.deepcopy(classifier_model)
        self._unfreeze_model(adv_model)
        adv_model.train()
        adv_optimizer = optim.SGD(adv_model.parameters(), lr=self.adv_lr, momentum=0.9, weight_decay=5e-4)

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.trigger_epochs):
            for inputs, labels in dataloader:                      # line 3
                inputs, labels = inputs.to(device), labels.to(device)

                # Recompute lambda each batch since theta' drifts batch-to-batch.
                similarity = self._get_model_similarity(classifier_model, adv_model)
                dynamic_lambda = self.lambda_balance * similarity

                poisoned_labels = torch.full((inputs.size(0),), target_class, dtype=torch.long, device=device)

                # --- lines 4-8: K_trigger PGD steps on delta ---
                for _ in range(self.trigger_pgd_steps):
                    poisoned_inputs = self._apply_trigger_to_batch(inputs, pattern)

                    outputs_orig = classifier_model(poisoned_inputs)
                    backdoor_loss = loss_fn(outputs_orig, poisoned_labels)

                    outputs_adv = adv_model(poisoned_inputs)
                    adaptation_loss = loss_fn(outputs_adv, poisoned_labels)

                    total_loss = backdoor_loss + dynamic_lambda * adaptation_loss

                    if pattern.grad is not None:
                        pattern.grad.zero_()
                    total_loss.backward()

                    with torch.no_grad():
                        # PGD step using the sign of the gradient, as in the official repo.
                        pattern.sub_(self.trigger_lr * pattern.grad.sign())
                        pattern.clamp_(0, 1)  # Project back to valid range

                # --- lines 9-10: one SGD step on theta'_t, true labels, detached pattern ---
                poisoned_inputs_for_adv = self._apply_trigger_to_batch(inputs, pattern.detach())
                adv_optimizer.zero_grad()
                adv_outputs = adv_model(poisoned_inputs_for_adv)
                adv_loss = loss_fn(adv_outputs, labels)  # TRUE labels: this is the unlearning step
                adv_loss.backward()
                adv_optimizer.step()

        self._freeze_model(adv_model)
        adv_model.eval()
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