import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss():
    """
    Implementation of Focal Loss for classification tasks.

    Args:
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted. Default is 2.0.
        alpha (float, optional): Balancing factor for class imbalance. Can be a scalar or a tensor with class weights. Default is None.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'. Default is 'mean'.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): Predictions from the model (logits or probabilities).
            targets (torch.Tensor): Ground truth labels (integers representing class indices).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Convert logits to probabilities if necessary
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            probs = F.softmax(predictions, dim=-1)
        else:
            probs = predictions

        # Gather the probabilities of the true class
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(-1)).float()
        probs_true_class = (probs * targets_one_hot).sum(dim=-1)

        # Compute the focal loss modulating factor
        focal_weight = (1 - probs_true_class) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_weight = torch.full_like(targets, fill_value=self.alpha, dtype=torch.float32)
            else:
                alpha_weight = self.alpha[targets]

            focal_weight = focal_weight * alpha_weight

        # Compute cross-entropy loss
        log_probs = torch.log(probs_true_class.clamp(min=1e-12))
        loss = -focal_weight * log_probs

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss