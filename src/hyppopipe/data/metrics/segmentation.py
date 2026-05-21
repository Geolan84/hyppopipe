"""Segmentation loss helpers (BCE + Dice)."""

from torch import Tensor, nn, sigmoid


def dice_loss(inputs, targets, smooth=1e-6):
    """Compute one minus the Dice coefficient between logits and binary targets.

    Args:
        inputs: Model logits (sigmoid applied internally).
        targets: Binary target tensor with the same flattenable shape.
        smooth: Small constant for numerical stability.

    Returns:
        Scalar Dice loss ``1 - dice``.
    """
    inputs = sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE-with-logits and Dice loss for binary segmentation."""

    def __init__(self, weight: Tensor | None = None, smooth: int | float = 1e-6):
        """Configure BCE class weights and Dice smoothing.

        Args:
            weight: Per-class weight for :class:`torch.nn.BCEWithLogitsLoss`.
            smooth: Smoothing term passed to :func:`dice_loss`.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)
        self.smooth = smooth

    def forward(self, inputs, targets):
        """Return ``bce_loss + dice_loss`` for one batch.

        Args:
            inputs: Model logits.
            targets: Binary targets.

        Returns:
            Scalar combined loss.
        """
        bce_loss = self.bce(inputs, targets)
        d_loss = dice_loss(inputs, targets, smooth=self.smooth)
        return bce_loss + d_loss
