from torch import Tensor, nn, sigmoid


def dice_loss(inputs, targets, smooth=1e-6):
    inputs = sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, weight: Tensor | None = None, smooth: int | float = 1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)
        self.smooth = smooth

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        d_loss = dice_loss(inputs, targets, smooth=self.smooth)
        return bce_loss + d_loss
