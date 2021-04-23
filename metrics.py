import torch


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor, EPSILON = 1e-15) -> torch.Tensor:
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets * output).sum()

    union = targets.sum() + output.sum() - intersection

    result = (intersection + EPSILON) / (union + EPSILON)

    return result
