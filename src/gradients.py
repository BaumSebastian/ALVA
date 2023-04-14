from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# Imports
import torch


def fgsm(
    data: torch.Tensor, epsilon: float, lower_norm: float = -1, higher_norm: float = 1
) -> torch.Tensor:
    """
    Performs FGSM on data.

    :param data: The original unperturbated data. The tensor requires gradient
    :param epsilon: epsilon weight value for sign data.
    :param lower_norm: the lower value for normalizing the data.
    :param higher_norm: the lower value for normalizing the data.

    :return: The perturbated noise.
    """
    # get the gradient
    data_grad = data.grad.data
    sign_data_grad = data_grad.sign()

    # perturbate data and normalize it
    per_data = data + epsilon * sign_data_grad
    return torch.clamp(per_data, lower_norm, higher_norm)
