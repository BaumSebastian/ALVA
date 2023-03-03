import torch


def gradient_selector(gradient_type: str):
    """
    A helper function to access the gradient methods to calculate gradient in pipeline.

    :param gradient_type: the name of the gradient method to use.

    :return: The gradient method, that can be executed.
    """

    gradients = {
        'fgsm': fgsm,
        'fgsm_mul' :fgsm_mul,
    }

    assert (
        gradient_type in gradients
    ), f'The selected gradient "{gradient_type}" is not part of {gradient_selector.__name__}. Please choose one out of following: {", ".join(gradients.keys())}'

    return gradients[gradient_type]


def fgsm(
    data: torch.Tensor, epsilon: float, lower_norm: float, higher_norm: float
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
    perturbated_data = data + epsilon * sign_data_grad
    return torch.clamp(perturbated_data, lower_norm, higher_norm)    

def fgsm_mul(
    data: torch.Tensor, epsilon: float, lower_norm: float, higher_norm: float
) -> torch.Tensor:
    """
    Performs slighty differnt FGSM on data.

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
    perturbated_data = data + data * sign_data_grad
    return torch.clamp(perturbated_data, lower_norm, higher_norm)