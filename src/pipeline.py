from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# For annotation
from typing import Tuple

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Load helper functions
from generative_model_base import (
    is_generative_model,
    is_conditional_generative_model,
    GenerativeModel,
)
from utils import fgsm

# To check if generator has correct forward process
import inspect


class ImplementationError(Exception):
    """
    Exception raised for errors in implementation.

    :params message: explanation of the error
    """

    def __init__(self, message=""):
        super().__init__(message)


def check_implementation(generator: nn.Module, target: torch.Tensor) -> None:
    """
    Initializes the generator and classificator.

    :param classifier: The classifier.
    :param generator: The generator.
    :param target: the class that the generator generates.

    :return: the initialized models
    """

    # Check if the class is correct implemented
    if not is_generative_model(generator):
        raise ImplementationError(
            f"Generator has to implement {GenerativeModel.__name__} of {GenerativeModel.__module__}"
        )

    if is_conditional_generative_model(generator):
        generator.set_label(target)

    # Check if the forward process of generator is only one_value
    n_non_optional_args = len(
        [
            param
            for param in inspect.signature(generator.forward).parameters.values()
            if param.default == inspect._empty
        ]
    )
    if n_non_optional_args != 1:
        raise ImplementationError(
            f"The forward method of generator has to be executed with just 1 argument for pipeline. {n_non_optional_args} non optional arguments found"
        )


def pipeline(
    classifier: nn.Module,
    generator: nn.Module,
    device: torch.device,
    target: torch.Tensor,
    n_generated_samples: int,
    epsilon: float,
    timeout_tries: int,
) -> Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Executes the pipeline and generates the samples

    :param classifier: the classifier that classifies generated sample x
    :param generator: the generator, that generates x out of z
    :param device: The device to put the data and networks on
    :param target: target label of data that will be generated
    :param n_generated_samples: amount of samples the pipeline will generate.
    :param epsilon: for FGSM (will be outsources)
    :param timeout tries: amount of tries before interrupted.

    :return: 4 tensors with: the original latent variable z, the original prediction y, the perturbated latent variable pert_z and the perturbated prediction pert_y
    """

    ######################
    # Initialize Variables
    ######################

    # size of latent variable
    nz = generator.get_noise().shape[1]

    # result - original values
    orig_z = torch.empty(size=(n_generated_samples, nz)).to(device)
    orig_y = torch.empty(size=(n_generated_samples, 1)).to(device)
    # result - perturbated values
    pert_z = torch.empty(size=(n_generated_samples, nz)).to(device)
    pert_y = torch.empty(size=(n_generated_samples, 1)).to(device)

    # Number of tries to generated examples
    n_tries = 0
    # Numbers of samples successfully created
    z_idx = 0

    ##########
    # Pipeline
    ##########

    while (
        timeout_tries > n_tries
        and z_idx < n_generated_samples
        and n_generated_samples > 0
    ):
        # get noise and require gradient for gradient method
        z = generator.get_noise().to(device)
        z.requires_grad = True

        # Generate an unperturbated sample
        x = generator(z)
        y_hat = classifier(x)
        y_label = y_hat.max(1, keepdim=True)[1]

        # If the initial prediction differs from target, don't trie to perturbated.
        if y_label.item() != target.item():
            n_tries += 1
            continue

        # Calculate the loss
        loss = F.nll_loss(y_hat, target)

        # Zero all existing gradients
        generator.zero_grad()
        classifier.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Call gradient method
        z_prime = fgsm(z, epsilon, -1, 1)

        # Re-classify the perturbed image
        x_prime = generator(z_prime)
        y_hat_prime = classifier(x_prime)
        y_label_prime = y_hat_prime.max(1, keepdim=True)[1]

        # Check for success (attack)
        if y_label_prime.item() != y_label.item():
            z.requires_grad = False
            orig_z[z_idx] = z
            orig_y[z_idx] = y_label
            pert_z[z_idx] = z_prime
            pert_y[z_idx] = y_label_prime

            z_idx += 1

        n_tries += 1

    if n_tries >= timeout_tries and z_idx != n_generated_samples:
        print(f"Timeout ({timeout_tries}) reached. {z_idx} images returned.")

    return (
        orig_z[:z_idx].detach(),
        orig_y[:z_idx].detach(),
        pert_z[:z_idx].detach(),
        pert_y[:z_idx].detach(),
    )


def generate_samples(
    classifier: nn.Module,
    generator: nn.Module,
    device: torch.device,
    target: torch.Tensor,
    n_generated_samples: int,
    epsilon: float,
    timeout_tries: int,
) -> Tuple[
    nn.Module, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Generates the samples with the pipeline

    :param device: the device to load the data/models
    :param target: The target label that will be generated.
    :param classifier_pth: The path to the classifier pth file.
    :param generator_pth: The path to the generator pth file.
    :param n_generated_samples: The number of samples that will be tried to generate.
    :param timeout: The maximum amount of tries to generate.
    :param gradient_type: The type of gradient method that will be used. -> see gradient_selector
    :param gradient_arguments: The arguments for gradient method.

    :return: classifier, generator and 4 tensors with: the original latent variable z, the original prediction y, the perturabed latent variable pert_z and the perturabed prediction perty
    """

    #######################
    # Preprocess Parameters
    #######################

    # Set target to long tensor if not available.
    if not isinstance(target, torch.LongTensor):
        target = torch.LongTensor([target]).to(device)

    # Check timeout parameter
    assert timeout_tries > 0, "timeout has to be an non-negative integer."
    assert (
        timeout_tries >= n_generated_samples
    ), f"timeout is lower than n_generated_samples. Can not generate #{n_generated_samples} samples."

    #####################################
    # Initialize and execute the pipeline
    #####################################

    check_implementation(generator, target)

    return (
        classifier,
        generator,
        pipeline(
            classifier,
            generator,
            device,
            target,
            n_generated_samples,
            epsilon,
            timeout_tries,
        ),
    )


def generate_samples_with_iterative_epsilons(
    device: str,
    target: int,
    classifier_pth: str,
    generator_pth: str,
    n_generated_samples: int,
    timeout: int,
    gradient_type: str,
    gradient_arguments: dict,
    epsilons=None,
    return_epsilons=False,
) -> Tuple[
    nn.Module,
    nn.Module,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Generates samples with pipline configured by DictConfig. It uses an interative ascend on epsilon weighting factor for fgsm. ATTENTION: To get the used epsilons, give parameter: return_epsilons = True. Then it returns 5 Tensors (I know bad practice)

    :param device: the device to load the data/models
    :param target: The target label that will be generated.
    :param classifier_pth: The path to the classifier pth file.
    :param generator_pth: The path to the generator pth file.
    :param n_generated_samples: The number of samples that will be tried to generate.
    :param timeout: The maximum amount of tries to generate.
    :param gradient_type: The type of gradient method that will be used. -> see gradient_selector
    :param gradient_arguments: The arguemnts for gradient method.
    :param epsilons: If value None, the already implemented list will be used. make sure that the epsilon is iterateable.
    :param return_epsilons: If value true, returns the 5th tensor with used epsilons.

    :return: classifer, generator and 5 (or 4) tensors with: the original latent variable z, the original prediction y, the perturabed latent variable pert_z, the perturabed prediction perty and the list with epsilons used (only if return_epsilons = True).
    """

    # Check for arguments. This argument need to be set as it will be overwritten.
    assert (
        "epsilon" in gradient_arguments
    ), "You are not using epsilon as gradient argument. Epsilon can not be passed to pipeline."

    # Make sure epsilons is set right.
    if epsilons is None:
        epsilons = np.arange(0.1, 0.85, 0.05)
    elif not isinstance(epsilons, list):
        epsilons = [epsilons]

    # variable declaraiton
    n_to_generate = n_generated_samples
    classifier = None
    generator = None
    z = []
    y = []
    per_z = []
    per_y = []
    used_epsilons = torch.zeros(n_generated_samples)
    z_idx = 0

    # exectuing epsilons
    for epsilon in epsilons:
        print(f"Try to generate {n_to_generate} samples with epsilon = {epsilon:.6}")

        # Set the value and execute pipeline
        n_generated_samples = n_to_generate
        gradient_arguments["epsilon"] = float(
            epsilon
        )  # Value conversion to float is needed, as the dictionary only supports python primitive types

        classifier, generator, pipeline_result = generate_samples(
            device,
            target,
            classifier_pth,
            generator_pth,
            n_generated_samples,
            timeout,
            gradient_type,
            gradient_arguments,
        )

        # Append results
        n_generation_result = len(pipeline_result[0])

        if n_generation_result > 0:
            z.append(pipeline_result[0])
            y.append(pipeline_result[1])
            per_z.append(pipeline_result[2])
            per_y.append(pipeline_result[3])
            used_epsilons[z_idx : z_idx + n_generation_result] = torch.full(
                (1, n_generation_result), epsilon
            )

        # Check how many values are generated
        n_to_generate -= n_generation_result
        z_idx += n_generation_result
        print(f"Generated {n_generation_result} samples")

        # Enough generated
        if n_to_generate <= 0:
            break

    # Only cat if elements are provided, otherwise empty tensor
    if len(z) > 0:
        z, y, per_z, per_y = (
            torch.cat(z),
            torch.cat(y),
            torch.cat(per_z),
            torch.cat(per_y),
        )
    else:
        z, y, per_z, per_y = (
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
            torch.Tensor(),
        )

    # Return epsilons if the user wants
    result = [z, y, per_z, per_y]
    if return_epsilons:
        result.append(used_epsilons)

    return (classifier, generator, tuple(result))
