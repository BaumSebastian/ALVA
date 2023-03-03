from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class GenerativeModel(ABC):
    """
    This class defines every method generative models need to implement for pipeline data generation.
    """

    @abstractmethod
    def get_noise(self, n: int = 1) -> torch.Tensor:
        """
        Get the noise that is required for input of generative model.

        :param n: the number of noise tensors.
        :return: torch.Tensor with noise for input of generative model.
        """
        pass


class ConditionalGenerativeModel(GenerativeModel):
    """
    This class defines every method conditional generative models need to implement for pipeline data generation.
    """

    @abstractmethod
    def set_label(self, label) -> None:
        """
        Set the label that is required for input of conditional generative models.

        :param label: the label of the conditional generative model, that will be generated.
        """
        pass


def is_generative_model(instance):
    """
    Indicates if the instance is an instance of a class, that DIRECTLY inherits the abstract.GenerativeModel class.

    :param instance: instance of an class that will be checked.

    :return: true if it is an instance, false otherwise.
    """
    return issubclass(type(instance), GenerativeModel)


def is_conditional_generative_model(instance):
    """
    Indicates if the instance is an instance of a class, that DIRECTLY inherits the abstract.GenerativeModel class.

    :param instance: instance of an class that will be checked.

    :return: true if it is an instance, false otherwise.
    """
    return issubclass(type(instance), ConditionalGenerativeModel)
