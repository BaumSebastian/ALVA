from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# Imports
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

        :param n: The number of noise tensors.
        :return: Tensor with noise for input of generative model.
        """
        pass


class ConditionalGenerativeModel(GenerativeModel):
    """
    This class defines every method conditional generative models need to implement for pipeline data generation.
    """

    @abstractmethod
    def set_label(self, label: torch.LongTensor) -> None:
        """
        Set the label that is required for input of conditional generative models.

        :param label: The label of the conditional generative model, that will be generated.
        """
        pass


def is_generative_model(instance: torch.nn.Module) -> bool:
    """
    Indicates if the instance is an instance of a class, that DIRECTLY inherits the abstract.GenerativeModel class.

    :param instance: Instance of an class that will be checked.

    :return: True if it is an instance, false otherwise.
    """
    return issubclass(type(instance), GenerativeModel)


def is_conditional_generative_model(instance: torch.nn.Module) -> bool:
    """
    Indicates if the instance is an instance of a class, that DIRECTLY inherits the abstract.GenerativeModel class.

    :param instance: Instance of an class that will be checked.

    :return: True if it is an instance, false otherwise.
    """
    return issubclass(type(instance), ConditionalGenerativeModel)
