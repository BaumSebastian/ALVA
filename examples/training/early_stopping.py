from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"
__credits__ = [
    "https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch/71999355#71999355"
]


class EarlyStopping:
    """
    Class for early stopping.
    """

    def __init__(
        self: EarlyStopping, tolerance: int = 5, min_delta: int = 0
    ) -> EarlyStopping:
        """
        Initializes instance.

        :param tolerance: The amount of times the delta can be hit.
        :param min_delta: The delta that is allowed  between loss.

        :return: Initialized class.
        """
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.__counter = 0
        self.__early_stop = False

    def __call__(self: EarlyStopping, train_loss, valid_loss) -> None:
        """
        Determines if early stop.

        :param train_loss: The loss on training.
        :param valid_loss: The loss on validation.

        :return: None
        """
        if (valid_loss - train_loss) > self.min_delta:
            self.__counter += 1
            if self.__counter >= self.tolerance:
                self.__early_stop = True

    def early_stop(self) -> bool:
        """
        Indicates if the training should stop early.

        :return: True if stop early; false otherwise.
        """
        return self.__early_stop
