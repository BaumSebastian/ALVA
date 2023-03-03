from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__= "Prototype"
__credits__ = [
    "Richmond Alake - https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342"
]

# Imports
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    Lenet5 implementation from Yann LeCun's Lenet.
    The original structure from Alake Richmond is adapted to the generator used in this example code, that outputs shape is 28x28 pixel. The first convolution has a padding of 2 and therefore does not need input of 32x32.
    This implementation is an adaption of https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342.
    """

    def __init__(self) -> LeNet5:
        super(LeNet5, self).__init__()
        """
        Initializes a class instance of a classifier for MNIST dataset.
        Own Adaption: Changing the padding to same, so the 28*28 input structure is preserved and the GAN structure has not to be adapted to 32*32 pixel.
        """
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.
        
        :param x: The input data for computation.

        :return: The result of computation applied to input data.
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
