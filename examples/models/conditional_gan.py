from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"
__credits__ = ["Artur Lacerda  - https://github.com/arturml/mnist-cgan"]

# Insert the source folder
import sys

sys.path.append(r"src")

# Imports
import torch
import torch.nn as nn
import math

# Abstract base class from src folder
from generative_model_base import ConditionalGenerativeModel


class CGanGenerator(nn.Module, ConditionalGenerativeModel):
    def __init__(self, n_z: int, n_output: list[int]) -> CGanGenerator:
        """
        initializing a class instance of generator (inheriting nn.Module).

        :param n_z: dimension of random noise input.
        :param n_output: dimension of generated output.

        :return: Initialized generator.
        """
        # Check arguments
        assert (
            isinstance(n_z, int) and n_z > 0
        ), "dimension of latent space must be a positive integer"
        assert (
            isinstance(n_output, tuple) and len(n_output) > 1
        ), "output dimension must be the dimension of generated output and therefore dimension >1. use (1, x) instead."

        # Assign variables
        super(CGanGenerator, self).__init__()
        self.n_z = n_z
        self.n_output = n_output
        self.n_embedding = 10

        # Variable for label
        self.labels = None
        self.label_emb = nn.Embedding(self.n_embedding, self.n_embedding)

        self.model = nn.Sequential(
            nn.Linear(self.n_z + self.n_embedding, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, math.prod(self.n_output)),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.long = None) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        :param z: The input noise for computation.
        :param label: The label of the calculated data g(z). If you use a conditional GAN for convergence training: Make sure to set labels=None and implement the set_label method from the abstract base class.

        :return: The result of computation applied to input data.
        """
        z = z.view(-1, self.n_z)

        # Get the labels
        if labels == None:
            labels = self.labels.expand(z.shape[0])

        assert (
            labels != None
        ), "The labels can't be null, set them first with set_label()"

        # Create embedding information of label
        c = self.label_emb(labels)
        # Concatenate image with label
        x = torch.cat([z, c], 1)

        return self.model(x).view(-1, *self.n_output)

    def get_noise(self, n: int = 1) -> torch.Tensor:
        return torch.randn(n, self.n_z)

    def set_label(self, labels: torch.long) -> None:
        self.labels = labels


class CGanDiscriminator(nn.Module):
    def __init__(self, n_input: list[int]) -> CGanDiscriminator:
        """
        Initializing a class instance of Discriminator (inheriting nn.Module).

        :param n_input: Dimension of input.

        :return: Initialized discriminator.
        """
        # Check arguments
        assert (
            isinstance(n_input, tuple) and len(n_input) > 1
        ), "Input dimension must be  >1. use (1, x) instead."

        # Assign variables
        super(CGanDiscriminator, self).__init__()
        self.n_input = n_input
        self.n_embedding = 10
        self.label_emb = nn.Embedding(self.n_embedding, self.n_embedding)

        self.model = nn.Sequential(
            nn.Linear(math.prod(self.n_input) + self.n_embedding, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.long) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        :param x: The input data for computation.
        :param label: The label of the input data.

        :return: The result of computation applied to input data.
        """
        x = x.view(x.size(0), math.prod(self.n_input))
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        return self.model(x).squeeze()
