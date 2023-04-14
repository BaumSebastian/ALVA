from __future__ import annotations

# Annotations
__author__ = "Sebastian Baum"
__maintainer__ = "Sebastian Baum"
__version__ = "1.0.0"
__status__ = "Prototype"

# File for methods that are used everywhere
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# import cv2 as cv
import random

import os.path as path


# def D(t1: torch.Tensor, t2: torch.Tensor) -> float:
#    """
#    Calculates the distance between two vectors.
#
#    :param t1: The first tensor.
#    :param t2: The second tensor.
#
#    :return: The distance between the tensors.
#    """
#    return (t1 - t2).norm()
#
#
# def norm_tensor(t: torch.Tensor, zlb: bool = True) -> torch.Tensor:
#    """
#    Norms tensor, so maximum value is 1.
#
#    :param t: The tensor to rescale
#    :param zlb: zero lower bound means that the lowest values will be set to 0.
#
#    :return: the normed tensor.
#    """
#    if zlb:
#        t -= t.min()
#
#    return t / t.abs().max()
#
#
# def binearize_tensor(
#    t: torch.Tensor, threshold: float = 0.5, min: float = 0.0, max: float = 1.0
# ) -> torch.Tensor:
#    """
#    binearizes tensor to min and max values based on threshold
#
#    :param t: the tensor to binearize
#    :param threshold: threshold of bineraization. Default 0.5
#    :param min: the value to insert if below threshold. Default 0.0
#    :param max: the value to insert if above threshold. Default 1.0
#
#    :return binearized tensor.
#    """
#    return torch.where(t < threshold, torch.Tensor([max]), torch.Tensor([min])).float()
#
#
# def denoise_img(
#    img: torch.Tensor, shape: tuple(int, int, int) = (1, 28, 28)
# ) -> np.array:
#    """
#    Preprocess the image provided as torch.Tensor.
#
#    :param img: The tensor to process.
#
#    :return: The preprocessed tensor as np.array.
#    """
#    img = transforms.functional.to_pil_image(img.cpu().detach().reshape(*shape))
#    return cv.fastNlMeansDenoising(np.array(img), None, 20, 7, 21)
#
#
# def moving_average(data, window_size, fill=True):
#    """
#    Calculates the moving average given data.
#
#    :param data: The data to calculate the moving average.
#    :param window_size: the size of the window.
#    :param fill: If true, extends the result so it matches the length of data with the last calculated value.
#
#    :return: list with moving average values.
#    """
#    # Initialize an empty list to store moving averages
#    ma = []
#
#    # go threw data until window reaches end
#    for idx in range(len(data) - window_size + 1):
#        # Store elements from i to i+window_size
#        # in list to get the current window
#        window = data[idx : idx + window_size]
#
#        # Calculate the average of current window
#        wa = round(sum(window) / window_size, 2)
#
#        # Store the average of current
#        # window in moving average list
#        ma.append(wa)
#
#    if fill:
#        ma.extend([ma[-1] for _ in range(len(data) - len(ma))])
#
#    return ma


def set_random_seed(random_seed: int = 0):
    """
    Sets the random seed on all different python modules.

    :param random_seed: The seed to initialize the modules.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


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


def plot_prediction_switch(per_imgs, imgs, labels, per_labels):
    n = 25
    col = 5
    labels = labels[:n]
    per_labels = per_labels[:n]
    per_imgs = torchvision.utils.make_grid(per_imgs[:n].detach().cpu(), col)
    imgs = torchvision.utils.make_grid(imgs[:n].detach().cpu(), col)

    if not isinstance(per_imgs, list):
        per_imgs = [per_imgs]

    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, (ax1, ax2, ax3) = plt.subplots(1, ncols=3 * len(imgs), figsize=(14, 4))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        ax1.imshow(np.asarray(img))
        ax1.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax1.set_title("Original Images")

    for i, img in enumerate(per_imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        ax2.imshow(img)
        ax2.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    ax2.set_title("Perturbated Images")

    col = 0
    row = 0
    for idx, (l, per_l) in enumerate(zip(labels, per_labels)):
        l, per_l = int(l.item()), int(per_l.item())
        ax3.text(0.02 + 0.2 * row, 0.88 - 0.2 * col, f"{l} → {per_l}", fontsize=13)
        row += 1
        if row % 5 == 0 and row > 0:
            col += 1
            row = 0

    for x, y in zip(ax3.xaxis.get_major_ticks(), ax3.yaxis.get_major_ticks()):
        x.tick1line.set_visible(False)
        x.tick2line.set_visible(False)
        x.label1.set_visible(False)
        x.label2.set_visible(False)
        y.tick1line.set_visible(False)
        y.tick2line.set_visible(False)
        y.label1.set_visible(False)
        y.label2.set_visible(False)

    # ax3.axis('off')
    ax3.grid(True)
    ax3.set_title("Prediction Switch")
