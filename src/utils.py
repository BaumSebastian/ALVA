# File for methods that are used everywhere
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random

import os.path as path


def D(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """
    Calculates the distance between two vectors.

    :param t1: The first tensor.
    :param t2: The second tensor.

    :return: The distance between the tensors.
    """
    return (t1 - t2).norm()

def norm_tensor(t:torch.Tensor, zlb:bool=True) -> torch.Tensor:
    """
    Norms tensor, so maximum value is 1.
    
    :param t: The tensor to rescale
    :param zlb: zero lower bound means that the lowest values will be set to 0.
    
    :return: the normed tensor.
    """  
    if zlb:
        t -= t.min()
    
    return t / t.abs().max()

def binearize_tensor(t : torch.Tensor, threshold:float=.5, min:float=0.0, max:float=1.0) -> torch.Tensor:
    """
    binearizes tensor to min and max values based on threshold
    
    :param t: the tensor to binearize
    :param threshold: threshold of bineraization. Default 0.5
    :param min: the value to insert if below threshold. Default 0.0
    :param max: the value to insert if above threshold. Default 1.0
    
    :return binearized tensor.
    """
    return torch.where(t < threshold, torch.Tensor([max]), torch.Tensor([min])).float()


def denoise_img(
    img: torch.Tensor, shape: tuple(int, int, int) = (1, 28, 28)
) -> np.array:
    """
    Preprocess the image provided as torch.Tensor.

    :param img: The tensor to process.

    :return: The preprocessed tensor as np.array.
    """
    img = transforms.functional.to_pil_image(img.cpu().detach().reshape(*shape))
    return cv.fastNlMeansDenoising(np.array(img), None, 20, 7, 21)


def moving_average(data, window_size, fill=True):
    """
    Calculates the moving average given data.

    :param data: The data to calculate the moving average.
    :param window_size: the size of the window.
    :param fill: If true, extends the result so it matches the length of data with the last calculated value.

    :return: list with moving average values.
    """
    # Initialize an empty list to store moving averages
    ma = []

    # go threw data until window reaches end
    for idx in range(len(data) - window_size + 1):

        # Store elements from i to i+window_size
        # in list to get the current window
        window = data[idx : idx + window_size]

        # Calculate the average of current window
        wa = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        ma.append(wa)

    if fill:
        ma.extend([ma[-1] for _ in range(len(data) - len(ma))])

    return ma

def set_random_seed(random_seed: int = 0):
    """
    Sets the random seed on all different python modules.
    
    :param random_seed: The seed to initialize the modules.
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)