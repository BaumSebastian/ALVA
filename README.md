
# Adversarial Latent Vector Adjustment (ALVA)

## Table of Contents

- [Adversarial Latent Vector Adjustment (ALVA)](#adversarial-latent-vector-adjustment-alva)
  - [Table of Contents](#table-of-contents)
  - [General](#general)
  - [ALVA Concept](#alva-concept)
  - [How to use ALVA](#how-to-use-alva)
    - [Unconditional Generative Model](#unconditional-generative-model)
    - [Conditional Generative Model](#conditional-generative-model)
  - [Alva Example](#alva-example)
  - [Convergence Training](#convergence-training)


## General

Adversarial Latent Vector Adjustment (ALVA) is based on following paper: LINK. \
It addresses two problems in data augmentation domain:

- How can I generate data without prior knowledge about the data distribution?
- How can I make sure that the created data triggers new learning point for the data driven method?

In this paper this is achieved with a combination of Adversarial Attacks (FGSM) and Generative Models (GAN). For more details look into the paper.

## ALVA Concept

The concept of ALVA relies on the mathematical concept of Adversarial Attacks and requires two differentiable models (Generator and Classificator). \
In the paper of [Goodfellow et. al](https://arxiv.org/abs/1312.6199) they explain the intriguing properties of neural networks and propose a method called Fast Gradient Sign Method (FGSM) to fool a MLP. \
The basic concept is to perturbate a data entry $x$ with the weighted sign of the derivative of the classification. Formal spoken as follows: \
$x' = x + \epsilon \cdot sign (\nabla_x J(f_\theta(x,y)))$ \
The misclassification of $x'$ with $C(x_i) = y_i \And C(x_i') \neq y_i$ is illustrated below: \
<img src="docs/readme_pictures/Goodfellow%20illustration.jpg" alt="Illustration of an FGSM" width = 400 /> \
The original orange point is changed according to the gradient and gets into the red marked are. The difference between the real and learned classification boundary. \
The FGSM is used as regularization method as the perturbated data added to the training data set decreases the classification accuracy. \
Other work LINK shows, that the misclassification is because of the pixel fluctuation added to the data sample x. \
To overcome this property, while still be legitimate data, we expand the idea of the adversarial attack and plug in a generative Model (GAN). \
We no longer perturbate the data sample $x$ itself, but the latent vector $z$. To get a misclassification of $x'=G(z')$, where $C(x') \neq y$. \
This leads to following adaption fo the formula: \
$z' = z + \epsilon \cdot sign(\nabla_z J(f_\theta(z, y)))$ \
An successful attack is described with following illustrations. \
<img src="docs/readme_pictures/test.png" width = 600 /> \
Starting from an arbitrary latent vector $z$, we create a a data sample $(x_i, y_i)$ (It is possible to create $x$ and $y$ with a conditional GAN). Afterwards the latent vector will be attacked with $z'$ and a new data entry will be created with $(x_i',y_i)$.

## How to use ALVA

To use ALVA you need first of all a differentiable classificator $C$ and a differentiable generative model $G$. It is assumed that the output of $G(z)$ can be directly passed to the classificator $C$ with $C(G(z))$. While the implementation of $C$ is arbitrary, the generative model $G$ needs to inherit from base  [generative_model_base.py](src/generative_model_base.py). \

### Unconditional Generative Model

This generative model needs tio implement `GenerativeModel` from [generative_model_base.py](src/generative_model_base.py). To inherit `GenerativeModel` you need to implement static function `get_noise()`. This function is used to get a fitting latent vector $z$.

```python
n = 100              # Number of latent vectors
z = G.get_noise(100) # Returns 100 latent vectors of specific shape defined in G.
x = G(z)             # 100 data points
```

To check if the model is implemented correctly, the method `is_generative_model` in [generative_model_base.py](src/generative_model_base.py) should return `True`.

### Conditional Generative Model

This model needs more information and therefore implement `ConditionalGenerativeModel` from [generative_model_base.py](src/generative_model_base.py).

## Alva Example

## Convergence Training

We didn't mentioned it in the paper, but we also investigated the concept of something we call convergence training. \
We propose convergence training as repeating stept of executing the pipeline of ALVA, add the generated samples to the training dataset of the classificator and repeat the step, until the classificator can't be fooled.
