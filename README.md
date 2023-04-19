
# Adversarial Latent Vector Adjustment (ALVA) <!-- omit from toc -->

## Table of Content <!-- omit from toc -->

- [General](#general)
- [ALVA Concept](#alva-concept)
  - [Related Work](#related-work)
  - [Concept Idea](#concept-idea)
- [How to use ALVA](#how-to-use-alva)
  - [Models](#models)
    - [Unconditional Generative Model](#unconditional-generative-model)
    - [Conditional Generative Model](#conditional-generative-model)
  - [Generate Samples](#generate-samples)
- [Alva Example](#alva-example)
- [Convergence Training](#convergence-training)

___

## General

Adversarial Latent Vector Adjustment (ALVA) is a novel data augmentation method proposed by the following paper: [insert link]. It addresses two challenges in the field of data augmentation:

- How can we generate representative data without any prior knowledge about the underlying data distribution?
- How can we ensure that the newly generated data will provide new, unseen edge cases and are therefore valuable for further training?

In this paper, the authors propose a method that leverages Adversarial Attacks (FGSM) and Generative Models (GAN) to address these challenges. For a comprehensive understanding of the ALVA, please refer to the original paper above.

___

## ALVA Concept

The concept of ALVA is based on the mathematical concept of Adversarial Attacks and requires two differentiable models (Generator and Classificator).

### Related Work

In the paper by [Goodfellow et. al](https://arxiv.org/abs/1412.6572), they explain the idea behind adversarial examples and propose a method called Fast Gradient Sign Method  (FGSM) to generate such examples that "fool" a MLP. The fundamental idea behind this method is to modify a data point $x$ by adding a small perturbation obtained by computing the weighted sign of the derivative of the classification function. This can be mathematically expressed as: \
$x' = x + \epsilon \cdot sign (\nabla_x J(f_\theta(x,y)))$ \
Where $x'$ is the adversarial example, that is misclassified by $C$ with $C(x) = y \And C(x') \neq y_i$. The process is illustrated below. \
<img src="docs/readme_pictures/Goodfellow%20illustration.jpg" alt="Illustration of an FGSM" width = 400 /> \
The original data point, depicted as orange orange, is perturbated based on the gradient  (indicated by the arrow) and falls within the red shaded region. The orange dot is classified as a rectangle despite being an orange data point. The red area represents the disparity between the true and learned classification boundary. To minimize the difference between $x$ and $x'$ the perturbation is weighted with $\epsilon$. Therefore the data sample $x'$ looks similar to $x$. Although the FGSM is utilized as a regularization method, adding the perturbed data into the training dataset can leads to a decrease in classification accuracy. Previous research (insert link) show that the misclassification is a result of the pixel fluctuation/noise added to $x$.

### Concept Idea

The concept of ALVA is to leverage the principles of FGSM as a guide to generate new and unseen data. Instead of perturbing the original data sample $x$, the latent vector $z$ of the generative model is manipulated (in this example, we use the generator of a GAN structure). The new data is the $x'=G(z')$, while a misclassification of $x'=G(z')$, where $C(x') \neq y$ is described as successful attack. This leads to following adaption of the FGSM formula: \
$z' = z + \epsilon \cdot sign(\nabla_z J(f_\theta(z, y)))$ \
Starting from an arbitrary latent vector $z$, we first create a data sample $x$ (Or $(x,y)$ if the generator is conditional) using a generative model. Afterwards we perturbate the latent vector with formula mentioned earlier to obtain $z'$. Using this perturbated latent vector, we generate a new data sample $x'$with the generative model.

___

## How to use ALVA

Firstly, it is important to ensure that the generative model is implemented as specified [below](#models). Once the model is implemented, the ALVA code provided can be used. For experimental purposes, it is recommended to examine the [convergence training chapter](#convergence-training), which has yielded promising results. However, it should be noted that these results may not necessarily generalize to new datasets and therefore require further evaluation.

### Models

To utilize ALVA, you require two differentiable models: a classificator $C$ and a generative model $G$. It is required that the output of $G$ can be passed directly to the classificator $C$ with $C(G(z))$. Although the implementation of $C$ is arbitrary, the generative model $G$ must inherit from the [generative_model_base.py](/src/generative_model_base.py) based on whether it is unconditional or conditional.

#### Unconditional Generative Model

For an unconditional generative model, you need to implement the `GenerativeModel` class from [generative_model_base.py](/src/generative_model_base.py). To inherit this class, you must implement the static function `get_noise()`. This function is used to obtain a latent vector $z$ according to $G$.

```python
n = 100              # Number of latent vectors.
z = G.get_noise(n)   # Returns 100 latent vectors of specific shape defined in G.
x = G(z)             # 100 data points.
```

This method has been implemented in the [example cgan](examples/models/conditional_gan.py).\
To check if the model is implemented correctly, the method `is_generative_model` in [generative_model_base.py](src/generative_model_base.py) should return `True`.

#### Conditional Generative Model

In order to use ALVA for unconditional and conditional generative models, the conditional model needs to implement `get_noise()` from [generative_model_base.py](src/generative_model_base.py) as well as inheriting `ConditionalGenerativeModel` from [generative_model_base.py](src/generative_model_base.py). For implementing `get_noise()` please see the [chapter above](#unconditional-generative-model). To inherit `ConditionalGenerativeModel` the static function `set_label()` has to be implemented. It sets the label for all generation until it is changed via `set_label()` or passed in the `forward` method of the generative model.

```python
label_class = 8 # label class refers to the number in MNIST as in example code. 
label = torch.Long([label_class])
G.set_label(label) # Now G will only generate data of the class 8 in MNIST.

# As above
n = 100
z = G.get_noise(n)
x = G(z) # Generate 100 mnist samples of class 8.
```

### Generate Samples

See [examples](#alva-example) below or see [examples source code](examples/pipeline_example.ipynb).

___

## Alva Example

___

## Convergence Training

We didn't mentioned it in the paper, but we also investigated the concept of something we call convergence training. \
We propose convergence training as repeating steps of executing the pipeline of ALVA, add the generated samples to the training dataset of the classificator and repeat the step, until the classificator can't be fooled.
