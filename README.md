
# Adversarial Latent Vector Adjustment (ALVA) <!-- omit from toc -->

Adversarial Latent Vector Adjustment (ALVA) is a novel data augmentation method. This repository has implemented ALVA for MNIST Dataset. Following pictures have been created, with a LeNet-5 implementation predicted these pictures with the label below each picture. ALVA creates guided new unseen data.

|Generated Image $x$ (By ALVA)|![generated image 0](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_0.png)|![generated image 1](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_1.png)|![generated image 2](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_2.png)|![generated image 3](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_3.png)|![generated image 4](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_4.png)|![generated image 5](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_5.png)|![generated image 6](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_6.png)|![generated image 7](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_7.png)|![generated image 8](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_8.png)|![generated image 9](https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/alva_generation/alva_9.png)|
|-----| :--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |:--: |
| Original  Label $y$ |0|1|2|3|4|5|6|7|8|9|
| Predicted Label* $\hat y$ |**7**|**7**|**7**|**7**|**7**|**7**|**7**|**3**|**7**|**7**|


*) The predicted label of a reference MLP ([LeNet5 Adaption](https://github.com/BaumSebastian/ALVA/blob/main/examples/models/lenet5.py))

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
- [Convergence Training](#convergence-training)

___

## General

Adversarial Latent Vector Adjustment (ALVA) is a novel data augmentation method proposed by the following paper: [link will come]. It addresses two challenges in the field of data augmentation:

- How can we generate representative data without any prior knowledge about the underlying data distribution?
- How can we ensure that the newly generated data will provide new, unseen edge cases and are therefore valuable for further training?

In this paper, the authors propose a method that leverages Adversarial Attacks (FGSM) and Generative Models (GAN) to address these challenges. For a comprehensive understanding of the ALVA, please refer to the original paper above.

___

## ALVA Concept

The concept of ALVA is based on the mathematical concept of Adversarial Attacks and requires two differentiable models (Generator and Classificator).

### Related Work

In the paper by [Goodfellow et. al](https://arxiv.org/abs/1412.6572), they explain the idea behind adversarial examples and propose a method called Fast Gradient Sign Method  (FGSM) to generate such examples that "fool" a MLP. The fundamental idea behind this method is to modify a data point $x$ by adding a small perturbation obtained by computing the weighted sign of the derivative of the classification function. This can be mathematically expressed as: \
$x^\prime = x + \epsilon \cdot sign (\nabla_x J(f_\theta(x,y)))$ \
Where $x^\prime$ is the adversarial example, that is misclassified by $C$ with $C(x) = y\ \And\ C(x^\prime) \neq y$. The process is illustrated below.
<p align="center">
<img src="docs/readme_pictures/FGSM_illustration.jpg" alt="Illustration of FGSM" width = 500/>
<p/>

The original data point, depicted as orange orange, is perturbated based on the gradient  (indicated by the arrow) and falls within the red shaded region. The orange dot is classified as a rectangle despite being an orange data point. The red area represents the disparity between the true and learned classification boundary. To minimize the difference between $x$ and $x^\prime$ the perturbation is weighted with $\epsilon$. Therefore the data sample $x^\prime$ looks similar to $x$. Although the FGSM is utilized as a regularization method, adding the perturbed data into the training dataset can leads to a decrease in classification accuracy. Related work indicates that the misclassification is a result of the pixel fluctuation/noise added to $x^\prime$.

### Concept Idea

The concept of ALVA is to leverage the principles of FGSM as a guide to generate new and unseen data. Instead of perturbing the original data sample $x$, the latent vector $z$ of the generative model is manipulated (in this example, we use the generator of a GAN structure). The new data is the $x^\prime=G(z^\prime)$, while a misclassification of $x^\prime=G(z^\prime)$, where $C(x^\prime) \neq y$ is described as successful attack. This leads to following adaption of the FGSM formula: \
$z^\prime = z + \epsilon \cdot sign(\nabla_z J(f_\theta(z, y)))$ \
Starting from an arbitrary latent vector $z$, we first create a data sample $x$ (Or $(x,y)$ if the generator is conditional) using a generative model. Afterwards we perturbate the latent vector with formula mentioned earlier to obtain $z^\prime$. Using this perturbated latent vector, we generate a new data sample $x^\prime$ with the generative model.

___

## How to use ALVA

Firstly, it is important to ensure that the generative model is implemented as specified [below](#models). Once the model is implemented, the ALVA code provided can be used. For experimental purposes, it is recommended to examine the [convergence training chapter](#convergence-training), which has yielded promising results. However, it should be noted that these results may not necessarily generalize to new datasets and therefore require further evaluation.

### Models

To utilize ALVA, you require two differentiable models: a classificator $C$ and a generative model $G$. It is required that the output of $G$ can be passed directly to the classificator $C$ with $C(G(z))$. Although the implementation of $C$ is arbitrary, the generative model $G$ must inherit from the [generative_model_base.py](https://github.com/BaumSebastian/ALVA/blob/main/src/generative_model_base.py) based on whether it is unconditional or conditional.

#### Unconditional Generative Model

For an unconditional generative model, you need to implement the `GenerativeModel` class from [generative_model_base.py](https://github.com/BaumSebastian/ALVA/blob/main/src/generative_model_base.py). To inherit this class, you must implement the static function `get_noise()`. This function is used to obtain a latent vector $z$ according to $G$.

```python
n = 100              # Number of latent vectors.
z = G.get_noise(n)   # Returns 100 latent vectors of specific shape defined in G.
x = G(z)             # 100 data points.
```

This method has been implemented in the [example cgan](https://github.com/BaumSebastian/ALVA/blob/main/examples/models/conditional_gan.py).\
To check if the model is implemented correctly, the method `is_generative_model` in [generative_model_base.py](https://github.com/BaumSebastian/ALVA/blob/main/src/generative_model_base.py) should return `True`.

#### Conditional Generative Model

In order to use ALVA for unconditional and conditional generative models, the conditional model needs to implement `get_noise()` from [generative_model_base.py](https://github.com/BaumSebastian/ALVA/blob/main/src/generative_model_base.py) as well as inheriting `ConditionalGenerativeModel` from [generative_model_base.py](https://github.com/BaumSebastian/ALVA/blob/main/src/generative_model_base.py). For implementing `get_noise()` please see the [chapter above](#unconditional-generative-model). To inherit `ConditionalGenerativeModel` the static function `set_label()` has to be implemented. It sets the label for all generation until it is changed via `set_label()` or passed in the `forward` method of the generative model.

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

To generate the samples you need to `import alva` (from [alva.py](https://github.com/BaumSebastian/ALVA/blob/main/src/alva.py)). This module implements modules `generate_samples()`.

```python
from alva import generate_samples

"""
Required parameters:
classifier          - The classifier that will be fooled.
generator           - The generator that will create new data
DEVICE              - The device where pytorch will be executed.
TARGET              - The target class that will be generated
N_GENERATED_SAMPLES - The number of samples to generate
N_TIMEOUT           - The number of tries before returning
EPSILON             - Weighting Factor [0,1].

Returns:
the latent vector z with label y and the perturbated z' with label y'.
"""
z, y, per_z, per_y = generate_samples ( 
      classifier, 
      generator, 
      DEVICE, 
      TARGET, 
      N_GENERATED_SAMPLES, 
      N_TIMEOUT, 
      EPSILON
      )
```

One parameter will be explained more in detail: The weighting factor `EPSILON`. It is the weighting factor of FGSM. It is not directly the magnitude of the perturbation, but the magnitude of the perturbation of $z$.

See [examples source code](https://github.com/BaumSebastian/ALVA/blob/main/examples/alva_example.ipynb) for a detailed implementation and example models trained on MNIST dataset.

___

## Convergence Training

In addition to the experiments described in the paper, we also investigated a technique we call "convergence training". This involves the following steps:

1. Train a classifier $C$ on a dataset $D$.
2. Use ALVA to generate $n$ samples from each class in $D$.
3. Add the generated samples to $D$.
4. Repeat steps 1-3 until ALVA can no longer "fool" $C$.

We implemented this technique on the MNIST dataset, which contains 60.000 images. We repeated the process step 1-3 100 times, generating 60 fake images per class in each iteration. The goal was to double the size of the original dataset with 60.000 real images and 60.000 fake images. The resulting composition of the dataset is shown in the following graph. Note that the generator was not trained during this process.

<p align="center">
<img src="https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/convergence_training/Dataset_Composition.jpeg" alt="Composition of dataset" width = 400/>
<p/>

This graph shows that ALVA was not able to generate 600 images every iteration, resulting in 1600 missing images. However, we also tracked the prediction accuracy every iteration, as shown in the graph below.

<p align="center">
<img src="https://github.com/BaumSebastian/ALVA/blob/main/docs/readme_pictures/convergence_training/Accuracy.jpeg" alt="Accuracy on Dataset" width = 700/>
<p/>

The graph demonstrates that by using convergence training with ALVA, we were able to double the size of the dataset while maintaining a high prediction accuracy on the original dataset (~98%) and a slightly lower accuracy on the expanded dataset (~94.5%). It is important to note that the ALVA method does not have the same properties as FGSM, which can result in a massive decrease in accuracy.
