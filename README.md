# Pytorch VAE

A stripped down fork of [https://github.com/AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)

---

## Description

A collection of Variational AutoEncoders (VAEs) implemented in pytorch with focus on reproducibility. The aim of this project is to provide
a quick and simple working example for many of the cool VAE models out there. All the models are trained on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
for consistency and comparison. The architecture of all the models are kept as similar as possible with the same layers, except for cases where the original paper necessitates 
a radically different architecture (Ex. VQ VAE uses Residual layers and no Batch-Norm, unlike other models).
Here are the [results](https://github.com/AntixK/PyTorch-VAE/blob/master/README.md#--results) of each model.

## Requirements
- Python >= 3.6
- PyTorch >= 1.5.0
- Pytorch Lightning >= 1.0.0 ([GitHub Repo](https://github.com/PyTorchLightning/pytorch-lightning/tree/deb1581e26b7547baf876b7a94361e60bb200d32))

## Installation
```
$ git clone https://github.com/AntixK/PyTorch-VAE
$ cd PyTorch-VAE
$ pip install -r requirements.txt
```