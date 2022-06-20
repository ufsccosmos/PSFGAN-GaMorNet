# PSFGAN-GaMorNet
This repository contains source code for the `PSFGAN-GaMorNet` framework (discussed in "Using Machine Learning to Determine Morphologies of $z<1$ AGN Host Galaxies in the Hyper Suprime-Cam Wide Survey").
## Environment \& Dependencies
`Linux` or `OSX`

`Python 2.7` for `PSFGAN`

`Python 3.6` for `GaMorNet`

Necessary packages (including but not limited to): `NumPy`, `SciPy`, `Astropy`, `Pandas`, `TensorFlow`, etc.
## Installation
Clone this repository and change the present working directory to `PSFGAN-GaMorNet/`.
## A comprehensive guide to using PSFGAN-GaMorNet
In this section, we will give an exhaustive guide about how to use our `PSFGAN-GaMorNet` framework as well as other related modules.
### Introduction
The `PSFGAN-GaMorNet` framework has multiple components, and they are expected to be executed in a **fixed** order. The output of the N-th component is by default the input od the (N+1)-th component.
