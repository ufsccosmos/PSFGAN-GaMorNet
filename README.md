# PSFGAN-GaMorNet
This repository contains source code for the `PSFGAN-GaMorNet` framework (discussed in "Using Machine Learning to Determine Morphologies of $z<1$ AGN Host Galaxies in the Hyper Suprime-Cam Wide Survey").
## CONTENTS
```bash
PSFGAN-GaMorNet
    ├── CONTENTS
    ├── Dependencies
    ├── Installation
    └── A comprehensive guide to using PSFGAN-GaMorNet
        ├── Introduction
        ├── Initial training with simulated galaxies
            ├── Data splitting
            ├── Artificial AGN creation
            ├── Training PSFGAN
            ├── Applying trained PSFGAN
            ├── Generating morphological labels
            └── Training GaMorNet
        ├── Transfer learning with real galaxies
        └── Applying on real AGN
```
## Dependencies
`Linux` or `OSX`

`Python 2.7` for `PSFGAN`

`Python 3.6` for `GaMorNet`

Python modules (including but not limited to): `NumPy`, `SciPy`, `Astropy`, `Pandas`, `TensorFlow`, etc.
## Installation
Clone this repository and change the present working directory to `PSFGAN-GaMorNet/`.
## A comprehensive guide to using PSFGAN-GaMorNet
In this section, we will give an exhaustive guide about how to use our `PSFGAN-GaMorNet` framework as well as other related modules.
### Introduction
This guide will allow readers to:
1) Train their own version of `PSFGAN-GaMorNet` with simulated and/or real data
2) Apply our trained version of `PSFGAN-GaMorNet` on real AGN in the HSC Wide Survey
3) Use our code as a reference to build your customized computing framework based on `PSFGAN` and `GaMorNet`

Note: The `PSFGAN-GaMorNet` framework has multiple components, and they are expected to be executed in a **fixed** order. The output of the N-th component is by default the input of the (N+1)-th component.
However, if you already have data that is equivalent to the output of the N-th component, you may skip using the N-th and all previous components and jump to the (N+1)-th component directly.
### Initial training with simulated galaxies
Before start, please make sure you have the following directory structure:
```bash
PSFGAN-GaMorNet/
├── PSFGAN 
    ├── data_split.py
    └── gal_sim_0_0.25
        └── g-band
            └── raw_data
                ├── images
                └── sim_para_all.csv
└── GaMorNet
```

The `PSFGAN-GaMorNet` assumes raw data images are stored (in .fits format) in an `image` folder. There should also be a separate catalog file (in .csv format) that contains necessary information of each image. (Please refer to these files for detailed information)
In this guide, we will use simulated galaxies (which were created 
#### Data splitting


#### Artificial AGN creation
#### Training PSFGAN
#### Applying trained PSFGAN
#### Generating morphological labels
#### Training GaMorNet
