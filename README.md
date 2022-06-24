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
            ├── Simulated AGN creation
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
    ├── add_label.py
    ├── config.py
    ├── data.py
    ├── data_split.py
    ├── galfit.py
    ├── model.py
    ├── normalizing.py
    ├── photometry.py
    ├── roouhsc.py
    ├── test.py
    ├── train.py
    ├── utils.py
    └── gal_sim_0_0.25
        └── g-band
            ├── catalog_star.csv
            ├── fits_star
            └── raw_data
                ├── images
                └── sim_para_all.csv
             
└── GaMorNet
```

The `PSFGAN-GaMorNet` assumes raw data images are stored (in .fits format) in an `image` folder. There should also be a separate catalog file (in .csv format) that contains necessary information of each image. (Please refer to these files for detailed information)
In this guide, we will use $150,000$ simulated galaxies (which were created w.r.t. 0<z<0.25 real galaxies in HSC Wide Survey) as example.
#### Data splitting
The first step is to split raw data images into five subsets:
- `fits_train` (training set for `PSFGAN`)
- `fits_eval` (validation set for `PSFGAN`)
- `gmn_train` (training set for `GaMorNet`)
- `gmn_eval` (validation set for `GaMorNet`)
- `fits_test` (common test set)

To do so, we will need to use `data_split.py`. Set the following parameters to correct values before proceed:
- `core_path`: path in which PSFGAN is stored (see above)
- `galaxy_main`: `core_path` + `gal_sim_0_0.25/`
- `filter_strings`: `['g']` (filter(s) of raw data images)
- `desired_shape`: `[239, 239]` (desired shape of output images in pixels)
- `--gmn_train`, `--gmn_eval`, `--psf_train`, `--psf_eval`, and `--test`: set their default values to numbers of images you want each subset to have (they should sum up to $150,000$, the number of images in `raw_data` of `gal_sim_0_0.25`)
- `--shuffle`: `1` (`1` to shuffle images before splitting, `0` otherwise)
- `--source`: `'gal_sim_0_0.25'` (name of the source of the raw data --- this should be the same of the corresponding folder name)
- `--split`: `equal` (`equal` will ensure roughly the same ratio of disks to bulges to indeterminates across subsets --- look for the corresponding part in the source code for `unequal` split)

Once these parameters are properly set, ran `python PSFGAN-GaMorNet/PSFGAN/data_split.py`.
Corresponding folders and their associated catalogs will be created.
#### Simulated AGN creation
The next step is to create artificial AGN point sources, add them with simulated galaxies, and normalize all of these images using the chosen stretch function.

The `PSFGAN-GaMorNet` assumes star images (used to create AGN PS) are stored (in .fits format) in a `fits_star` folder.
There should also be a separate catalog file, `catalog_star.csv`, that contains necessary information of each star.
(Please refer to these files for detailed information)

Certain parameters need to be properly set before we proceed:

In `config.py`:
- `redshift`:  `'gal_sim_0_0.25'` (name of the source of our data)
- `filters_`: `['g']` (filter(s) of data images)
- `stretch_type` and `scale_factor`: `'asinh'` and `50` (we suggest to use these values to start --- feel free to change as you wish)
- `if redshift == 'gal_sim_0_0.25':` then `pixel_max_value`: `25000` (the largest pixel value allowed (pre-normliazation)) **Once this value is chosen it should be fixed for the entire dataset** `gal_sim_0_0.25`. This value (`25000`) is adequate for `gal_sim_0_0.25`. If you are using your own data, please make sure no pixel value (pre-normliazation) is larger than `pixel_max_value`, and `pixel_max_value` is not far larger than the maximum pixel value (pre-normliazation).
- `max_contrast_ratio` and `min_contrast_ratio`: `3.981` and `0.1` (what they do are self-evident)
- `uniform_logspace`: `True` (contrast ratios will be uniformly distributed in logspace if this is `True`, or they will be uniformly distributed in linearspace if this is `False`)
- `num_star_per_psf`: `50` (how many stars you want to use to create each artificial AGN PS)

In `roouhsc.py`:
- `--source`: `'gal_sim_0_0.25'` (name of the source of the data --- this should be the same of the corresponding folder name)
- `--crop`: `0` (set this to be zero so images are not cropped during normalization)
- `--save_psf`: `0` (whether to save created artificial AGN point sources. `0`: No; `1`: Yes)
- `--save_raw_input`: `1` (whether to save created simulated AGN (simulated galaxies + added AGN point sources). `0`: No; `1`: Yes)

Once all parameters are set, ran the following to create simulated AGN (and normalize all images using the chosen stretch function) for all five subsets:
```bash
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 0
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 1
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 2
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 3
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 4
```

Corresponding folders and associated catalogs will be created. 

Normalized simulated galaxies and simulated AGN are stored (in .npy format) in corresponding folders under `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/`. So as their associated catalogs.
#### Training PSFGAN
Now we start training a version of `PSFGAN` from scratch using (normalized) simulated galaxies/AGN.

Relevant parameters and settings:

In `config.py`:
- `learning_rate`: `0.00005` (we suggest to use `0.00005` for this particular dataset, yet please feel free to explore other values)
- `attention_parameter`: `0.05` (it governs the relative importance of the central focused region in loss calculation --- see below)
- `model_path`: `''` (during training, the model path **must** be an empty string)
- `start_epoch` and `max_epoch`: `0` and `20` (that is, to train for `20` epochs)
- `img_size`: `239`
- `train_size`: `239`

Other parameters should be kept the same as in the previous step.

In `model.py`:
- `self.image_11`: `tf.slice(self.image, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.cond_11`: `tf.slice(self.cond, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.g_img_11`: `tf.slice(self.gen_img, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`

This sets a square region between pixel `[108, 108]` and pixel `[130, 130]`. When calculating the loss, pixels within this region will be treated `1/attention_parameter` times more important than generic pixels on the entire image. In the paper, we refer to this region as **"attention window"**.

Besides, in `discriminator(self, img, cond, reuse)` and `generator(self, cond)`, you may want to modify their structures. The default structure is suitable for `gal_sim_0_0.25`.

Once they are properly set, in a `Python 2.7` environment, ran `python PSFGAN-GaMorNet/PSFGAN/train.py` to train a new version of `PSFGAN` from scratch using corresponding training data of simulated galaxies/AGN created in the previous step.

The trained model will be saved in `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/model/ `.

Application results of trained model (individual epochs) on corresponding validation data will be saved under `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output/`. 

**Remember to change `.../PSFGAN_output/` to a different name since application results on test data will also be saved in the same location.**
#### Applying trained PSFGAN
Now it's time to apply the trained `PSFGAN` on the training/validation sets for `GaMorNet` and the common test set. We need to remove the added AGN point sources for these subsets with the aid of `PSFGAN`.

Set the following parameters before proceed:
In `config.py`:
- `model_path`: `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/model/model.ckpt` (location of the trained model)
- `test_epoch`: `20` (by default, this should equal to `max_epoch` --- assuming we want to apply the trained model at the maximum epoch)

Then, please make sure path `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output/` is empty. 

Ran `python PSFGAN-GaMorNet/PSFGAN/test.py --mode gmn_train` to apply the trained model on the training set for `GaMorNet`. Change `.../PSFGAN_output/` to another name (e.g. `.../PSFGAN_output_gmn_train/`) to make sure path `PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output/` is kept empty. 

Repeat this step with mode `gmn_train`  (`test`) to apply the trained model on the validation set for `GaMorNet` (common test set). Change the folder name after each time of application.
#### Generating morphological labels
Before training `GaMorNet`, one should create morphologcial labels for galaxies in the training/validation sets for `GaMorNet` and the common test set, using relevant parameters in corresponding catalogs. 

To do so, simply ran the following:

```bash
python /gpfs/loomis/project/urry/ct564/HSC/data/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'gal_sim_0_0.25' 
python /gpfs/loomis/project/urry/ct564/HSC/data/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'gal_sim_0_0.25' 
python /gpfs/loomis/project/urry/ct564/HSC/data/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'gal_sim_0_0.25' 
```

These commends will create new catalogs based on original catalogs. Each new catalog will contain three additional columns: `is_disk`, `is_indeterminate`, and `is_bulge`.

Please refer to the paper for exact rules we used in morphologcial label creation.
#### Training GaMorNet
