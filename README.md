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
            └── Training and applying GaMorNet
        ├── Transfer learning with real galaxies
            ├── Data splitting
            ├── Realistic simulated AGN creation
            ├── Training PSFGAN
            ├── Applying trained PSFGAN
            ├── Generating morphological labels
            └── Fine-tuning and applying GaMorNet
        └── Applying trained PSFGAN and GaMorNet on real AGN
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
1) Train their own version of `PSFGAN-GaMorNet` with simulated and/or real data (jump to **'Initial training with simulated galaxies'** and **'Transfer learning with real galaxies'**)
2) Apply our trained version of `PSFGAN-GaMorNet` on real AGN in the HSC Wide Survey (jump to **'Applying on real AGN'**)
3) Use our code as a reference to build your customized computing framework based on `PSFGAN` and `GaMorNet`



Note: The `PSFGAN-GaMorNet` framework has multiple components, and they are expected to be executed in a **fixed** order. The output of the N-th component is by default the input of the (N+1)-th component.
However, if you already have data that is equivalent to the output of the N-th component, you may skip using the N-th and all previous components and jump to the (N+1)-th component directly.
### Initial training with simulated galaxies (standalone)
In this section, we will illustrate details in training single-band `PSFGAN` and `GaMorNet` (from scratch), all using simulated galaxies (simulated AGN).

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
    ├── main.py
    └── {other files and folders}
```

The `PSFGAN-GaMorNet` assumes raw data images are stored (in .fits format) in an `image` folder. There should also be a separate catalog file (in .csv format) that contains necessary information of each image. (Please refer to these files for detailed information)

We will use $150,000$ simulated galaxies (which were created w.r.t. 0<z<0.25 real galaxies in HSC Wide Survey) as example.
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
- `--save_raw_input`: `1` (whether to save created simulated AGN (simulated galaxies + added AGN point sources). `0`: No; `1`: Yes. **We suggest to save them.**)

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

Note: please use a `Python 2.7` environment for `PSFGAN` related tasks.

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
- `self.image_00`: `tf.slice(self.image, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.cond_00`: `tf.slice(self.cond, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.g_img_00`: `tf.slice(self.gen_img, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`

This sets a square region between pixel `[108, 108]` and pixel `[130, 130]`. When calculating the loss, pixels within this region will be treated `1/attention_parameter` times more important than generic pixels on the entire image. In the paper, we refer to this region as **"attention window"**.

Besides, in `discriminator(self, img, cond, reuse)` and `generator(self, cond)`, you may want to modify their structures. The default structure is suitable for `gal_sim_0_0.25`.

Once they are properly set, ran `python PSFGAN-GaMorNet/PSFGAN/train.py` to train a new version of `PSFGAN` from scratch using corresponding training data of simulated galaxies/AGN created in the previous step.

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
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'gal_sim_0_0.25' 
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'gal_sim_0_0.25' 
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/g-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'gal_sim_0_0.25' 
```

These commends will create new catalogs based on original catalogs. Each new catalog will contain three additional columns: `is_disk`, `is_indeterminate`, and `is_bulge`.

Please refer to the paper for exact rules we used in morphologcial label creation.
#### Training and applying GaMorNet
Now we start training a version of `GaMorNet` from scratch using (normalized) recovered (simulated) host galaxies in `gmn_train`.

Notes: 
1) Please use a `Python 3.6` environment for `GaMorNet` related tasks.
2) Install appropriate versions of `CUDA` and `cuDNN` if you are using a `GPU`. 
3) In our experiment, we used `Python 3.6.12` with `cuDNN 7.6.2.24` and `CUDA 10.0.130`.
4) See [GaMorNet Tutorial Pages about GPU](https://gamornet.readthedocs.io/en/latest/getting_started.html#gpu-support) for more information.

To start, please find the `main.py` file under `PSFGAN-GaMorNet/GaMorNet/`. Since we can directly load/unload `GaMorNet` modules, we have integrated all necessary `GaMorNet` related operations in this file. 

In `main.py`:

If you are using a `GPU`, ran the following before loading `GaMorNet` modules:
```bash
### Preparation
# First, we will check whether we have access to a GPU (required by the GPU version GaMorNet)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

print(tf.__version__)
print(gamornet.__version__)
```

Then, we should set a few parameters:
- `filter_string`: `'g'` (note: `GaMorNet` is single-band by design)
- `image_shape`: `[239, 239]` (should be the same image shape as in previous steps)
- `img_center`: `[119, 119]` (location of the user-defined central pixel)

Training and validation set paths: 
- `train_set`: `'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output_gmn_train/epoch_20/fits_output/' % filter_string` (location of the training set for `GaMorNet`, recovered galaxies --- **assuming you have changed the corresponding folder name as mentioned in previous steps**)
- `train_catalog`: `pandas.read_csv(glob.glob('PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/npy_input/catalog_gmn_train_npy_input_labeled.csv' % filter_string)[0])` (location of the corresponding catalog of the training set for `GaMorNet` --- **please make sure to use the morphologically labelled version!**)
- `eval_set`: `'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output_gmn_eval/epoch_20/fits_output/' % filter_string` (location of the validation set for `GaMorNet`, recovered galaxies --- **assuming you have changed the corresponding folder name as mentioned in previous steps**)
- `eval_catalog`: `pandas.read_csv(glob.glob('PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/npy_input/catalog_gmn_eval_npy_input_labeled.csv' % filter_string)[0])` (location of the corresponding catalog of the validation set for `GaMorNet` --- **please make sure to use the morphologically labelled version!**)

And the common test set paths:
- `pre_psf`: `'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/fits_test/' % filter_string` (location of the common test set, original galaxies)
- `post_psf`: `'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output_test/epoch_20/fits_output/' % filter_string` (location of the common test set, recovered galaxies --- **assuming you have changed the corresponding folder name as mentioned in previous steps**)
- `cond_input`: `'PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/fits_test_condinput/' % filter_string` (location of the common test set, original galaxies + AGN point sources --- **assuming you have saved the simulated AGN as suggested in previous steps**)
- `test_catalog`: `pandas.read_csv(glob.glob('PSFGAN-GaMorNet/PSFGAN/gal_sim_0_0.25/%s-band/asinh_50/npy_input/catalog_test_npy_input_labeled.csv' % filter_string)[0])` (location of the corresponding catalog of the common test set --- **please make sure to use the morphologically labelled version!**)

Once these parameters and paths are properly set, load all functions from:
```bash
### Defination
# Define a function to convert row_num into object_id
def row_to_id(catalog, row_num):
    return catalog.at[row_num, 'object_id']
```
to
```bash
# Define a function to save prediction labels (save to test_catalog)
def save_labels(pre_prediction_labels, post_prediction_labels, cond_prediction_labels, radius=0.0, type='',
                catalog=test_catalog, catalog_folder=''):
    length_catalog = len(catalog)

    catalog['pre_disk'] = [0.0]*length_catalog
    catalog['pre_indeterminate'] = [0.0]*length_catalog
    catalog['pre_bulge'] = [0.0]*length_catalog
    catalog['post_disk'] = [0.0] * length_catalog
    catalog['post_indeterminate'] = [0.0] * length_catalog
    catalog['post_bulge'] = [0.0] * length_catalog
    catalog['cond_disk'] = [0.0]*length_catalog
    catalog['cond_indeterminate'] = [0.0]*length_catalog
    catalog['cond_bulge'] = [0.0]*length_catalog

    row_num_list = list(range(length_catalog))
    for row_num in row_num_list:
        catalog.at[row_num, 'pre_disk'] = pre_prediction_labels[row_num, 0]
        catalog.at[row_num, 'pre_indeterminate'] = pre_prediction_labels[row_num, 1]
        catalog.at[row_num, 'pre_bulge'] = pre_prediction_labels[row_num, 2]
        catalog.at[row_num, 'post_disk'] = post_prediction_labels[row_num, 0]
        catalog.at[row_num, 'post_indeterminate'] = post_prediction_labels[row_num, 1]
        catalog.at[row_num, 'post_bulge'] = post_prediction_labels[row_num, 2]
        catalog.at[row_num, 'cond_disk'] = cond_prediction_labels[row_num, 0]
        catalog.at[row_num, 'cond_indeterminate'] = cond_prediction_labels[row_num, 1]
        catalog.at[row_num, 'cond_bulge'] = cond_prediction_labels[row_num, 2]

    # Save the catalog
    if type == '':
        catalog.to_csv(catalog_folder + 'catalog_test_entirety_' + str(length_catalog) + '.csv', index=False)
    else:
        catalog.to_csv(catalog_folder + 'catalog_test_entirety_' + str(length_catalog) + '_' + type + str(int(radius)) + '.csv', index=False)

```

Then, we are ready to invoke `GaMorNet`.

Load data: (**please change row number limits accordingly if you are using a different split than the one described in the paper.**)
```bash
radius = 0.0
type = ''

# Load data
# For sim_gal_0_0.25, sim_gal_0.25_0.5, sim_gal_0.5_0.75, sim_gal_0.5_1.0
training_imgs_0, training_labels_0 = load_train_data(row_num_limits=[0, 45000], radius=radius, type=type)
training_imgs_1, training_labels_1 = load_train_data(row_num_limits=[45000, 90000], radius=radius, type=type)
training_imgs = np.concatenate((training_imgs_0, training_imgs_1), axis=0)
training_labels = np.concatenate((training_labels_0, training_labels_1), axis=0)
validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 10000], radius=radius, type=type)
```

Train the model:
```bash
train_model_folder = 'PSFGAN-GaMorNet/GaMorNet/saves/{your favorite folder name}' 
if not os.path.exists(train_model_folder):
    os.makedirs(train_model_folder)
gamornet_train_keras(training_imgs=training_imgs, training_labels=training_labels, validation_imgs=validation_imgs, validation_labels=validation_labels,
                     input_shape=(239, 239, 1),
                     files_save_path=train_model_folder,
                     epochs=100, checkpoint_freq=0,
                     batch_size=128, lr=0.00005, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy',
                     load_model=False, model_load_path='./', save_model=True, verbose=2)
```
Note: for arguments in 'gamornet_train_keras', please refer to [GaMorNet API Documentation](https://gamornet.readthedocs.io/en/latest/api_docs.html#module-gamornet.tflearn_module) for more information.

Apply the model: (**please change row number limits accordingly if you are using a different split than the one described in the paper.**)
```bash
# Load pre and post PSFGAN results along with conditional inputs
pre_imgs = load_pre_psf(row_num_limits=[0, 40000], radius=radius, type=type)
post_imgs = load_post_psf(row_num_limits=[0, 40000], radius=radius, type=type)
cond_imgs = load_cond_inputs(row_num_limits=[0, 40000], radius=radius, type=type)
rsdl_imgs = post_imgs - pre_imgs
```

```bash
# Use the model to make prediction
test_model ='PSFGAN-GaMorNet/GaMorNet/saves/{your favorite folder name}/trained_model.hdf5'
pre_prediction_labels = gamornet_predict_keras(img_array=pre_imgs,
                                                 model_load_path=test_model,
                                                 input_shape=(image_shape[0], image_shape[1], 1),
                                                 batch_size=64, individual_arrays=False)
post_prediction_labels = gamornet_predict_keras(img_array=post_imgs,
                                                  model_load_path=test_model,
                                                  input_shape=(image_shape[0], image_shape[1], 1),
                                                  batch_size=64, individual_arrays=False)
cond_prediction_labels = gamornet_predict_keras(img_array=cond_imgs,
                                                  model_load_path=test_model,
                                                  input_shape=(image_shape[0], image_shape[1], 1),
                                                  batch_size=64, individual_arrays=False)
```

At last, save their outputs:
```bash
## Save the prediction labels together with the test catalog (creating a new catalog)
# Remember to set the catalog folder.
save_labels(pre_prediction_labels=pre_prediction_labels, post_prediction_labels=post_prediction_labels,
            cond_prediction_labels=cond_prediction_labels, radius=radius, type=type,
            catalog_folder={where you want to create a catalog containing model outputs})
```

### Transfer learning with real galaxies (please read 'Initial training with simulated galaxies' first)
In this section, we will illustrate details in training multi-band `PSFGAN` and fine-tuning previously trained `GaMorNet`, all using real galaxies (realistic simulated AGN).

Before proceed, please familarize yourself about the "Initial training with simulated galaxies" section. Since there is a huge overlap, we will not go over every detail. Instead, we will highlight their difference (whenever there is one).

Please make sure you have the following directory structure:
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
    └── simard
        ├──  g-band
            ├── catalog_star.csv
            ├── fits_star
            └── raw_data
                ├── images
                └── Simard_Match_HSCWideAll_SelectedRows.csv
        ├── r-band
            └── {same as in g-band}
        ├── i-band
            └── {same as in g-band}
        ├── z-band
            └── {same as in g-band}
        └── y-band
            └── {same as in g-band}
└── GaMorNet
    ├── main.py
    └── {other files and folders}
```

Note that each real galaxy has images in five HSC Wide filters: `g`, `r`, `i`, `z`, and `y`. Please refer to image files and catalogs for detailed information.

We will use $50,873$ real galaxies (mostly with $z<0.25$) selected from [Simard et al. (2011)](https://iopscience.iop.org/article/10.1088/0067-0049/196/1/11/pdf) (which are also imaged in HSC Wide Survey) as example.
#### Data splitting
Similar as in the "Initial training with simulated galaxies" section. Use `data_split.py` to split data. Set parameters accordingly. Ran `python PSFGAN-GaMorNet/PSFGAN/data_split.py`.

Notes:
- `filter_strings`: this should now be `['g', 'r', 'i', 'z', 'y']`
- `desired_shape`: this is still `[239, 239]`
#### Realistic simulated AGN creation
The next step is to create artificial AGN point sources, add them with real galaxies, and normalize all of these images using the chosen stretch function. This is similar as in the "Initial training with simulated galaxies" section, except now we have five filters. 

Star images and catalogs should have the same format as in the "Initial training with simulated galaxies" section.

Notes on a few parameters:

In `config.py`:
- `filters_`: this should now be `['g', 'r', 'i', 'z', 'y']`
- `if redshift == 'simard'`: then `pixel_max_value`: `45000` should be adequate for our dataset **(Once this value is chosen it should be fixed for the entire dataset `simard`)**

Please be aware that in each filter, we can have an individual AGN PS to host galaxy flux contrast ratio. In reality, the five contrast ratios are not independent. Nonetheless, for illustration purposes, we use five independently sampled contrast ratios in our guide. Please modify the corresponding code block accordingly, should you want to use real AGN SEDs (for example) to create realistically mutually-dependent contrast ratios in the five HSC Wide filters.
- `max_contrast_ratio` and `min_contrast_ratio`: `3.981` and `0.1` 
- `uniform_logspace`: `True`
- `num_star_per_psf`: `50`

Set other parameters accordingly.

In `roouhsc.py`:
- `--save_raw_input`: `1` (**We suggest to save them.**)
Others are similar as in the "Initial training with simulated galaxies" section. Set them accordingly.

Once all parameters are set, ran the following to create realistic simulated AGN (and normalize all images using the chosen stretch function):
```bash
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 0
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 1
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 2
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 3
python PSFGAN-GaMorNet/PSFGAN/roouhsc_agn.py --mode 4
```
#### Training PSFGAN
Now we start training a version of multi-band `PSFGAN` from scratch using (normalized) realistic simulated galaxies/AGN.

Note: please use a `Python 2.7` environment for `PSFGAN` related tasks.

Notes on parameters:

In `config.py`:
- `learning_rate`: `0.00009` 
- `attention_parameter`: `0.05` 
- `model_path`: `''` (during training, the model path **must** be an empty string)
- `start_epoch` and `max_epoch`: `0` and `20` (that is, to train for `20` epochs)
- `img_size`: `239`
- `train_size`: `239`

Other parameters should be kept the same as in the previous step.

In `model.py`:
- `self.image_00`: `tf.slice(self.image, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.cond_00`: `tf.slice(self.cond, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`
- `self.g_img_00`: `tf.slice(self.gen_img, [0, 108, 108, 0], [1, 22, 22, conf.img_channel])`

Besides, in `discriminator(self, img, cond, reuse)` and `generator(self, cond)`, you may want to modify their structures. The default structure is suitable for an image shape of `[239, 239]`. 

Once they are properly set, ran `python PSFGAN-GaMorNet/PSFGAN/train.py` to train a new version of `PSFGAN` from scratch using corresponding training data of realistic simulated galaxies/AGN created in the previous step.

Five **identical** copies of the trained multi-band model will be saved in similar locations in each of the five filter-subfolders. 

**Remember to change `.../PSFGAN_output/` to a different name IN EACH FILTER since application results on test data will also be saved in the same locations.**
#### Applying trained PSFGAN
Now it's time to apply the trained `PSFGAN` on the training/validation sets for `GaMorNet` and the common test set. We need to remove the added AGN point sources for these subsets with the aid of `PSFGAN`.

Set the following parameters before proceed:

In `config.py`:
- `model_path`: `PSFGAN-GaMorNet/PSFGAN/simard/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/model/model.ckpt` (location of the trained model --- you can also change `g-band` to other filter-subfolders since models saved in different filter-subfolders are identical)
- `test_epoch`: `20` (by default, this should equal to `max_epoch` --- assuming we want to apply the trained model at the maximum epoch)

Then, please make sure path `PSFGAN-GaMorNet/PSFGAN/simard/{filter}-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/PSFGAN_output/` is empty for each of the five filters.

You can now use command `python PSFGAN-GaMorNet/PSFGAN/test.py --mode gmn_train` and two other commands for `--mode gmn_eval` and `--mode test` to apply the trained model to remove added AGN point sources for `GaMorNet` and for the common test set. Change the according path names as in the "Initial training with simulated galaxies" section. **The only difference is that one needs to change folder names and keep original paths empty in all FIVE filter-subfolders.**
#### Generating morphological labels
Ran the following to create morphological labels for each filter-subfolder:
```bash
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/g-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/g-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/g-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'simard' --use_label 'n'

python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/r-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/r-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/r-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'simard' --use_label 'n'

python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/i-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/i-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/i-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'simard' --use_label 'n'

python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/z-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/z-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/z-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'simard' --use_label 'n'

python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/y-band/asinh_50/npy_input/catalog_gmn_train_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/y-band/asinh_50/npy_input/catalog_gmn_eval_npy_input.csv' --source 'simard' --use_label 'n'
python PSFGAN-GaMorNet/PSFGAN/add_label.py --input_path 'PSFGAN-GaMorNet/PSFGAN/simard/y-band/asinh_50/npy_input/catalog_test_npy_input.csv' --source 'simard' --use_label 'n'
```

Please refer to the paper for exact rules we used in morphologcial label creation.
#### Fine-tuning and applying GaMorNet
Now we start fine-tuning the version of `GaMorNet` previously trained from scratch in the "Initial training with simulated galaxies" section, using (normalized) recovered (real) host galaxies in `gmn_train`.

Notes: 
1) Please use a `Python 3.6` environment for `GaMorNet` related tasks.
2) Install appropriate versions of `CUDA` and `cuDNN` if you are using a `GPU`. 
3) In our experiment, we used `Python 3.6.12` with `cuDNN 7.6.2.24` and `CUDA 10.0.130`.
4) See [GaMorNet Tutorial Pages about GPU](https://gamornet.readthedocs.io/en/latest/getting_started.html#gpu-support) for more information.

To start, please find the `main.py` file under `PSFGAN-GaMorNet/GaMorNet/`. 

In `main.py`:

If you are using a `GPU`, ran the following before loading `GaMorNet` modules:
```bash
### Preparation
# First, we will check whether we have access to a GPU (required by the GPU version GaMorNet)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

print(tf.__version__)
print(gamornet.__version__)
```

Please be aware that although we used a multi-band `PSFGAN`, `GaMorNet` is single-band by design. **Please ran the following steps for each of the five filters (or for a subset of filters you're interested).**

Set a few parameters:
- `filter_string`: `{filter}` (e.g. `'g'`, `'r'`, etc.)
- `image_shape`: `[239, 239]` 
- `img_center`: `[119, 119]`

Then, set training set, validation set, and common test set paths accordingly, in the chosen filter-subfolder.

Once these parameters and paths are properly set, load all functions needed.

Then, we are ready to invoke `GaMorNet`.

Load data: (**please change row number limits accordingly if you are using a different split than the one described in the paper.**)
```bash
radius = 0.0
type = ''

# Load data
# For simard
training_imgs, training_labels = load_train_data(row_num_limits=[0, 32400], radius=radius, type=type)
validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 3600], radius=radius, type=type)
```

Fine-tune the previously learned model:
```bash
previous_model_folder = '{location of the previously trained model folder}'
tl_model_folder = '{destination folder --- fine-tuned model will be saved there}'
if not os.path.exists(tl_model_folder):
    os.makedirs(tl_model_folder)
gamornet_tl_keras(training_imgs=training_imgs, training_labels=training_labels, validation_imgs=validation_imgs, validation_labels=validation_labels,
                  input_shape=(239, 239, 1),
                  load_layers_bools=[True, True, True, True, True, True, True, True],
                  trainable_bools=[False, False, False, True, True, True, True, True],
                  model_load_path=previous_model_folder+'trained_model.hdf5',
                  files_save_path=tl_model_folder,
                  epochs=200, checkpoint_freq=0,
                  batch_size=128, lr=5e-05, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy',
                  save_model=True, verbose=2)
```
Note: for arguments in 'gamornet_train_keras', please refer to [GaMorNet API Documentation](https://gamornet.readthedocs.io/en/latest/api_docs.html#module-gamornet.tflearn_module) for more information.

Apply the model: (**please change row number limits accordingly if you are using a different split than the one described in the paper.**)
```bash
# Load pre and post PSFGAN results along with conditional inputs
pre_imgs = load_pre_psf(row_num_limits=[0, 9000], radius=radius, type=type)
post_imgs = load_post_psf(row_num_limits=[0, 9000], radius=radius, type=type)
cond_imgs = load_cond_inputs(row_num_limits=[0, 9000], radius=radius, type=type)
rsdl_imgs = post_imgs - pre_imgs
```

```bash
# Use the model to make prediction
test_model ='{location of the fine-tuned model folder}/trained_model.hdf5'
pre_prediction_labels = gamornet_predict_keras(img_array=pre_imgs,
                                                 model_load_path=test_model,
                                                 input_shape=(image_shape[0], image_shape[1], 1),
                                                 batch_size=64, individual_arrays=False)
post_prediction_labels = gamornet_predict_keras(img_array=post_imgs,
                                                  model_load_path=test_model,
                                                  input_shape=(image_shape[0], image_shape[1], 1),
                                                  batch_size=64, individual_arrays=False)
cond_prediction_labels = gamornet_predict_keras(img_array=cond_imgs,
                                                  model_load_path=test_model,
                                                  input_shape=(image_shape[0], image_shape[1], 1),
                                                  batch_size=64, individual_arrays=False)
```

At last, save their outputs:
```bash
## Save the prediction labels together with the test catalog (creating a new catalog)
# Remember to set the catalog folder.
save_labels(pre_prediction_labels=pre_prediction_labels, post_prediction_labels=post_prediction_labels,
            cond_prediction_labels=cond_prediction_labels, radius=radius, type=type,
            catalog_folder={where you want to create a catalog containing model outputs})
```

### Applying trained PSFGAN and GaMorNet on real AGN (standalone)
In this section, we will illustrate details of how to apply trained (multi-band) `PSFGAN` and `GaMorNet` on real AGN (from HSC Wide) in order to morphologically classify their host galaxies.

Before start, please make sure you have the following directory structure:
```bash
PSFGAN-GaMorNet/
├── PSFGAN 
    ├── add_label.py
    ├── config.py
    ├── data.py
    ├── data_split_agn.py
    ├── galfit.py
    ├── model.py
    ├── normalizing.py
    ├── photometry.py
    ├── roouhsc_agn.py
    ├── test.py
    ├── train.py
    ├── utils.py
    └── {target dataset name}
        ├──  g-band
            └── raw_data
                ├── images
                └── {catalog in .csv format}
        ├── r-band
            └── {same as in g-band}
        ├── i-band
            └── {same as in g-band}
        ├── z-band
            └── {same as in g-band}
        └── y-band
            └── {same as in g-band}
└── GaMorNet
    ├── main.py
    └── {other files and folders}
```
**Note: in this guide we assume your target dataset has images in five HSC Wide filters, but of course it can have less than five.**

**In addition to these files, you should also have (pre-)trained `PSFGAN` and `GaMorNet` (which you are going to apply) stored at some place you know.**

(In each filter) for the target dataset, its raw data images should be stored (in .fits format) in an `image` folder. There should also be a separate catalog file (in .csv format) that contains necessary information of each image. **The file name of each .fits image as well as its corresponding row in the catalog can have various forms.** **Please change codes in `data_split_agn.py`, `roouhsc_agn.py`, etc. appropriately so images can be correctly processed.**
#### Data splitting



#### Realistic simulated AGN creation
#### Training PSFGAN
#### Applying trained PSFGAN
#### Generating morphological labels
#### Fine-tuning and applying GaMorNet
### Notes on our trained PSFGAN and GaMorNet models
