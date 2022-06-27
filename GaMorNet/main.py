# This Python script is used to run various modules of the GaMorNet

import argparse
import os
import sys
import glob
import pandas
import numpy as np
import random
from astropy.io import fits
import tensorflow as tf
import gamornet
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import seaborn as sns


### Preparation
# First, we will check whether we have access to a GPU (required by the GPU version GaMorNet)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.test.gpu_device_name() != '/device:GPU:0':
  print('WARNING: GPU device not found.')
else:
  print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

print(tf.__version__)
print(gamornet.__version__)

from gamornet.keras_module import gamornet_train_keras, gamornet_tl_keras, gamornet_predict_keras
from gamornet.tflearn_module import gamornet_train_tflearn, gamornet_tl_tflearn, gamornet_predict_tflearn




# The filter to be used
filter_string = 'g'
# Image shape to be used
image_shape = [239, 239]
# Image center when calculating radial profiles
img_center = [119, 119]
# Number of threads for multiprocessing
NUM_THREADS = 2
## Number of galaxies included when calculating prediction labels, start from the top row in the test catalog
#max_nsamples = 50000
# Folders and catalogs
# train_set = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/simard_hscw/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output_gmn_train/epoch_20/fits_output/'
train_set = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/dimauro_0.5_1.0/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-06/PSFGAN_output_gmn_train/epoch_40/fits_output/' % filter_string
train_catalog = pandas.read_csv(glob.glob('/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/dimauro_0.5_1.0/%s-band/asinh_50/npy_input/catalog_gmn_train_npy_input_labeled_n.csv' % filter_string)[0])
# eval_set = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/simard_hscw/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-05/PSFGAN_output_gmn_eval/epoch_20/fits_output/'
eval_set = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/dimauro_0.5_1.0/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_5e-06/PSFGAN_output_gmn_eval/epoch_40/fits_output/' % filter_string
eval_catalog = pandas.read_csv(glob.glob('/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/dimauro_0.5_1.0/%s-band/asinh_50/npy_input/catalog_gmn_eval_npy_input_labeled_n.csv' % filter_string)[0])

pre_psf = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/liu/%s-band/fits_test/' % filter_string
post_psf = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/liu/%s-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/PSFGAN_output/epoch_20/fits_output/' % filter_string
cond_input = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/liu/%s-band/fits_test/' % filter_string
test_catalog = pandas.read_csv(glob.glob('/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/liu/%s-band/asinh_50/npy_input/catalog_test_npy_input.csv' % filter_string)[0])


### Defination
# Define a function to convert row_num into object_id
def row_to_id(catalog, row_num):
    return catalog.at[row_num, 'object_id']

def row_to_id_iloc(catalog, row_num):
    catalog_row = catalog.iloc[row_num]
    return catalog_row['object_id']

# Define a function to modify an image (galaxy) as either a volcano or flattop mountain
def gal_to_donut(img, radius, type=''):
    x = np.arange(0, image_shape[0])
    y = np.arange(0, image_shape[1])

    mask_main = (x[np.newaxis,:]-img_center[0])**2 + (y[:,np.newaxis]-img_center[1])**2 < radius**2
    if type == 'ft':
        mask_ring = ((x[np.newaxis,:]-img_center[0])**2 + (y[:,np.newaxis]-img_center[1])**2 >= radius**2) & ((x[np.newaxis,:]-img_center[0])**2 + (y[:,np.newaxis]-img_center[1])**2 < (radius+1)**2)
        ring_avg = np.average(img[mask_ring])
        img[mask_main] = np.random.normal(loc=ring_avg, scale=0.01, size=img[mask_main].shape)
    elif type == 'vc':
        img[mask_main] = np.random.normal(loc=0.0, scale=0.01, size=img[mask_main].shape)

    return None

# Define image handlers for loading large array of images
def train_handler(row_num, radius=0.0, type=''):
    object_id = row_to_id(train_catalog, row_num)
    img = fits.getdata(train_set + str(int(object_id)) + '-' + filter_string + '.fits')
    if type == 'ft':
        gal_to_donut(img=img, radius=radius, type='ft')
    elif type == 'vc':
        gal_to_donut(img=img, radius=radius, type='vc')
    return np.reshape(img, newshape=(image_shape[0], image_shape[1], 1))

def eval_handler(row_num, radius=0.0, type=''):
    object_id = row_to_id(eval_catalog, row_num)
    img = fits.getdata(eval_set + str(int(object_id)) + '-' + filter_string + '.fits')
    if type == 'ft':
        gal_to_donut(img=img, radius=radius, type='ft')
    elif type == 'vc':
        gal_to_donut(img=img, radius=radius, type='vc')
    return np.reshape(img, newshape=(image_shape[0], image_shape[1], 1))

def pre_psf_handler(row_num, radius=0.0, type=''):
    object_id = row_to_id(test_catalog, row_num)
    img = fits.getdata(pre_psf + str(int(object_id)) + '-' + filter_string + '.fits')
    if type == 'ft':
        gal_to_donut(img=img, radius=radius, type='ft')
    elif type == 'vc':
        gal_to_donut(img=img, radius=radius, type='vc')
    return np.reshape(img, newshape=(image_shape[0], image_shape[1], 1))

def post_psf_handler(row_num, radius=0.0, type=''):
    object_id = row_to_id(test_catalog, row_num)
    img = fits.getdata(post_psf + str(int(object_id)) + '-' + filter_string + '.fits')
    if type == 'ft':
        gal_to_donut(img=img, radius=radius, type='ft')
    elif type == 'vc':
        gal_to_donut(img=img, radius=radius, type='vc')
    return np.reshape(img, newshape=(image_shape[0], image_shape[1], 1))

def cond_input_handler(row_num, radius=0.0, type=''):
    object_id = row_to_id(test_catalog, row_num)
    img = fits.getdata(cond_input + str(int(object_id)) + '-' + filter_string + '.fits')
    if type == 'ft':
        gal_to_donut(img=img, radius=radius, type='ft')
    elif type == 'vc':
        gal_to_donut(img=img, radius=radius, type='vc')
    return np.reshape(img, newshape=(image_shape[0], image_shape[1], 1))

# Also, define label handlers
def train_label_handler(row_num):
    return [train_catalog.at[row_num, 'is_disk'], train_catalog.at[row_num, 'is_indeterminate'], train_catalog.at[row_num, 'is_bulge']]

def eval_label_handler(row_num):
    return [eval_catalog.at[row_num, 'is_disk'], eval_catalog.at[row_num, 'is_indeterminate'], eval_catalog.at[row_num, 'is_bulge']]

# Define a function to load training data
def load_train_data(row_num_limits, radius, type, train_catalog=train_catalog):
    print('Begin loading data...')
    print(str(row_num_limits[1] - row_num_limits[0]) + ' galaxies to be loaded for the training set')
    print('Type: ' + type + ';')
    if not type == '':
        print('Radius=' + str(radius) + ';')

    pl = Pool(NUM_THREADS)

    train_handler_w_type = partial(train_handler, radius=radius, type=type)
    training_imgs = np.array(pl.map(train_handler_w_type, range(row_num_limits[0], row_num_limits[1])), dtype='float64')
    training_labels = np.array(pl.map(train_label_handler, range(row_num_limits[0], row_num_limits[1])), dtype='float64')

    return training_imgs, training_labels

# Define a function to load validation data
def load_eval_data(row_num_limits, radius, type, eval_catalog=eval_catalog):
    print('Begin loading data...')
    print(str(row_num_limits[1] - row_num_limits[0]) + ' galaxies to be loaded for the validation set')
    print('Type: ' + type + ';')
    if not type == '':
        print('Radius=' + str(radius) + ';')

    pl = Pool(NUM_THREADS)

    eval_handler_w_type = partial(eval_handler, radius=radius, type=type)
    validation_imgs = np.array(pl.map(eval_handler_w_type, range(row_num_limits[0], row_num_limits[1])), dtype='float64')
    validation_labels = np.array(pl.map(eval_label_handler, range(row_num_limits[0], row_num_limits[1])), dtype='float64')

    return validation_imgs, validation_labels

# Define a function to load pre PSFGAN results only

def load_pre_psf(row_num_limits, radius, type, test_catalog=test_catalog):

    print('Begin loading pre PSFGAN results...')
    print(str(row_num_limits[1] - row_num_limits[0]) + ' galaxies to be loaded for the test set')
    print('Type: ' + type + ';')
    if not type == '':
        print('Radius=' + str(radius) + ';')

    pl = Pool(NUM_THREADS)

    pre_psf_handler_w_type = partial(pre_psf_handler, radius=radius, type=type)
    pre_imgs = np.array(pl.map(pre_psf_handler_w_type, range(row_num_limits[0], row_num_limits[1])), dtype='float64')

    return pre_imgs

def load_post_psf(row_num_limits, radius, type, test_catalog=test_catalog):

    print('Begin loading post PSFGAN results...')
    print(str(row_num_limits[1] - row_num_limits[0]) + ' galaxies to be loaded for the test set')
    print('Type: ' + type + ';')
    if not type == '':
        print('Radius=' + str(radius) + ';')

    pl = Pool(NUM_THREADS)

    post_psf_handler_w_type = partial(post_psf_handler, radius=radius, type=type)
    post_imgs = np.array(pl.map(post_psf_handler_w_type, range(row_num_limits[0], row_num_limits[1])), dtype='float64')

    return post_imgs

def load_cond_inputs(row_num_limits, radius, type, test_catalog=test_catalog):

    print('Begin loading conditional inputs...')
    print(str(row_num_limits[1] - row_num_limits[0]) + ' galaxies to be loaded for the test set')
    print('Type: ' + type + ';')
    if not type == '':
        print('Radius=' + str(radius) + ';')

    pl = Pool(NUM_THREADS)

    cond_input_handler_w_type = partial(cond_input_handler, radius=radius, type=type)
    cond_imgs = np.array(pl.map(cond_input_handler_w_type, range(row_num_limits[0], row_num_limits[1])), dtype='float64')

    return cond_imgs


def center_flux(img, abs=False):
    img_center = img[99:139, 99:139]
    if abs == False:
        return np.sum(img_center)
    else: # abs == True
        return np.sum(np.abs(img_center))

# Define a function to calculate the pixelwise average of all images (along axis 0)
def pixelwise_average(imgs):
    num_imgs = imgs.shape[0]
    return np.sum(imgs, axis=0) / num_imgs


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



### Execution
## Modification parameters
radius = 0.0
type = ''
## Training

# Load data
# For sim_gal_0_0.25, sim_gal_0.25_0.5, sim_gal_0.5_0.75, sim_gal_0.5_1.0
#training_imgs_0, training_labels_0 = load_train_data(row_num_limits=[0, 45000], radius=radius, type=type)
#training_imgs_1, training_labels_1 = load_train_data(row_num_limits=[45000, 90000], radius=radius, type=type)
#training_imgs = np.concatenate((training_imgs_0, training_imgs_1), axis=0)
#training_labels = np.concatenate((training_labels_0, training_labels_1), axis=0)
#validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 10000], radius=radius, type=type)

# For simard_hscw
# training_imgs, training_labels = load_train_data(row_num_limits=[0, 27000], radius=radius, type=type)
# validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 3000], radius=radius, type=type)

# For simard
#training_imgs, training_labels = load_train_data(row_num_limits=[0, 32400], radius=radius, type=type)
#validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 3600], radius=radius, type=type)

# For dimauro_0_0.5, dimauro_0.5_0.75, dimauro_0.5_1.0
training_imgs, training_labels = load_train_data(row_num_limits=[0, 6300], radius=radius, type=type)
validation_imgs, validation_labels = load_eval_data(row_num_limits=[0, 700], radius=radius, type=type)


# Train the model (gal_sim_0_0.25, gal_sim_0.25_0.5, gal_sim_0.5_0.75, gal_sim_0.5_1.0)
train_model_folder = '/gpfs/loomis/project/urry/ct564/HSC/GaMorNet/saves/gal_sim_0.5_1.0/%s-band/contrast_0.1to3.981/PSFGAN_lr0.00002_box12by12_epc20/GaMorNet_epc100_bs128_lr0.00005_mmt0.9_dc0.0_nstrF_lsCE/' % filter_string
if not os.path.exists(train_model_folder):
    os.makedirs(train_model_folder)
gamornet_train_keras(training_imgs=training_imgs, training_labels=training_labels, validation_imgs=validation_imgs, validation_labels=validation_labels,
                     input_shape=(239, 239, 1),
                     files_save_path=train_model_folder,
                     epochs=100, checkpoint_freq=0,
                     batch_size=128, lr=0.00005, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy',
                     load_model=False, model_load_path='./', save_model=True, verbose=2)

# Transfer learn the trained model (simard, dimauro_0_0.5, dimauro_0.5_0.75 and dimauro_0.5_1.0)
previous_model_folder = '/gpfs/loomis/project/urry/ct564/HSC/GaMorNet/saves/gal_sim_0.5_1.0/i-band/contrast_0.1to3.981/PSFGAN_lr0.00002_box12by12_epc20/GaMorNet_epc100_bs128_lr0.00005_mmt0.9_dc0.0_nstrF_lsCE/'
tl_model_folder = previous_model_folder+'transfer_learning_models/dimauro_0.5_1.0/%s-band/contrast_0.1to3.981/PSFGAN_lr0.000005_box12by12_epc40/TTTTTTTT_FFFTTTTT_epc200_bs128_lr0.00005_mmt0.9_dc0.0_nstrF_lsCE/' % filter_string
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


## Prediction
# Load pre and post PSFGAN results
#pre_imgs, post_imgs = load_psf_results(row_num_limits=[0, 40000])
pre_imgs = load_pre_psf(row_num_limits=[0, 40000], radius=radius, type=type)
post_imgs = load_post_psf(row_num_limits=[0, 40000], radius=radius, type=type)
cond_imgs = load_cond_inputs(row_num_limits=[0, 40000], radius=radius, type=type)
rsdl_imgs = post_imgs - pre_imgs
# Use the model to make prediction
test_model ='/gpfs/loomis/project/urry/ct564/HSC/GaMorNet/saves/gal_sim_0_0.25/g-band/contrast_0.1to3.981/PSFGAN_lr0.00005_box22by22_epc20/GaMorNet_epc100_bs256_lr0.00005_mmt0.9_dc0.0_nstrF_lsCE/' \
            + 'transfer_learning_models/simard/%s-band/contrast_0.1to3.981/PSFGAN_lr0.00009_box22by22_epc20/*TTTTTFFF_FTTTTTTT_epc100_bs128_lr0.00005_mmt0.9_dc0.0_nstrF_lsCE/trained_model.hdf5' % filter_string
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



## Save the prediction labels together with the test catalog (creating a new catalog)
# Remember to set the catalog folder.
save_labels(pre_prediction_labels=pre_prediction_labels, post_prediction_labels=post_prediction_labels,
            cond_prediction_labels=cond_prediction_labels, radius=radius, type=type)

