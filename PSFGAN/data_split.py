# Note: use to preprocess simulated or real HSC galaxies
# We will split the simulation dataset into five subsets:
# 1.training set for GaMorNet 
# 2.validation set for GaMorNet 
# 3.training set for PSFGAN 
# 4.validation set for PSFGAN 
# 5.common test set for GaMorNet + PSFGAN 
# Modified from "sim_gal_preprocess.py"

import argparse
import os
import glob
import pandas
import numpy as np
import random
from astropy.io import fits

# Paths
core_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/'
galaxy_main = core_path + 'dimauro_0.5_1.0/'

# Other parameters
# Order doesn't matter (e.g. ['g', 'r'] is the same as ['r', 'g'])
filter_strings = ['g', 'r', 'i', 'z', 'y']

# flux conversion parameters 
# i.e. [flux in nanoJy] * nJy_to_adu_per_AA = [flux in adu]
# HSC uses nanoJy; GalSim uses adu
# HSCWide-G: radius=4.1m, exp_time=10min, quantum efficiency=0.864, gain=3.0, lambda effctive=4754 Angstrom
# HSCWide-R: radius=4.1m, exp_time=10min, quantum efficiency=0.956, gain=3.0, lambda effctive=6175 Angstrom
# HSCWide-I: radius=4.1m, exp_time=20min, quantum efficiency=0.882, gain=3.0, lambda effctive=7711 Angstrom
# HSCWide-Z: radius=4.1m, exp_time=20min, quantum efficiency=0.821, gain=3.0, lambda effctive=8898 Angstrom
# HSCWide-Y: radius=4.1m, exp_time=20min, quantum efficiency=0.517, gain=3.0, lambda effctive=9762 Angstrom
#nJy_to_adu_per_AA_filters = [0.0289698414, 0.0246781434, 0.0364652697, 0.0294152337, 0.0168839201]
nJy_to_adu_per_AA_filters = [0.0289698414, 0.0246781434, 0.0364652697, 0.0294152337, 0.0168839201]
    

# The desired image shape. Images of other shapes will not pass the selection (thus be filtered out)
desired_shape = [239, 239]

parser = argparse.ArgumentParser()

def data_split():
    # Make the split predictable
    np.random.seed(42)
    parser.add_argument("--gmn_train", default=6300)
    parser.add_argument("--gmn_eval", default=700)
    parser.add_argument("--psf_train", default=4500)
    parser.add_argument("--psf_eval", default=500)
    parser.add_argument("--test", default=1528)
    parser.add_argument("--shuffle", default="1")
    # Identify source of the raw data. This will determine the names of columns in catalogs being created
    # Options: "sim_hsc_0_0.25", "simard_cross_hsc"...(more to be added)
    parser.add_argument("--source", default="dimauro_0.5_1.0")
    parser.add_argument("--split", default="unequal")
    args = parser.parse_args()

    gmn_train = int(args.gmn_train)
    gmn_eval = int(args.gmn_eval)
    psf_train = int(args.psf_train)
    psf_eval = int(args.psf_eval)
    test = int(args.test)
    shuffle = bool(int(args.shuffle))
    source = str(args.source)
    split = str(args.split)
    num_filters = len(filter_strings)

    num_total = 0
    num_gmn_train = 0
    num_gmn_eval = 0
    num_psf_train = 0
    num_psf_eval = 0
    num_test = 0
    num_resized = 0
    num_correctly_resized = 0
    num_negative_flux = 0

    # Input and output locations
    hsc_folders = []
    hsc_catalogs = []

    gmn_train_folders = []
    gmn_eval_folders = []
    psf_train_folders = []
    psf_eval_folders = []
    test_folders = []

    gmn_train_catalogs = []
    gmn_eval_catalogs = []
    psf_train_catalogs = []
    psf_eval_catalogs = []
    test_catalogs = []

    for filter_string in filter_strings:
        galaxy_per_filter = galaxy_main + filter_string + '-band/'

        hsc_folder = glob.glob(galaxy_per_filter + 'raw_data/images/')[0]
        hsc_catalog = pandas.read_csv(glob.glob(galaxy_per_filter + 'raw_data/*.csv')[0])

        gmn_train_folder = galaxy_per_filter + 'gmn_train/'
        gmn_eval_folder = galaxy_per_filter + 'gmn_eval/'
        psf_train_folder = galaxy_per_filter + 'fits_train/'
        psf_eval_folder = galaxy_per_filter + 'fits_eval/'
        test_folder = galaxy_per_filter + 'fits_test/'
        if not os.path.exists(gmn_train_folder):
            os.makedirs(gmn_train_folder)
        if not os.path.exists(gmn_eval_folder):
            os.makedirs(gmn_eval_folder)
        if not os.path.exists(psf_train_folder):
            os.makedirs(psf_train_folder)
        if not os.path.exists(psf_eval_folder):
            os.makedirs(psf_eval_folder)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
            column_list = ['object_id', 'num_components', 'sersic_idx_d', 'R_e_d', 'axis_ratio_d', 'PA_d', 'flux_frac_d',
                          'sersic_idx_b', 'R_e_b', 'axis_ratio_b', 'PA_b', 'flux_frac_b',
                          filter_string + '_total_flux']
            gmn_train_catalog = pandas.DataFrame(columns=column_list)
            gmn_eval_catalog = pandas.DataFrame(columns=column_list)
            psf_train_catalog = pandas.DataFrame(columns=column_list)
            psf_eval_catalog = pandas.DataFrame(columns=column_list)
            test_catalog = pandas.DataFrame(columns=column_list)
        elif source == "simard":
            column_list = ['object_id', 'ra', 'dec', 'photoz_best', 'SClass', 'z', 'Scale', 'Rhlg', 'Rhlr', 'Rchl,g', 'Rchl,r',
                          '(B/T)g', 'e(B/T)g', '(B/T)r', 'e(B/T)r',
                          filter_string + '_total_flux']
            gmn_train_catalog = pandas.DataFrame(columns=column_list)
            gmn_eval_catalog = pandas.DataFrame(columns=column_list)
            psf_train_catalog = pandas.DataFrame(columns=column_list)
            psf_eval_catalog = pandas.DataFrame(columns=column_list)
            test_catalog = pandas.DataFrame(columns=column_list)
        elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
            column_list = ['object_id', 'ra', 'dec', 'photoz_best', 'RE_F606', 'RE_F814', 'RE_F125', 'RE_F160',
                          'N_F606', 'N_F814', 'N_F125', 'N_F160', 'B_T_m',
                          filter_string + '_total_flux']
            gmn_train_catalog = pandas.DataFrame(columns=column_list)
            gmn_eval_catalog = pandas.DataFrame(columns=column_list)
            psf_train_catalog = pandas.DataFrame(columns=column_list)
            psf_eval_catalog = pandas.DataFrame(columns=column_list)
            test_catalog = pandas.DataFrame(columns=column_list)

        hsc_folders.append(hsc_folder)
        hsc_catalogs.append(hsc_catalog)
        gmn_train_folders.append(gmn_train_folder)
        gmn_eval_folders.append(gmn_eval_folder)
        psf_train_folders.append(psf_train_folder)
        psf_eval_folders.append(psf_eval_folder)
        test_folders.append(test_folder)
        gmn_train_catalogs.append(gmn_train_catalog)
        gmn_eval_catalogs.append(gmn_eval_catalog)
        psf_train_catalogs.append(psf_train_catalog)
        psf_eval_catalogs.append(psf_eval_catalog)
        test_catalogs.append(test_catalog)


    # Main loop
    # Start the loop by iterating over the row number based on the first catalog from hsc_catalogs
    row_num_list = list(range(2, len(hsc_catalogs[0]) + 2))
    
    # Equal or unequal data split
    # When using "unequal" split, please make sure "hsc_catalogs[0]" is already labeled.
    if split == "equal":
        if shuffle:
            np.random.shuffle(row_num_list)
            
    elif split == "unequal":
        # Get the bulge list first
        bulge_list = list(hsc_catalogs[0]["is_bulge"])
        num_bulges = np.sum(bulge_list)
        num_non_bulges = len(hsc_catalogs[0]) - num_bulges
        # Then sort "row_num_list" according to "bulge_list" (bulges will be sorted to the bottom)
        row_num_list = [x for _, x in sorted(zip(bulge_list, row_num_list))]
        
        # If shuffle is True:
        # First shuffle subset of bulges and subset of nonbulges
        if shuffle:
            non_bulge_row_num_list = row_num_list[:num_non_bulges]
            bulge_row_num_list = row_num_list[num_non_bulges:]
            np.random.shuffle(non_bulge_row_num_list)
            np.random.shuffle(bulge_row_num_list)
            row_num_list = non_bulge_row_num_list + bulge_row_num_list
        
        # Next shuffle subset of psf_train&psf_eval and subset of gmn_train&gmn_eval&test
        if shuffle:
            psf_row_num_list = row_num_list[:(psf_train+psf_eval)]
            gmn_test_row_num_list = row_num_list[(psf_train+psf_eval):]
            np.random.shuffle(psf_row_num_list)
            np.random.shuffle(gmn_test_row_num_list)
            row_num_list = psf_row_num_list + gmn_test_row_num_list

    
    for row_num in row_num_list:
        if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
            obj_id = int(row_num - 2)
        elif (source == "simard") or (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
            obj_id = int(row_num)

        # Read the images
        images = []
        for i in range(num_filters):
            if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                fits_path = '%s/%s.fits' % (hsc_folders[i], obj_id)
            elif (source == "simard") or (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                fits_path = '%s/%s-cutout-*.fits' % (hsc_folders[i], obj_id)
            file = glob.glob(fits_path)[0]
            image = fits.getdata(file)
            images.append(image)

        # Check whether the flux is positive in each filter
        # If not, quit the loop
        positive_flux_booleans = []
        for i in range(num_filters):
            current_row = hsc_catalogs[i].iloc[row_num-2]
            if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                total_flux = current_row['total_flux']
            elif (source == "simard") or (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                total_flux = current_row[filter_strings[i] + '_cmodel_flux']
            if total_flux < 0:
                positive_flux_boolean = False
            else:
                positive_flux_boolean = True
            positive_flux_booleans.append(positive_flux_boolean)
        if False in positive_flux_booleans:
            num_negative_flux += 1
            continue
            
        # Check whether the images have desired shapes in each filter
        # If not, resize the image
        desired_shape_booleans = []
        for i in range(num_filters):
            current_shape = list(images[i].shape)
            if not (current_shape[0] == desired_shape[0] and current_shape[1] == desired_shape[1]):
                desired_shape_boolean = False
                
                # Start to resize the first dimension
                if current_shape[0] < desired_shape[0]:
                    if (desired_shape[0]-current_shape[0]) % 2 == 0:
                        images[i] = np.pad(images[i], (( (desired_shape[0]-current_shape[0])//2, (desired_shape[0]-current_shape[0])//2 ), (0, 0)), 'reflect')
                    else: # (desired_shape[0] - current_shape[0]) % 2 == 1:
                        images[i] = np.pad(images[i], (( (desired_shape[0]-current_shape[0])//2, (desired_shape[0]-current_shape[0])//2 + 1), (0, 0)), 'reflect')
                elif current_shape[0] > desired_shape[0]:
                    if (current_shape[0]-desired_shape[0]) % 2 == 0:
                        images[i] = images[i][(current_shape[0]-desired_shape[0])//2 : -((current_shape[0]-desired_shape[0])//2), :]
                    else: # (current_shape[0]-desired_shape[0]) % 2 == 1:
                        images[i] = images[i][(current_shape[0]-desired_shape[0])//2: -((current_shape[0]-desired_shape[0])//2 + 1), :]
                # Then resize the second dimension
                if current_shape[1] < desired_shape[1]:
                    if (desired_shape[1]-current_shape[1]) % 2 == 0:
                        images[i] = np.pad(images[i], ((0, 0), ( (desired_shape[1]-current_shape[1])//2, (desired_shape[1]-current_shape[1])//2 )), 'reflect')
                    else: # (desired_shape[1]-current_shape[1]) % 2 == 1:
                        images[i] = np.pad(images[i], ((0, 0), ( (desired_shape[1]-current_shape[1])//2, (desired_shape[1]-current_shape[1])//2 + 1)), 'reflect')
                elif current_shape[1] > desired_shape[1]:
                    if (current_shape[1]-desired_shape[1]) % 2 == 0:
                        images[i] = images[i][:, (current_shape[1]-desired_shape[1])//2 : -((current_shape[1]-desired_shape[1])//2)]
                    else: # (current_shape[1]-desired_shape[1]) % 2 == 1:
                        images[i] = images[i][:, (current_shape[1]-desired_shape[1])//2: -((current_shape[1]-desired_shape[1])//2 + 1)]
                
            else:
                desired_shape_boolean = True
            desired_shape_booleans.append(desired_shape_boolean)
        if False in desired_shape_booleans:
            num_resized += 1
        
        # Check if each galaxy has been correctly resized
        if False in desired_shape_booleans:
            correctly_resized_booleans = []
            for i in range(num_filters):
                current_shape = list(images[i].shape)
                if not (current_shape[0] == desired_shape[0] and current_shape[1] == desired_shape[1]):
                    correctly_resized_boolean = False
                else:
                    correctly_resized_boolean = True
                correctly_resized_booleans.append(correctly_resized_boolean)
            if False not in correctly_resized_booleans:
                num_correctly_resized += 1
        

        # Otherwise, let's proceed
        if num_psf_train < psf_train:
            for i in range(num_filters):
                # Save the image
                image_name = psf_train_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                    psf_train_catalogs[i] = psf_train_catalogs[i].append({'object_id': obj_id, 'num_components': current_row['num_components'],
                    'sersic_idx_d': current_row['sersic_idx_d'],
                    'R_e_d': current_row['R_e_d'],
                    'axis_ratio_d': current_row['axis_ratio_d'],
                    'PA_d': current_row['PA_d'],
                    'flux_frac_d': current_row['flux_frac_d'],
                    'sersic_idx_b': current_row['sersic_idx_b'],
                    'R_e_b': current_row['R_e_b'],
                    'axis_ratio_b': current_row['axis_ratio_b'],
                    'PA_b': current_row['PA_b'],
                    'flux_frac_b': current_row['flux_frac_b'],
                    filter_strings[i] + '_total_flux': current_row['total_flux']}, ignore_index=True)
                elif source == "simard":
                    psf_train_catalogs[i] = psf_train_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'SClass': current_row['SClass'],
                    'z': current_row['z'],
                    'Scale': current_row['Scale'],
                    'Rhlg': current_row['Rhlg'],
                    'Rhlr': current_row['Rhlr'],
                    'Rchl,g': current_row['Rchl,g'],
                    'Rchl,r': current_row['Rchl,r'],
                    '(B/T)g': current_row['(B/T)g'],
                    'e(B/T)g': current_row['e(B/T)g'],
                    '(B/T)r': current_row['(B/T)r'],
                    'e(B/T)r': current_row['e(B/T)r'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                    psf_train_catalogs[i] = psf_train_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160'],
                    'B_T_m': current_row['B_T_m'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
            num_psf_train += 1
            num_total += 1
            continue
        if num_psf_eval < psf_eval:
            for i in range(num_filters):
                # Save the image
                image_name = psf_eval_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                    psf_eval_catalogs[i] = psf_eval_catalogs[i].append({'object_id': obj_id, 'num_components': current_row['num_components'],
                    'sersic_idx_d': current_row['sersic_idx_d'],
                    'R_e_d': current_row['R_e_d'],
                    'axis_ratio_d': current_row['axis_ratio_d'],
                    'PA_d': current_row['PA_d'],
                    'flux_frac_d': current_row['flux_frac_d'],
                    'sersic_idx_b': current_row['sersic_idx_b'],
                    'R_e_b': current_row['R_e_b'],
                    'axis_ratio_b': current_row['axis_ratio_b'],
                    'PA_b': current_row['PA_b'],
                    'flux_frac_b': current_row['flux_frac_b'],
                    filter_strings[i] + '_total_flux': current_row['total_flux']}, ignore_index=True)
                elif source == "simard":
                    psf_eval_catalogs[i] = psf_eval_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'SClass': current_row['SClass'],
                    'z': current_row['z'],
                    'Scale': current_row['Scale'],
                    'Rhlg': current_row['Rhlg'],
                    'Rhlr': current_row['Rhlr'],
                    'Rchl,g': current_row['Rchl,g'],
                    'Rchl,r': current_row['Rchl,r'],
                    '(B/T)g': current_row['(B/T)g'],
                    'e(B/T)g': current_row['e(B/T)g'],
                    '(B/T)r': current_row['(B/T)r'],
                    'e(B/T)r': current_row['e(B/T)r'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                    psf_eval_catalogs[i] = psf_eval_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160'],
                    'B_T_m': current_row['B_T_m'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
            num_psf_eval += 1
            num_total += 1
            continue
        if num_gmn_train < gmn_train:
            for i in range(num_filters):
                # Save the image
                image_name = gmn_train_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                    gmn_train_catalogs[i] = gmn_train_catalogs[i].append({'object_id': obj_id, 'num_components': current_row['num_components'],
                    'sersic_idx_d': current_row['sersic_idx_d'],
                    'R_e_d': current_row['R_e_d'],
                    'axis_ratio_d': current_row['axis_ratio_d'],
                    'PA_d': current_row['PA_d'],
                    'flux_frac_d': current_row['flux_frac_d'],
                    'sersic_idx_b': current_row['sersic_idx_b'],
                    'R_e_b': current_row['R_e_b'],
                    'axis_ratio_b': current_row['axis_ratio_b'],
                    'PA_b': current_row['PA_b'],
                    'flux_frac_b': current_row['flux_frac_b'],
                    filter_strings[i] + '_total_flux': current_row['total_flux']}, ignore_index=True)
                elif source == "simard":
                    gmn_train_catalogs[i] = gmn_train_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'SClass': current_row['SClass'],
                    'z': current_row['z'],
                    'Scale': current_row['Scale'],
                    'Rhlg': current_row['Rhlg'],
                    'Rhlr': current_row['Rhlr'],
                    'Rchl,g': current_row['Rchl,g'],
                    'Rchl,r': current_row['Rchl,r'],
                    '(B/T)g': current_row['(B/T)g'],
                    'e(B/T)g': current_row['e(B/T)g'],
                    '(B/T)r': current_row['(B/T)r'],
                    'e(B/T)r': current_row['e(B/T)r'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True) 
                elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                    gmn_train_catalogs[i] = gmn_train_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160'],
                    'B_T_m': current_row['B_T_m'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)   
            num_gmn_train += 1
            num_total += 1
            continue
        if num_gmn_eval < gmn_eval:
            for i in range(num_filters):
                # Save the image
                image_name = gmn_eval_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                    gmn_eval_catalogs[i] = gmn_eval_catalogs[i].append({'object_id': obj_id, 'num_components': current_row['num_components'],
                    'sersic_idx_d': current_row['sersic_idx_d'],
                    'R_e_d': current_row['R_e_d'],
                    'axis_ratio_d': current_row['axis_ratio_d'],
                    'PA_d': current_row['PA_d'],
                    'flux_frac_d': current_row['flux_frac_d'],
                    'sersic_idx_b': current_row['sersic_idx_b'],
                    'R_e_b': current_row['R_e_b'],
                    'axis_ratio_b': current_row['axis_ratio_b'],
                    'PA_b': current_row['PA_b'],
                    'flux_frac_b': current_row['flux_frac_b'],
                    filter_strings[i] + '_total_flux': current_row['total_flux']}, ignore_index=True)
                elif source == "simard":
                    gmn_eval_catalogs[i] = gmn_eval_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'SClass': current_row['SClass'],
                    'z': current_row['z'],
                    'Scale': current_row['Scale'],
                    'Rhlg': current_row['Rhlg'],
                    'Rhlr': current_row['Rhlr'],
                    'Rchl,g': current_row['Rchl,g'],
                    'Rchl,r': current_row['Rchl,r'],
                    '(B/T)g': current_row['(B/T)g'],
                    'e(B/T)g': current_row['e(B/T)g'],
                    '(B/T)r': current_row['(B/T)r'],
                    'e(B/T)r': current_row['e(B/T)r'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)   
                elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                    gmn_eval_catalogs[i] = gmn_eval_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160'],
                    'B_T_m': current_row['B_T_m'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
            num_gmn_eval += 1
            num_total += 1
            continue
        if num_test < test:
            for i in range(num_filters):
                # Save the image
                image_name = test_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id, 'num_components': current_row['num_components'],
                    'sersic_idx_d': current_row['sersic_idx_d'],
                    'R_e_d': current_row['R_e_d'],
                    'axis_ratio_d': current_row['axis_ratio_d'],
                    'PA_d': current_row['PA_d'],
                    'flux_frac_d': current_row['flux_frac_d'],
                    'sersic_idx_b': current_row['sersic_idx_b'],
                    'R_e_b': current_row['R_e_b'],
                    'axis_ratio_b': current_row['axis_ratio_b'],
                    'PA_b': current_row['PA_b'],
                    'flux_frac_b': current_row['flux_frac_b'],
                    filter_strings[i] + '_total_flux': current_row['total_flux']}, ignore_index=True)
                elif source == "simard":
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'SClass': current_row['SClass'],
                    'z': current_row['z'],
                    'Scale': current_row['Scale'],
                    'Rhlg': current_row['Rhlg'],
                    'Rhlr': current_row['Rhlr'],
                    'Rchl,g': current_row['Rchl,g'],
                    'Rchl,r': current_row['Rchl,r'],
                    '(B/T)g': current_row['(B/T)g'],
                    'e(B/T)g': current_row['e(B/T)g'],
                    '(B/T)r': current_row['(B/T)r'],
                    'e(B/T)r': current_row['e(B/T)r'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "dimauro_0_0.5") or (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'photoz_best': current_row['photoz_best'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160'],
                    'B_T_m': current_row['B_T_m'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
            num_test += 1
            num_total += 1
            continue


    # At last, save the catalogs
    for i in range(num_filters):
        galaxy_per_filter = galaxy_main + filter_strings[i] + '-band/'
        gmn_train_catalogs[i].to_csv(galaxy_per_filter + 'gmn_train.csv', index=False)
        gmn_eval_catalogs[i].to_csv(galaxy_per_filter + 'gmn_eval.csv', index=False)
        psf_train_catalogs[i].to_csv(galaxy_per_filter + 'catalog_train.csv', index=False)
        psf_eval_catalogs[i].to_csv(galaxy_per_filter + 'catalog_eval.csv', index=False)
        test_catalogs[i].to_csv(galaxy_per_filter + 'catalog_test.csv', index=False)

    # Print out how many galaxies are selected
    print(str(num_total) + ' galaxies are selected in total:')
    print(str(num_gmn_train) + ' galaxies in train set for GaMorNet')
    print(str(num_gmn_eval) + ' galaxies in eval set for GaMorNet')
    print(str(num_psf_train) + ' galaxies in train set for PSFGAN')
    print(str(num_psf_eval) + ' galaxies in eval set for PSFGAN')
    print(str(num_test) + ' galaxies in common test set')
    print(str(num_resized) + ' galaxies have been resized for having different initial sizes')
    print(str(num_correctly_resized) + ' galaxies have been CORRECTLY resized')
    print(str(num_negative_flux) + ' galaxies are discarded for having negative flux(es) in at least one filter')


if __name__ == '__main__':
    data_split()
