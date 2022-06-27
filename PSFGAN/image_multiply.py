# Note: use to create rotation/reflection copies of real galaxies
# Modified from "data_split.py"

import argparse
import os
import glob
import pandas
import numpy as np
import random
from astropy.io import fits

# Paths
core_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/'
original = core_path + 'dimauro_0.5_1.0_original/'
target = core_path + 'dimauro_0.5_1.0/'

# Other parameters
# Order doesn't matter (e.g. ['g', 'r'] is the same as ['r', 'g'])
filter_strings = ['g', 'r', 'i', 'z', 'y']

## The desired image shape. Images of other shapes will not pass the selection (thus be filtered out)
#desired_shape = [239, 239]

parser = argparse.ArgumentParser()

def image_multiply():
    # Make the split predictable
    np.random.seed(42)
    parser.add_argument("--m_rot", default=4)
    parser.add_argument("--m_ref", default=2)
    args = parser.parse_args()

    m_rot = int(args.m_rot)
    m_ref = int(args.m_ref)
    m_tot = m_rot * m_ref
    num_filters = len(filter_strings)

    num_original = 0
    num_target = 0

    # Input and output locations
    original_folders = []
    original_catalogs = []
    
    target_folders = []
    target_catalogs = []

    for filter_string in filter_strings:
        original_per_filter = original + filter_string + '-band/'
        target_per_filter = target + filter_string + '-band/'

        original_folder = glob.glob(original_per_filter + 'raw_data/images/')[0]
        original_catalog = pandas.read_csv(glob.glob(original_per_filter + 'raw_data/*.csv')[0])

        target_folder = target_per_filter + 'raw_data/images/'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        column_list = ['object_id', 'ra', 'dec', 'tract', 'photoz_best', 
                       'g_cmodel_flux', 'r_cmodel_flux', 'i_cmodel_flux', 'z_cmodel_flux', 'y_cmodel_flux',
                       'FIELD', 'B_T_m', 'RE_F606', 'RE_F814', 'RE_F125', 'RE_F160',
                       'N_F606', 'N_F814', 'N_F125', 'N_F160']
        target_catalog = pandas.DataFrame(columns=column_list)

        original_folders.append(original_folder)
        original_catalogs.append(original_catalog)
        target_folders.append(target_folder)
        target_catalogs.append(target_catalog)

    # Main loop
    # Start the loop by iterating over the row number based on the first catalog from original_catalogs
    row_num_list = list(range(2, len(original_catalogs[0]) + 2))

    for row_num in row_num_list:
        id = int(row_num)

        for i in range(num_filters):
            # Read the images
            fits_path = '%s/%s-cutout-*.fits' % (original_folders[i], id)
            file = glob.glob(fits_path)[0]
            image = fits.getdata(file)

            # Create rotation/reflection copies of real galaxies
            image_list = []
            image_list.append(image)
            if m_rot == 2:
                image_list.append(np.rot90(image, 2))
            elif m_rot == 4:
                image_list.append(np.rot90(image, 1))
                image_list.append(np.rot90(image, 2))
                image_list.append(np.rot90(image, 3))

            if m_ref == 2:
                image_list += list(np.fliplr(image) for image in image_list)

            # Save images
            for j in range(m_tot):
                name = target_folders[i] + str(2 + m_tot*(id-2) + j) + '-cutout-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(image_list[j])
                hdu.writeto(name, overwrite=True)

            # Create m_tot number of (identical) rows in the new catalog
            current_row = original_catalogs[i].iloc[row_num - 2]
            for j in range(m_tot):
                target_catalogs[i] = target_catalogs[i].append({'object_id': current_row['object_id'],
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'tract': current_row['tract'],
                    'photoz_best': current_row['photoz_best'],
                    'g_cmodel_flux': current_row['g_cmodel_flux'],
                    'r_cmodel_flux': current_row['r_cmodel_flux'],
                    'i_cmodel_flux': current_row['i_cmodel_flux'],
                    'z_cmodel_flux': current_row['z_cmodel_flux'],
                    'y_cmodel_flux': current_row['y_cmodel_flux'],
                    'FIELD': current_row['FIELD'],
                    'B_T_m': current_row['B_T_m'],
                    'RE_F606': current_row['RE_F606'],
                    'RE_F814': current_row['RE_F814'],
                    'RE_F125': current_row['RE_F125'],
                    'RE_F160': current_row['RE_F160'],
                    'N_F606': current_row['N_F606'],
                    'N_F814': current_row['N_F814'],
                    'N_F125': current_row['N_F125'],
                    'N_F160': current_row['N_F160']}, ignore_index=True)

        num_original += 1
        num_target += m_tot


    # At last, save the catalogs
    for i in range(num_filters):
        target_per_filter = target + filter_strings[i] + '-band/'
        target_catalogs[i].to_csv(target_per_filter + 'raw_data/dimauro_0.5_1.0_multiplied.csv', index=False)

    # Print out how many galaxies are selected
    print('m_rotation = ' + str(m_rot))
    print('m_reflection = ' + str(m_ref))
    print(str(num_original) + ' galaxies are in the original dataset')
    print(str(num_target) + ' galaxies are in the target dataset')

if __name__ == '__main__':
    image_multiply()