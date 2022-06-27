# Note: use to preprocess real AGN host galaxies 
# No actual data split is done since all images will be put into the test set
# This script is written to keep accordance with "data_split.py"
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
galaxy_main = core_path + 'liu/'

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
    parser.add_argument("--test", default=1359)
    parser.add_argument("--shuffle", default="1")
    # Identify source of the raw data. This will determine the names of columns in catalogs being created
    # Options: "sim_hsc_0_0.25", "simard_cross_hsc"...(more to be added)
    parser.add_argument("--source", default="liu")
    args = parser.parse_args()

    test = int(args.test)
    shuffle = bool(int(args.shuffle))
    source = str(args.source)
    num_filters = len(filter_strings)

    num_total = 0
    num_test = 0
    num_resized = 0
    num_correctly_resized = 0
    num_negative_flux = 0

    # Input and output locations
    hsc_folders = []
    hsc_catalogs = []

    test_folders = []
    test_catalogs = []

    for filter_string in filter_strings:
        galaxy_per_filter = galaxy_main + filter_string + '-band/'

        hsc_folder = glob.glob(galaxy_per_filter + 'raw_data/images/')[0]
        hsc_catalog = pandas.read_csv(glob.glob(galaxy_per_filter + 'raw_data/*.csv')[0])

        test_folder = galaxy_per_filter + 'fits_test/'
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        if source == "nair":
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous', 'z', 'gmag', 'rmag', 'gMag',
                          'Lg', 'Rp', 'Rp50', 'Rp90', 'Area', 'b/a', 'Seeg', 'ng', 'nr', 'R50n', 'R90n',
                          'TType', 'flag',
                          filter_string + '_total_flux']
        elif (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75"):
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous', 'z',
                          'mh', 'mh_gal', 'rp', 'rh_gal', 'nh', 'nh_gal', 'mp', 'rn0', 'Flag', 'Class',
                          filter_string + '_total_flux'] 
        elif (source == "povic_0_0.5") or (source == "povic_0.5_0.75"):
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                          'FS', 'FH', 'FVH', 'FTot', 'FVH2', 'FTot2', 'Rc', 'zph', 'BMAG', 'Stellarity', 'p1', 'p2', 
                          filter_string + '_total_flux']  
        elif (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0"):
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                          'Z', 'SPECZ', 'PHOTOZ', 'L_bol_sed_md', 'L_x_md', 'L_bol_x_md', 'M_star_md', 'SFR_md', 'Nh_md', 'SFR_norm_md', 
                          filter_string + '_total_flux']   
        elif source == "liu":
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                          'z', 
                          filter_string + '_total_flux']                   
        test_catalog = pandas.DataFrame(columns=column_list)
        

        hsc_folders.append(hsc_folder)
        hsc_catalogs.append(hsc_catalog)
        test_folders.append(test_folder)
        test_catalogs.append(test_catalog)


    # Main loop
    # Start the loop by iterating over the row number based on the first catalog from hsc_catalogs
    row_num_list = list(range(2, len(hsc_catalogs[0]) + 2))
    
    
    if shuffle:
        np.random.shuffle(row_num_list)
            
    
    for row_num in row_num_list:
        if (source == "nair") or (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75") or (source == "povic_0_0.5") or (source == "povic_0.5_0.75") or (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0") or (source == "liu"):
            obj_id = int(row_num)

        # Read the images
        images = []
        for i in range(num_filters):
            if (source == "nair") or (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75") or (source == "povic_0_0.5") or (source == "povic_0.5_0.75") or (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0") or (source == "liu"):
                fits_path = '%s/%s-cutout-*.fits' % (hsc_folders[i], obj_id)
            file = glob.glob(fits_path)[0]
            image = fits.getdata(file)
            images.append(image)

        # Check whether the flux is positive in each filter
        # If not, quit the loop
        positive_flux_booleans = []
        for i in range(num_filters):
            current_row = hsc_catalogs[i].iloc[row_num-2]
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
        if num_test < test:
            for i in range(num_filters):
                # Save the image
                image_name = test_folders[i] + str(obj_id) + '-' + filter_strings[i] + '.fits'
                hdu = fits.PrimaryHDU(images[i])
                hdu.writeto(image_name, overwrite=True)
                # Also, create a row for this image in the new catalog
                current_row = hsc_catalogs[i].iloc[row_num-2]
                if source == "nair":
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'z': current_row['z'],
                    'gmag': current_row['gmag'],
                    'rmag': current_row['rmag'],
                    'gMag': current_row['gMag'],
                    'Lg': current_row['Lg'],
                    'Rp': current_row['Rp'],
                    'Rp50': current_row['Rp50'],
                    'Rp90': current_row['Rp90'],
                    'Area': current_row['Area'],
                    'b/a': current_row['b/a'],
                    'Seeg': current_row['Seeg'],
                    'ng': current_row['ng'],
                    'nr': current_row['nr'],
                    'R50n': current_row['R50n'],
                    'R90n': current_row['R90n'],
                    'TType': current_row['TType'],
                    'flag': current_row['flag'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75"):
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'z': current_row['z'],
                    'mh': current_row['mh'],
                    'mh_gal': current_row['mh_gal'],
                    'rp': current_row['rp'],
                    'rh_gal': current_row['rh_gal'],
                    'nh': current_row['nh'],
                    'nh_gal': current_row['nh_gal'],
                    'mp': current_row['mp'],
                    'rn0': current_row['rn0'],
                    'Flag': current_row['Flag'],
                    'Class': current_row['Class'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "povic_0_0.5") or (source == "povic_0.5_0.75"):
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'FS': current_row['FS'],
                    'FH': current_row['FH'],
                    'FVH': current_row['FVH'],
                    'FTot': current_row['FTot'],
                    'FVH2': current_row['FVH2'],
                    'FTot2': current_row['FTot2'],
                    'Rc': current_row['Rc'],
                    'zph': current_row['zph'],
                    'BMAG': current_row['BMAG'],
                    'Stellarity': current_row['Stellarity'],
                    'p1': current_row['p1'],
                    'p2': current_row['p2'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0"):
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'Z': current_row['Z'],
                    'SPECZ': current_row['SPECZ'],
                    'PHOTOZ': current_row['PHOTOZ'],
                    'L_bol_sed_md': current_row['L_bol_sed_md'],
                    'L_x_md': current_row['L_x_md'],
                    'L_bol_x_md': current_row['L_bol_x_md'],
                    'M_star_md': current_row['M_star_md'],
                    'SFR_md': current_row['SFR_md'],
                    'Nh_md': current_row['Nh_md'],
                    'SFR_norm_md': current_row['SFR_norm_md'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
                elif source == "liu":
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'z': current_row['z'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
            num_test += 1
            num_total += 1
            continue

    # At last, save the catalogs
    for i in range(num_filters):
        galaxy_per_filter = galaxy_main + filter_strings[i] + '-band/'
        test_catalogs[i].to_csv(galaxy_per_filter + 'catalog_test.csv', index=False)

    # Print out how many galaxies are selected
    print(str(num_total) + ' galaxies are selected in total:')
    print(str(num_test) + ' galaxies in common test set')
    print(str(num_resized) + ' galaxies have been resized for having different initial sizes')
    print(str(num_correctly_resized) + ' galaxies have been CORRECTLY resized')
    print(str(num_negative_flux) + ' galaxies are discarded for having negative flux(es) in at least one filter')


if __name__ == '__main__':
    data_split()