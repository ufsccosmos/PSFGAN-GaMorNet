# Please note, this script is forked from roouhsc.py
# It has been modified to process real AGN
# Overall, it serves a similar role of roouhsc.py, except:
# 1. No artificial AGN PS is added (AGN host galaxy already has that).
# 2. Images will still be normalized. 
# 3. Both positions of "figure_combined" will be the image of AGN host galaxy. 
# (Originally, the first half will be the image of original galaxy while the second half be galaxy + artificial AGN PS)
# 4. Only deal with the test set -- the only set for real AGN host galaxy data
# Author: Chuan Tian, Department of Physics, Yale University


import argparse
import glob
import os
import random
import numpy as np
import pandas
from astropy.io import fits
from scipy.stats import norm

import photometry
from config import Config as conf

parser = argparse.ArgumentParser()


def roou():
    random.seed(42)
    parser.add_argument("--outputs", default=conf.data_paths)
    parser.add_argument("--source", default="liu")
    parser.add_argument("--crop", default="0")
    args = parser.parse_args()

    outputs = args.outputs
    source = args.source
    cropsize = int(args.crop)

    # Conf parameters
    filters_string = conf.filters_
    num_filters = conf.num_filters

    # Data: from
    galaxy_inputs = []
    galaxy_catalog_paths = []
    galaxy_catalogs = []
    for f_index in range(num_filters):
        galaxy_input = '%s/fits_test' % conf.run_cases[f_index]
        galaxy_catalog_path = '%s/catalog_test.csv' % conf.run_cases[f_index]
        galaxy_catalog = pandas.read_csv(galaxy_catalog_path)

        galaxy_inputs.append(galaxy_input)
        galaxy_catalog_paths.append(galaxy_catalog_path)
        galaxy_catalogs.append(galaxy_catalog)
        print('Input files : %s' % galaxy_input)


    # Data: to
    # Target folder for processed images
    test_folders = []
    for f_index in range(num_filters):
        test_folder = '%s/test' % outputs[f_index]
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        test_folders.append(test_folder)
    # Corresponding catalogs for target folders
    catalog_test_npy_inputs = []
    # When using simulated galaxies from GalSim, flags in the catalogs are different.
    # That's why we need to dealt with them separately
    if source == 'nair':
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous', 'z', 'gmag', 'rmag', 'gMag',
                         'Lg', 'Rp', 'Rp50', 'Rp90', 'Area', 'b/a', 'Seeg', 'ng', 'nr', 'R50n', 'R90n',
                         'TType', 'flag',
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)
    elif (source == 'gabor_0.3_0.5') or (source == 'gabor_0.5_0.75'):
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous', 'z',
                         'mh', 'mh_gal', 'rp', 'rh_gal', 'nh', 'nh_gal', 'mp', 'rn0', 'Flag', 'Class',
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)
    elif (source == 'povic_0_0.5') or (source == 'povic_0.5_0.75'):
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                         'FS', 'FH', 'FVH', 'FTot', 'FVH2', 'FTot2', 'Rc', 'zph', 'BMAG', 'Stellarity', 'p1', 'p2', 
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)
    elif (source == 'stemo_0.2_0.5') or (source == 'stemo_0.5_1.0'):
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                         'Z', 'SPECZ', 'PHOTOZ', 'L_bol_sed_md', 'L_x_md', 'L_bol_x_md', 'M_star_md', 'SFR_md', 'Nh_md', 'SFR_norm_md', 
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)
    elif source == 'liu':
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                         'z',
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)


    # Prepare to read data
    files_mb = []
    for f_index in range(num_filters):
        fits_path = '%s/*-%s.fits' % (galaxy_inputs[f_index], filters_string[f_index])
        files = glob.glob(fits_path)
        files_mb.append(files)
    galaxy_input_size = len(files_mb[0])

    # Main loop
    # Iterate over all files in the directory 'galaxy_input' 
    pixel_max = 0
    not_found = 0
    for f_index in range(num_filters):
        image_processed = 0
        files = files_mb[f_index]
        for i in files:
            image_id = os.path.basename(i).replace('-' + filters_string[f_index] + '.fits', '')
            # print('\n')
            # print(image_id)

            obj_line = galaxy_catalogs[f_index].loc[galaxy_catalogs[f_index]['object_id'] == int(image_id)]
            if obj_line.empty:
                not_found = not_found + 1
                print('Galaxy not found in filter %s' % filters_string[f_index])
                continue

            f = i

            rfits = fits.open(f)
            data_r = rfits[0].data
            rfits.close()

            if (source == 'nair') or (source == 'gabor_0.3_0.5') or (source == 'gabor_0.5_0.75') or (source == 'povic_0_0.5') or (source == 'povic_0.5_0.75') or (source == 'stemo_0.2_0.5') or (source == 'stemo_0.5_1.0') or (source == 'liu'):
                flux = obj_line[filters_string[f_index] + '_total_flux'].item()
                if flux < 0:
                    print(filters_string[f_index] + '_total_flux' + ' value in catalog is negative!')
                    continue

            # Create a row in the corresponding catalog
            # Note we have several cases depending on the source of our data
            if source == 'nair':
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'z': obj_line['z'].item(),
                                                              'gmag': obj_line['gmag'].item(),
                                                              'rmag': obj_line['rmag'].item(),
                                                              'gMag': obj_line['gMag'].item(),
                                                              'Lg': obj_line['Lg'].item(),
                                                              'Rp': obj_line['Rp'].item(),
                                                              'Rp50': obj_line['Rp50'].item(),
                                                              'Rp90': obj_line['Rp90'].item(),
                                                              'Area': obj_line['Area'].item(),
                                                              'b/a': obj_line['b/a'].item(),
                                                              'Seeg': obj_line['Seeg'].item(),
                                                              'ng': obj_line['ng'].item(),
                                                              'nr': obj_line['nr'].item(),
                                                              'R50n': obj_line['R50n'].item(),
                                                              'R90n': obj_line['R90n'].item(),
                                                              'TType': obj_line['TType'].item(),
                                                              'flag': obj_line['flag'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index
            elif (source == 'gabor_0.3_0.5') or (source == 'gabor_0.5_0.75'):
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'z': obj_line['z'].item(),
                                                              'mh': obj_line['mh'].item(),
                                                              'mh_gal': obj_line['mh_gal'].item(),
                                                              'rp': obj_line['rp'].item(),
                                                              'rh_gal': obj_line['rh_gal'].item(),
                                                              'nh': obj_line['nh'].item(),
                                                              'nh_gal': obj_line['nh_gal'].item(),
                                                              'mp': obj_line['mp'].item(),
                                                              'rn0': obj_line['rn0'].item(),
                                                              'Flag': obj_line['Flag'].item(),
                                                              'Class': obj_line['Class'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index
            elif (source == 'povic_0_0.5') or (source == 'povic_0.5_0.75'):
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'FS': obj_line['FS'].item(),
                                                              'FH': obj_line['FH'].item(),
                                                              'FVH': obj_line['FVH'].item(),
                                                              'FTot': obj_line['FTot'].item(),
                                                              'FVH2': obj_line['FVH2'].item(),
                                                              'FTot2': obj_line['FTot2'].item(),
                                                              'Rc': obj_line['Rc'].item(),
                                                              'zph': obj_line['zph'].item(),
                                                              'BMAG': obj_line['BMAG'].item(),
                                                              'Stellarity': obj_line['Stellarity'].item(),
                                                              'p1': obj_line['p1'].item(),
                                                              'p2': obj_line['p2'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index
            elif (source == 'stemo_0.2_0.5') or (source == 'stemo_0.5_1.0'):
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'Z': obj_line['Z'].item(),
                                                              'SPECZ': obj_line['SPECZ'].item(),
                                                              'PHOTOZ': obj_line['PHOTOZ'].item(),
                                                              'L_bol_sed_md': obj_line['L_bol_sed_md'].item(),
                                                              'L_x_md': obj_line['L_x_md'].item(),
                                                              'L_bol_x_md': obj_line['L_bol_x_md'].item(),
                                                              'M_star_md': obj_line['M_star_md'].item(),
                                                              'SFR_md': obj_line['SFR_md'].item(),
                                                              'Nh_md': obj_line['Nh_md'].item(),
                                                              'SFR_norm_md': obj_line['SFR_norm_md'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index    
            elif source == 'liu':
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'z': obj_line['z'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index                                                     
                         
            # (Optional) crop all images we care (inherit from 'roou.py')
            if (cropsize > 0):
                figure_agn = np.ones((2 * cropsize, 2 * cropsize, 1))
                figure_agn_agn[:, :, 0] = photometry.crop(data_r, cropsize)
            else:
                figure_agn = np.ones((data_r.shape[0], data_r.shape[1], 1))
                figure_agn[:, :, 0] = data_r

            # Preprocessing
            figure_agn = conf.stretch(figure_agn)

            # output result to pix2pix format
            # Note for AGN host galaxy, both "original" and "with PSF" figures are the same AGN host galaxy figure
            figure_combined = np.zeros((figure_agn.shape[0],
                                        figure_agn.shape[1] * 2, 1))
            figure_combined[:, :figure_agn.shape[1], :] = figure_agn[:, :, :]
            figure_combined[:, figure_agn.shape[1]:2*figure_agn.shape[1], :] = figure_agn[:, :, :]

            # Save the numpy inputs
            mat_path = '%s/test/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            np.save(mat_path, figure_combined)

            if np.max(photometry.crop(data_r, 30)) > pixel_max:
                pixel_max = np.max(photometry.crop(data_r, 30))

            # Print for every 100 image processed
            image_processed += 1
            if (image_processed % 100) == 0:
                print('%s/%s images processed in filter %s' % (image_processed, galaxy_input_size, filters_string[f_index]))

        # Save the catalog
        catalog_test_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_test_npy_input.csv', index=False)

    print('Maximum pixel value inside a box of 60x60 pixels around the center:')
    print(pixel_max)
    print("%s images have not been used because there were no corresponding" \
          " objects in the catalog") % not_found

if __name__ == '__main__':
    roou()
