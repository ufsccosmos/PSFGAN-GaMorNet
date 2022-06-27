# Please note, this script is forked from roou.py
# It has been modified to process hsc (instead of sdss and hubble) data
# Overall, it serves as the same role of roou.py
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
    parser.add_argument("--source", default="dimauro_0.5_1.0")
    # mode: 0 for testing, 1 for training, 2 for validation, 3 for gmn_train, 4 for gmn_eval
    parser.add_argument("--mode", default="0")
    parser.add_argument("--crop", default="0")
    parser.add_argument('--save_psf', default="0")
    parser.add_argument('--save_raw_input', default="1")
    args = parser.parse_args()

    outputs = args.outputs
    source = args.source
    mode = int(args.mode)
    cropsize = int(args.crop)
    save_psf = bool(int(args.save_psf))
    save_raw_input = bool(int(args.save_raw_input))

    # Conf parameters
    ratio_max = conf.max_contrast_ratio
    ratio_min = conf.min_contrast_ratio
    uniform_logspace = conf.uniform_logspace
    ratio_scaled_stddev = conf.contrast_ratio_scaled_stddev
    num_star_per_psf = conf.num_star_per_psf
    filters_string = conf.filters_
    num_filters = conf.num_filters

    # Data: from
    galaxy_inputs = []
    galaxy_catalog_paths = []
    galaxy_catalogs = []
    for f_index in range(num_filters):
        if mode == 0:  # Test set
            galaxy_input = '%s/fits_test' % conf.run_cases[f_index]
            galaxy_catalog_path = '%s/catalog_test.csv' % conf.run_cases[f_index]
        elif mode == 1:  # Train set
            galaxy_input = '%s/fits_train' % conf.run_cases[f_index]
            galaxy_catalog_path = '%s/catalog_train.csv' % conf.run_cases[f_index]
        elif mode == 2:  # Validation set
            galaxy_input = '%s/fits_eval' % conf.run_cases[f_index]
            galaxy_catalog_path = '%s/catalog_eval.csv' % conf.run_cases[f_index]
        elif mode == 3:  # Gmn_train set
            galaxy_input = '%s/gmn_train' % conf.run_cases[f_index]
            galaxy_catalog_path = '%s/gmn_train.csv' % conf.run_cases[f_index]
        elif mode == 4:  # Gmn_eval set
            galaxy_input = '%s/gmn_eval' % conf.run_cases[f_index]
            galaxy_catalog_path = '%s/gmn_eval.csv' % conf.run_cases[f_index]
        galaxy_catalog = pandas.read_csv(galaxy_catalog_path)

        galaxy_inputs.append(galaxy_input)
        galaxy_catalog_paths.append(galaxy_catalog_path)
        galaxy_catalogs.append(galaxy_catalog)
        print('Input files : %s' % galaxy_input)

    # Data: star
    star_inputs = []
    star_catalog_paths = []
    star_catalogs = []
    for f_index in range(num_filters):
        star_input = '%s/fits_star' % conf.run_cases[f_index]
        star_catalog_path = '%s/catalog_star.csv' % conf.run_cases[f_index]
        star_catalog = pandas.read_csv(star_catalog_path)

        star_inputs.append(star_input)
        star_catalog_paths.append(star_catalog_path)
        star_catalogs.append(star_catalog)

    # Data: to
    # Target folder for processed images
    test_folders = []
    train_folders = []
    eval_folders = []
    gmn_train_folders = []
    gmn_eval_folders = []
    for f_index in range(num_filters):
        test_folder = '%s/test' % outputs[f_index]
        train_folder = '%s/train' % outputs[f_index]
        eval_folder = '%s/eval' % outputs[f_index]
        gmn_train_folder = '%s/gmn_train' % outputs[f_index]
        gmn_eval_folder = '%s/gmn_eval' % outputs[f_index]
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        if not os.path.exists(gmn_train_folder):
            os.makedirs(gmn_train_folder)
        if not os.path.exists(gmn_eval_folder):
            os.makedirs(gmn_eval_folder)
        test_folders.append(test_folder)
        train_folders.append(train_folder)
        eval_folders.append(eval_folder)
        gmn_train_folders.append(train_folder)
        gmn_eval_folders.append(eval_folder)
    # Corresponding catalogs for target folders
    catalog_test_npy_inputs = []
    catalog_train_npy_inputs = []
    catalog_eval_npy_inputs = []
    catalog_gmn_train_npy_inputs = []
    catalog_gmn_eval_npy_inputs = []
    # When using simulated galaxies from GalSim, flags in these catalogs are different.
    # That's why we need to deal with them separately
    if (source == 'gal_sim_0_0.25') or (source == 'gal_sim_0.25_0.5') or (source == 'gal_sim_0.5_0.75') or (source == 'gal_sim_0.5_1.0'):
        for filter_string in filters_string:
            column_list=['object_id', 'num_components', 'sersic_idx_d', 'R_e_d', 'axis_ratio_d', 'PA_d', 'flux_frac_d',
                         'sersic_idx_b', 'R_e_b', 'axis_ratio_b', 'PA_b', 'flux_frac_b',
                         'galaxy_total_flux_' + filter_string,
                         'contrast_ratio_' + filter_string, 'R_HWHM_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)
            catalog_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_eval_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_eval_npy_input = pandas.DataFrame(columns=column_list)
            
            catalog_test_npy_inputs.append(catalog_test_npy_input)
            catalog_train_npy_inputs.append(catalog_train_npy_input)
            catalog_eval_npy_inputs.append(catalog_eval_npy_input)
            catalog_gmn_train_npy_inputs.append(catalog_gmn_train_npy_input)
            catalog_gmn_eval_npy_inputs.append(catalog_gmn_eval_npy_input)
    elif source == 'simard':
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'photoz_best', 'SClass', 'z', 'Scale',
                         'Rhlg', 'Rhlr', 'Rchl,g', 'Rchl,r', '(B/T)g', 'e(B/T)g', '(B/T)r', 'e(B/T)r',
                         'galaxy_total_flux_' + filter_string,
                         'contrast_ratio_' + filter_string, 'R_HWHM_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)
            catalog_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_eval_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_eval_npy_input = pandas.DataFrame(columns=column_list)
            
            catalog_test_npy_inputs.append(catalog_test_npy_input)
            catalog_train_npy_inputs.append(catalog_train_npy_input)
            catalog_eval_npy_inputs.append(catalog_eval_npy_input)
            catalog_gmn_train_npy_inputs.append(catalog_gmn_train_npy_input)
            catalog_gmn_eval_npy_inputs.append(catalog_gmn_eval_npy_input)
    elif (source == 'dimauro_0_0.5') or (source == 'dimauro_0.5_0.75') or (source == 'dimauro_0.5_1.0'):
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'photoz_best', 'RE_F606', 'RE_F814', 'RE_F125', 'RE_F160',
                         'N_F606', 'N_F814', 'N_F125', 'N_F160', 'B_T_m',
                         'galaxy_total_flux_' + filter_string,
                         'contrast_ratio_' + filter_string, 'R_HWHM_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)
            catalog_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_eval_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_train_npy_input = pandas.DataFrame(columns=column_list)
            catalog_gmn_eval_npy_input = pandas.DataFrame(columns=column_list)
            
            catalog_test_npy_inputs.append(catalog_test_npy_input)
            catalog_train_npy_inputs.append(catalog_train_npy_input)
            catalog_eval_npy_inputs.append(catalog_eval_npy_input)
            catalog_gmn_train_npy_inputs.append(catalog_gmn_train_npy_input)
            catalog_gmn_eval_npy_inputs.append(catalog_gmn_eval_npy_input)

    # Data: side product
    # For conditional inputs
    if save_raw_input:
        raw_test_folders = []
        raw_train_folders = []
        raw_eval_folders = []
        raw_gmn_train_folders = []
        raw_gmn_eval_folders = []
        for f_index in range(num_filters):
            # raw_test_folder is the path to the folder where the unstretched input
            # images (galaxy + AGN) are saved as .fits files.
            raw_test_folder = '%s/fits_test_condinput' % conf.run_cases[f_index]
            raw_train_folder = '%s/fits_train_condinput' % conf.run_cases[f_index]
            raw_eval_folder = '%s/fits_eval_condinput' % conf.run_cases[f_index]
            raw_gmn_train_folder = '%s/gmn_train_condinput' % conf.run_cases[f_index]
            raw_gmn_eval_folder = '%s/gmn_eval_condinput' % conf.run_cases[f_index]
            
            if not os.path.exists(raw_test_folder):
                os.makedirs(raw_test_folder)
            if not os.path.exists(raw_train_folder):
                os.makedirs(raw_train_folder)
            if not os.path.exists(raw_eval_folder):
                os.makedirs(raw_eval_folder)
            if not os.path.exists(raw_gmn_train_folder):
                os.makedirs(raw_gmn_train_folder)
            if not os.path.exists(raw_gmn_eval_folder):
                os.makedirs(raw_gmn_eval_folder)
            
            raw_test_folders.append(raw_test_folder)
            raw_train_folders.append(raw_train_folder)
            raw_eval_folders.append(raw_eval_folder)
            raw_gmn_train_folders.append(raw_gmn_train_folder)
            raw_gmn_eval_folders.append(raw_gmn_eval_folder)
    # For AGN PSFs
    if save_psf:
        psf_test_folders = []
        psf_train_folders = []
        psf_eval_folders = []
        psf_gmn_train_folders = []
        psf_gmn_eval_folders = []
        for f_index in range(num_filters):
            psf_test_folder = '%s/fits_test_psf' % conf.run_cases[f_index]
            psf_train_folder = '%s/fits_train_psf' % conf.run_cases[f_index]
            psf_eval_folder = '%s/fits_eval_psf' % conf.run_cases[f_index]
            psf_gmn_train_folder = '%s/gmn_train_psf' % conf.run_cases[f_index]
            psf_gmn_eval_folder = '%s/gmn_eval_psf' % conf.run_cases[f_index]
            
            if not os.path.exists(psf_test_folder):
                os.makedirs(psf_test_folder)
            if not os.path.exists(psf_train_folder):
                os.makedirs(psf_train_folder)
            if not os.path.exists(psf_eval_folder):
                os.makedirs(psf_eval_folder)
            if not os.path.exists(psf_gmn_train_folder):
                os.makedirs(psf_gmn_train_folder)
            if not os.path.exists(psf_gmn_eval_folder):
                os.makedirs(psf_gmn_eval_folder)
            
            psf_test_folders.append(psf_test_folder)
            psf_train_folders.append(psf_train_folder)
            psf_eval_folders.append(psf_eval_folder)
            psf_gmn_train_folders.append(psf_gmn_train_folder)
            psf_gmn_eval_folders.append(psf_gmn_eval_folder)

    # Prepare to read data
    files_mb = []
    for f_index in range(num_filters):
        fits_path = '%s/*-%s.fits' % (galaxy_inputs[f_index], filters_string[f_index])
        files = glob.glob(fits_path)
        files_mb.append(files)
    galaxy_input_size = len(files_mb[0])

    # Main loop
    # Iterate over all files in the directory 'galaxy_input' and create the conditional
    # input (galaxy + PS) for them.
    pixel_max = 0
    not_found = 0
    catalog_contrast_ratio = pandas.DataFrame(columns=['object_id', 'contrast_ratio_mean'])
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

            if (source == 'gal_sim_0_0.25') or (source == 'gal_sim_0.25_0.5') or (source == 'gal_sim_0.5_0.75') or (source == 'gal_sim_0.5_1.0') or (source == 'simard') or (source == 'dimauro_0_0.5') or (source == 'dimauro_0.5_0.75') or (source == 'dimauro_0.5_1.0'):
                flux = obj_line[filters_string[f_index] + '_total_flux'].item()
                if flux < 0:
                    print(filters_string[f_index] + '_total_flux' + ' value in catalog is negative!')
                    continue

            # To focus on the PSFGAN-GaMorNet framework, we use independently sampled contrast ratios across filters for illustration.
            # Readers are encouraged to insert their own module to sample a set of realistic contrast ratios across filters,
            # using real AGN SEDs as a reference.
            # ------------------------------------
            # Sample the contrast ratios from the distribution specified in the file
            # config.py
            # ------------------------------------
            # For each image, we randomly select a mean for contrast ratio before sampling according to contrast_ratio_scaled_stddev:
            # For the first filter, we select r_mean and save it
            if f_index == 0:
                if uniform_logspace:
                    r_mean_exponent = random.uniform(np.log10(ratio_min),
                                                np.log10(ratio_max))
                    r_mean = 10 ** r_mean_exponent
                else:
                    r_mean = random.uniform(ratio_min, ratio_max)
                catalog_contrast_ratio = catalog_contrast_ratio.append({'object_id': image_id,
                                                                        'contrast_ratio_mean': r_mean}
                                                                       , ignore_index=True)
            # For all filters, we sample r according to the mean and the standard deviation
            contrast_line = catalog_contrast_ratio.loc[catalog_contrast_ratio['object_id'] == image_id]
            r_mean = contrast_line['contrast_ratio_mean'].item()
            r_stddev = ratio_scaled_stddev
            if uniform_logspace:
                r_mean_exponent = np.log10(r_mean)
                r_exponent = norm.rvs(loc=r_mean_exponent, scale=r_stddev)
                r = 10 ** r_exponent
            else:
                r = norm.rvs(loc=r_mean, scale=r_stddev)
            # ------------------------------------

            # Calculate R_HWHM using its radial profile
            r_profile = photometry.radial_profile(data_r, (data_r.shape[0] // 2, data_r.shape[1] // 2))
            R_HWHM_found = 0
            for radius in range(len(r_profile)):
                if r_profile[radius] <= (r_profile[0] / 2) and R_HWHM_found == 0:
                    R_HWHM = radius
                    R_HWHM_found = 1

            # Get the conditional input (and probably the model psf)
            desired_psf_flux = r * flux

            if save_psf:
                data_PSF, PSF = photometry.add_hsc_PSF(data_r, desired_psf_flux, star_input=star_inputs[f_index], star_catalog=star_catalogs[f_index],
                                                       filter_string=filters_string[f_index], num_star_per_psf=num_star_per_psf,
                                                       save_psf=save_psf)
            else:
                data_PSF = photometry.add_hsc_PSF(data_r, desired_psf_flux, star_input=star_inputs[f_index], star_catalog=star_catalogs[f_index],
                                                  filter_string=filters_string[f_index], num_star_per_psf=num_star_per_psf,
                                                  save_psf=save_psf)


            # Create a row in the corresponding catalog
            # Note we have multiple cases depending on the source of our data
            if (source == 'gal_sim_0_0.25') or (source == 'gal_sim_0.25_0.5') or (source == 'gal_sim_0.5_0.75') or (source == 'gal_sim_0.5_1.0'):
                if mode == 0:  # Test set
                    catalog_per_index = catalog_test_npy_inputs[f_index]
                elif mode == 1:  # Train set
                    catalog_per_index = catalog_train_npy_inputs[f_index]
                elif mode == 2:  # Validation set
                    catalog_per_index = catalog_eval_npy_inputs[f_index]
                elif mode == 3:  # Gmn_train set
                    catalog_per_index = catalog_gmn_train_npy_inputs[f_index]
                elif mode == 4:  # Gmn_eval set
                    catalog_per_index = catalog_gmn_eval_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,            
                                                              'num_components': obj_line['num_components'].item(),
                                                              'sersic_idx_d': obj_line['sersic_idx_d'].item(),
                                                              'R_e_d': obj_line['R_e_d'].item(),
                                                              'axis_ratio_d': obj_line['axis_ratio_d'].item(),
                                                              'PA_d': obj_line['PA_d'].item(),
                                                              'flux_frac_d': obj_line['flux_frac_d'].item(),
                                                              'sersic_idx_b': obj_line['sersic_idx_b'].item(),
                                                              'R_e_b': obj_line['R_e_b'].item(),
                                                              'axis_ratio_b': obj_line['axis_ratio_b'].item(),
                                                              'PA_b': obj_line['PA_b'].item(),
                                                              'flux_frac_b': obj_line['flux_frac_b'].item(),
                                                              'PA_b': obj_line['PA_b'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item(),
                                                              'contrast_ratio_' + filters_string[f_index]: r,
                                                              'R_HWHM_' + filters_string[f_index]: R_HWHM}
                                                              , ignore_index=True)
                if mode == 0:  # Test set
                    catalog_test_npy_inputs[f_index] = catalog_per_index
                elif mode == 1:  # Train set
                    catalog_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 2:  # Validation set
                    catalog_eval_npy_inputs[f_index] = catalog_per_index
                elif mode == 3:  # Gmn_train set
                    catalog_gmn_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 4:  # Gmn_eval set
                    catalog_gmn_eval_npy_inputs[f_index] = catalog_per_index

            elif source == 'simard':
                if mode == 0:  # Test set
                    catalog_per_index = catalog_test_npy_inputs[f_index]
                elif mode == 1:  # Train set
                    catalog_per_index = catalog_train_npy_inputs[f_index]
                elif mode == 2:  # Validation set
                    catalog_per_index = catalog_eval_npy_inputs[f_index]
                elif mode == 3:  # Gmn_train set
                    catalog_per_index = catalog_gmn_train_npy_inputs[f_index]
                elif mode == 4:  # Gmn_eval set
                    catalog_per_index = catalog_gmn_eval_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'photoz_best': obj_line['photoz_best'].item(),
                                                              'SClass': obj_line['SClass'].item(),
                                                              'z': obj_line['z'].item(),
                                                              'Scale': obj_line['Scale'].item(),
                                                              'Rhlg': obj_line['Rhlg'].item(),
                                                              'Rhlr': obj_line['Rhlr'].item(),
                                                              'Rchl,g': obj_line['Rchl,g'].item(),
                                                              'Rchl,r': obj_line['Rchl,r'].item(),
                                                              '(B/T)g': obj_line['(B/T)g'].item(),
                                                              'e(B/T)g': obj_line['e(B/T)g'].item(),
                                                              '(B/T)r': obj_line['(B/T)r'].item(),
                                                              'e(B/T)r': obj_line['e(B/T)r'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item(),
                                                              'contrast_ratio_' + filters_string[f_index]: r,
                                                              'R_HWHM_' + filters_string[f_index]: R_HWHM},
                                                              ignore_index=True)
                if mode == 0:  # Test set
                    catalog_test_npy_inputs[f_index] = catalog_per_index
                elif mode == 1:  # Train set
                    catalog_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 2:  # Validation set
                    catalog_eval_npy_inputs[f_index] = catalog_per_index
                elif mode == 3:  # Gmn_train set
                    catalog_gmn_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 4:  # Gmn_eval set
                    catalog_gmn_eval_npy_inputs[f_index] = catalog_per_index
                                                                                                 
            elif (source == 'dimauro_0_0.5') or (source == 'dimauro_0.5_0.75') or (source == 'dimauro_0.5_1.0'):
                if mode == 0:  # Test set
                    catalog_per_index = catalog_test_npy_inputs[f_index]
                elif mode == 1:  # Train set
                    catalog_per_index = catalog_train_npy_inputs[f_index]
                elif mode == 2:  # Validation set
                    catalog_per_index = catalog_eval_npy_inputs[f_index]
                elif mode == 3:  # Gmn_train set
                    catalog_per_index = catalog_gmn_train_npy_inputs[f_index]
                elif mode == 4:  # Gmn_eval set
                    catalog_per_index = catalog_gmn_eval_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,            
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'photoz_best': obj_line['photoz_best'].item(),
                                                              'RE_F606': obj_line['RE_F606'].item(),
                                                              'RE_F814': obj_line['RE_F814'].item(),
                                                              'RE_F125': obj_line['RE_F125'].item(),
                                                              'RE_F160': obj_line['RE_F160'].item(),
                                                              'N_F606': obj_line['N_F606'].item(),
                                                              'N_F814': obj_line['N_F814'].item(),
                                                              'N_F125': obj_line['N_F125'].item(),
                                                              'N_F160': obj_line['N_F160'].item(),
                                                              'B_T_m': obj_line['B_T_m'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item(),
                                                              'contrast_ratio_' + filters_string[f_index]: r,
                                                              'R_HWHM_' + filters_string[f_index]: R_HWHM}
                                                              , ignore_index=True)                                           
                if mode == 0:  # Test set
                    catalog_test_npy_inputs[f_index] = catalog_per_index
                elif mode == 1:  # Train set
                    catalog_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 2:  # Validation set
                    catalog_eval_npy_inputs[f_index] = catalog_per_index
                elif mode == 3:  # Gmn_train set
                    catalog_gmn_train_npy_inputs[f_index] = catalog_per_index
                elif mode == 4:  # Gmn_eval set
                    catalog_gmn_eval_npy_inputs[f_index] = catalog_per_index


            # Saving the model PSF before stretching
            if save_psf:
                if mode == 0:  # Test set
                    psf_name = '%s/%s-%s.fits' % (psf_test_folders[f_index], image_id, filters_string[f_index])
                elif mode == 1:  # Train set
                    psf_name = '%s/%s-%s.fits' % (psf_train_folders[f_index], image_id, filters_string[f_index])
                elif mode == 2:  # Validation set
                    psf_name = '%s/%s-%s.fits' % (psf_eval_folders[f_index], image_id, filters_string[f_index])
                elif mode == 3:  # Gmn_train set
                    psf_name = '%s/%s-%s.fits' % (psf_gmn_train_folders[f_index], image_id, filters_string[f_index])
                elif mode == 4:  # Gmn_eval set
                    psf_name = '%s/%s-%s.fits' % (psf_gmn_eval_folders[f_index], image_id, filters_string[f_index])
                # Overwrite if files already exist.
                hdu = fits.PrimaryHDU(PSF)
                hdu.writeto(psf_name, overwrite=True)

            # Saving the "raw" data+PSF before stretching
            if save_raw_input:
                if mode == 0:  # Test set
                    raw_name = '%s/%s-%s.fits' % (raw_test_folders[f_index], image_id, filters_string[f_index])
                elif mode == 1:  # Train set
                    raw_name = '%s/%s-%s.fits' % (raw_train_folders[f_index], image_id, filters_string[f_index])
                elif mode == 2:  # Validation set
                    raw_name = '%s/%s-%s.fits' % (raw_eval_folders[f_index], image_id, filters_string[f_index])
                elif mode == 3:  # Gmn_train set
                    raw_name = '%s/%s-%s.fits' % (raw_gmn_train_folders[f_index], image_id, filters_string[f_index])
                elif mode == 4:  # Gmn_eval set
                    raw_name = '%s/%s-%s.fits' % (raw_gmn_eval_folders[f_index], image_id, filters_string[f_index])
                # Overwrite if files already exist.
                hdu = fits.PrimaryHDU(data_PSF)
                hdu.writeto(raw_name, overwrite=True)

            # (Optional) crop all images we care (inherit from 'roou.py')
            if (cropsize > 0):
                figure_original = np.ones((2 * cropsize, 2 * cropsize, 1))
                figure_original[:, :, 0] = photometry.crop(data_r, cropsize)
                figure_with_PSF = np.ones((2 * cropsize, 2 * cropsize, 1))
                figure_with_PSF[:, :, 0] = photometry.crop(data_PSF, cropsize)
            else:
                figure_original = np.ones((data_r.shape[0], data_r.shape[1], 1))
                figure_original[:, :, 0] = data_r
                figure_with_PSF = np.ones((data_r.shape[0], data_r.shape[1], 1))
                figure_with_PSF[:, :, 0] = data_PSF

            # Preprocessing
            figure_original = conf.stretch(figure_original)
            figure_with_PSF = conf.stretch(figure_with_PSF)

            # output result to pix2pix format
            figure_combined = np.zeros((figure_original.shape[0],
                                        figure_original.shape[1] * 2, 1))
            figure_combined[:, :figure_original.shape[1], :] = figure_original[:, :, :]
            figure_combined[:, figure_original.shape[1]:2*figure_original.shape[1], :] = figure_with_PSF[:, :, :]

            # Save the numpy inputs
            if mode == 0:  # Testing set
                mat_path = '%s/test/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            elif mode == 1:  # Training set
                mat_path = '%s/train/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            elif mode == 2:  # Validation set
                mat_path = '%s/eval/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            elif mode == 3:  # Gmn_train set
                mat_path = '%s/gmn_train/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            elif mode == 4:  # Gmn_eval set
                mat_path = '%s/gmn_eval/%s-%s.npy' % (outputs[f_index], image_id, filters_string[f_index])
            np.save(mat_path, figure_combined)

            if np.max(photometry.crop(data_PSF, 30)) > pixel_max:
                pixel_max = np.max(photometry.crop(data_PSF, 30))

            # Print for every 200 image processed
            image_processed += 1
            if (image_processed % 200) == 0:
                print('%s/%s images processed in filter %s' % (image_processed, galaxy_input_size, filters_string[f_index]))

        # Save the catalog
        if mode == 0:  # Testing set
            catalog_test_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_test_npy_input.csv', index=False)
        elif mode == 1:  # Training set
            catalog_train_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_train_npy_input.csv', index=False)
        elif mode == 2:  # Validation set
            catalog_eval_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_eval_npy_input.csv', index=False)
        elif mode == 3:  # Gmn_train set
            catalog_gmn_train_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_gmn_train_npy_input.csv', index=False)
        elif mode == 4:  # Gmn_eval set
            catalog_gmn_eval_npy_inputs[f_index].to_csv(outputs[f_index] + '/catalog_gmn_eval_npy_input.csv', index=False)

    print('Maximum pixel value inside a box of 60x60 pixels around the center:')
    print(pixel_max)
    print("%s images have not been used because there were no corresponding" \
          " objects in the catalog") % not_found

if __name__ == '__main__':
    roou()
