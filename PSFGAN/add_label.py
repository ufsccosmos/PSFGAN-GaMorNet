# This Python script adds classification labels to simulated HSC galaxies based on their parameters
# Input: a catalog with necessary parameters (must contain the following flags:
# -num_components: either 1 or 2
# -sersic_idx_d (if num_components == 1)
# -flux_frac_b (if num_components == 2)
# Output the same catalog with three extra columns: is_disk, is_indeterminate and is_bulge
# Modification on Mar. 2, 2021: Make it consistent with multi-band real HSC galaxies
# Modification on Aug. 24, 2021: Make it consistent with real AGN host galaxies

import argparse
import os
import sys
import glob
import pandas
import numpy as np
import random
from astropy.io import fits

parser = argparse.ArgumentParser()

def add_label():
    parser.add_argument("--input_path", default='')
    parser.add_argument("--source", default='')
    parser.add_argument("--use_label", default='')
    args = parser.parse_args()

    input_path = args.input_path
    source = str(args.source)
    use_label = str(args.use_label)

    catalog = pandas.read_csv(glob.glob(input_path)[0])
    length_catalog = len(catalog)

    # Check if the catalog contains necessary columns
    if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
        if not ('num_components' in catalog):
            sys.exit('catalog must contain the following column: num_components')
        if not ('flux_frac_b' in catalog):
            sys.exit('catalog must contain the following column: flux_frac_b')
        if not ('sersic_idx_d' in catalog):
            sys.exit('catalog must contain the following column: sersic_idx_d')
    elif source == "simard":
        if not ('(B/T)g' in catalog):
            sys.exit('catalog must contain the following column: (B/T)g') 
    elif source == "dimauro_0_0.5":
        if not ('B_T_m' in catalog):
            sys.exit('catalog must contain the following column: B_T_m') 
        if not ('N_F606' in catalog):
            sys.exit('catalog must contain the following column: N_F606') 
    elif (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
        if not ('B_T_m' in catalog):
            sys.exit('catalog must contain the following column: B_T_m') 
        if not ('N_F814' in catalog):
            sys.exit('catalog must contain the following column: N_F814') 
    elif source == "nair":
        if not ('TType' in catalog):
            sys.exit('catalog must contain the following column: TType') 
    elif (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75"):
        if not ('nh_gal' in catalog):
            sys.exit('catalog must contain the following column: nh_gal') 
    elif (source == "povic_0_0.5") or (source == "povic_0.5_0.75"):
        if not ('Stellarity' in catalog):
            sys.exit('catalog must contain the following column: Stellarity') 
        if not ('p1' in catalog):
            sys.exit('catalog must contain the following column: p1') 
        if not ('p2' in catalog):
            sys.exit('catalog must contain the following column: p2') 

    # Proceed if it contains all necessary columns
    catalog['is_disk'] = [0]*length_catalog
    catalog['is_indeterminate'] = [0]*length_catalog
    catalog['is_bulge'] = [0]*length_catalog

    # Main loop
    row_num_list = list(range(length_catalog))
    for row_num in row_num_list:
        if (source == "gal_sim_0_0.25") or (source == "gal_sim_0.25_0.5") or (source == "gal_sim_0.5_0.75") or (source == "gal_sim_0.5_1.0"):
            num_components = catalog.at[row_num, 'num_components']
            flux_frac_b = catalog.at[row_num, 'flux_frac_b']
            sersic_idx_d = catalog.at[row_num, 'sersic_idx_d']

            if num_components == 1:
                if sersic_idx_d < 2.0:
                    catalog.at[row_num, 'is_disk'] = 1
                elif sersic_idx_d > 2.5:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 2.0 <= sersic_idx_d <= 2.5
                    catalog.at[row_num, 'is_indeterminate'] = 1
            elif num_components == 2:
                if flux_frac_b < 0.45:
                    catalog.at[row_num, 'is_disk'] = 1
                elif flux_frac_b > 0.55:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 0.45 <= flux_frac_b <= 0.55
                    catalog.at[row_num, 'is_indeterminate'] = 1
        elif source == "simard":
            BT_g = catalog.at[row_num, '(B/T)g']
            
            if BT_g < 0.45:
                catalog.at[row_num, 'is_disk'] = 1
            elif BT_g > 0.55:
                catalog.at[row_num, 'is_bulge'] = 1
            else: # 0.45 <= BT_g <= 0.55
                catalog.at[row_num, 'is_indeterminate'] = 1
        elif source == "dimauro_0_0.5":
            B_T_m = catalog.at[row_num, 'B_T_m']
            N_F606 = catalog.at[row_num, 'N_F606']
            
            if use_label == 'bt':
                if B_T_m < 0.45:
                    catalog.at[row_num, 'is_disk'] = 1
                elif B_T_m > 0.55:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 0.45 <= B_T_m <= 0.55
                    catalog.at[row_num, 'is_indeterminate'] = 1
            elif use_label == 'n':
                if N_F606 < 2.0:
                    catalog.at[row_num, 'is_disk'] = 1
                elif N_F606 > 2.5:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 2.0 <= N_F606 <= 2.5
                    catalog.at[row_num, 'is_indeterminate'] = 1
        elif (source == "dimauro_0.5_0.75") or (source == "dimauro_0.5_1.0"):
            B_T_m = catalog.at[row_num, 'B_T_m']
            N_F814 = catalog.at[row_num, 'N_F814']
            
            if use_label == 'bt':
                if B_T_m < 0.45:
                    catalog.at[row_num, 'is_disk'] = 1
                elif B_T_m > 0.55:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 0.45 <= B_T_m <= 0.55
                    catalog.at[row_num, 'is_indeterminate'] = 1
            elif use_label == 'n':
                if N_F814 < 2.0:
                    catalog.at[row_num, 'is_disk'] = 1
                elif N_F814 > 2.5:
                    catalog.at[row_num, 'is_bulge'] = 1
                else: # 2.0 <= N_F814 <= 2.5
                    catalog.at[row_num, 'is_indeterminate'] = 1
        elif source == "nair":
            TType = catalog.at[row_num, 'TType']
            
            if (TType > -1) and (TType < 15):
                catalog.at[row_num, 'is_disk'] = 1
            elif TType == -5:
                catalog.at[row_num, 'is_bulge'] = 1
            else: # TType == -3, TType == -2, or TType == 99
                catalog.at[row_num, 'is_indeterminate'] = 1
        elif (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75"):
            nh_gal = catalog.at[row_num, 'nh_gal']
            
            if nh_gal < 2.0:
                catalog.at[row_num, 'is_disk'] = 1
            elif nh_gal > 2.5:
                catalog.at[row_num, 'is_bulge'] = 1
            else: # 2.0 <= nh_gal <= 2.5
                catalog.at[row_num, 'is_indeterminate'] = 1
        elif (source == "povic_0_0.5") or (source == "povic_0.5_0.75"):
            Stellarity = catalog.at[row_num, 'Stellarity']
            p1 = catalog.at[row_num, 'p1']
            p2 = catalog.at[row_num, 'p2']
            
            if Stellarity < 0.9:
                if (p1 > 0.75) or (p1 == 0.75):
                    catalog.at[row_num, 'is_bulge'] = 1
                elif p1 < 0.5: # p2 > 0.5
                    catalog.at[row_num, 'is_disk'] = 1
                else: # 0.5 <= p1 < 0.75
                    catalog.at[row_num, 'is_indeterminate'] = 1
            else: # Stellarity >= 0.9
                catalog.at[row_num, 'is_indeterminate'] = 1
            
        

    # Save the modified catalog
    if use_label == '':  
        catalog.to_csv(input_path.replace('.csv', '_labeled' + '.csv'), index=False)
    else: 
        catalog.to_csv(input_path.replace('.csv', '_labeled_' + use_label + '.csv'), index=False)
    

if __name__ == '__main__':
    add_label()
