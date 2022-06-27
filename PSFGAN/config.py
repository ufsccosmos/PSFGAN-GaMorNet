import os
from normalizing import Normalizer
class Config:
    ## data selection parameters
    redshift = 'liu'
    filters_ = ['g', 'r', 'i', 'z', 'y']
    num_filters = len(filters_)

    ## normalization parameters
    stretch_type = 'asinh'
    scale_factor = 50

    ## model parameters
    # learning_rate_original = 0.00009
    # learning_rate_gal_sim_0_0.25 = 0.00005
    # learning_rate_gal_sim_0.25_0.5 = 0.000015
    # learning_rate_gal_sim_0.5_0.75 = 0.00002
    # learning_rate_gal_sim_0.5_1.0 = 0.00002
    # learning_rate_simard = 0.00009
    # learning_rate_dimauro_0_0.5 = 0.00002
    # learning_rate_dimauro_0.5_0.75 = 0.000005
    # learning_rate_dimauro_0.5_1.0 = 0.000005
    learning_rate = 0.00009
    attention_parameter = 0.05
    # if you are not going to train from the very beginning (or only testing),
    # change this path to the existing model path (.cpkt file)
    #model_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/z_0.05_0.8_mb_07302020/i-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/model/model.ckpt'
    #model_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/gal_sim_4/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_6e-05/model/model.ckpt'
    #model_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/gal_sim_0.25_0.5/r-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_2.5e-05/model/model.ckpt'
    #model_path = '/gpfs/loomis/project/urry/ct564/HSC/PSFGAN/simard/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/model/model.ckpt'
    model_path = '/gpfs/loomis/scratch60/urry/ct564/simard/g-band/asinh_50/lintrain_classic_PSFGAN_0.05/lr_9e-05/model/model.ckpt'
    beta1 = 0.5
    L1_lambda = 100
    sum_lambda = 0

    ## directory tree setup
    # 1/ Dataset dependant
    # Working directory where the project is stored
    # Default to where this file is stored
    core_path = os.path.dirname(os.path.abspath(__file__))
    #ext = ''  # Custom extension to differentiate setups

    run_cases = []
    stretch_setups =[]
    sub_configs = []

    output_paths = []
    result_paths = []
    data_paths = []
    save_paths = []
    for filter_ in filters_:
        run_case = "%s/%s/%s-band" % (core_path, redshift, filter_)
        # 2/ Precomputation dependant
        stretch_setup = '%s/%s_%s' % (run_case, stretch_type, scale_factor)
        # 3/ PSFGAN model dependant
        sub_config = '%s/lintrain_classic_PSFGAN_%s/lr_%s' % (stretch_setup,
                                                              attention_parameter,
                                                              learning_rate)

        output_path = '%s/PSFGAN_output' % sub_config
        result_path = output_path
        data_path = "%s/npy_input" % stretch_setup
        save_path = "%s/model" % sub_config

        run_cases.append(run_case)
        stretch_setups.append(stretch_setup)
        sub_configs.append(sub_config)

        output_paths.append(output_path)
        result_paths.append(result_path)
        data_paths.append(data_path)
        save_paths.append(save_path)


    ## Datasets dependant value
    # This has been precomputed for SDSS datasets
    #if '0.01' in run_case:
    #    pixel_max_value = 41100
    #elif '0.05' in run_case:
    #    pixel_max_value = 6140
    #elif '0.1' in run_case:
    #    pixel_max_value = 1450
    #elif '0.2' in run_case:
    #    pixel_max_value = 1657
    ## Simulated Galaxies
    if redshift == 'gal_sim_0_0.25':
        pixel_max_value = 25000
    if redshift == 'gal_sim_0.25_0.5':
        pixel_max_value = 5000
    if redshift == 'gal_sim_0.5_0.75':
        pixel_max_value = 10000
    if redshift == 'gal_sim_0.5_1.0':
        pixel_max_value = 10000
    ## Real Galaxies
    if redshift == 'simard':
        pixel_max_value = 45000
    if redshift == 'dimauro_0_0.5':
        pixel_max_value = 10000
    if redshift == 'dimauro_0.5_0.75':
        pixel_max_value = 1000
    if redshift == 'dimauro_0.5_1.0':
        pixel_max_value = 1000
    ## Validation Galaxies (with labels)
    if redshift == 'nair':
        pixel_max_value = 45000
    if redshift == 'gabor_0.3_0.5':
        pixel_max_value = 10000
    if redshift == 'gabor_0.5_0.75':
        pixel_max_value = 1000
    if redshift == 'povic_0_0.5':
        pixel_max_value = 10000
    if redshift == 'povic_0.5_0.75':
        pixel_max_value = 1000
    ## Validation Galaxies (without labels)
    if redshift == 'liu':
        pixel_max_value = 45000
    if redshift == 'stemo_0.2_0.5':
        pixel_max_value = 10000
    if redshift == 'stemo_0.5_1.0':
        pixel_max_value = 1000
    #else:
        ## Default
        #pixel_max_value = 2000
    pixel_min_value = -0.1

    ## Normalizer, used to scale images
    normalizer = Normalizer(stretch_type,
                                 scale_factor,
                                 pixel_min_value,
                                 pixel_max_value)
    stretch = normalizer.stretch
    unstretch = normalizer.unstretch
    ## contrast distribution for the added PSF
    max_contrast_ratio = 3.981
    min_contrast_ratio = 0.1
    ## For gal_sim_*
    # max_contrast_ratio = 3.981
    # min_contrast_ratio = 0.1
    ## In old experiments:
    # max_contrast_ratio = 3.162 #10^(0.5)
    # min_contrast_ratio = 0.316 #10^(-0.5)
    uniform_logspace = True
    ## Number of stars to create each added PSF
    num_star_per_psf = 50
    ## Standard deviation when sampling a contrast ratio per image per filter in normal (log) space when uniform_logspace = False (True)
    contrast_ratio_scaled_stddev = 0.05

    ## training parameters
    # specify which GPU should be used in CUDA_VISIBLE_DEVICES
    use_gpu = 0
    start_epoch = 0
    save_per_epoch = 2
    max_epoch = 20
    test_epoch = 20
    img_size = 239
    train_size = 239
    img_channel = num_filters
    conv_channel_base = 64
    #use_gpu = 0
    #start_epoch = 0
    #save_per_epoch = 50
    #max_epoch = 50
    #img_size = 424
    #train_size = 424
    #img_channel = 1
    #conv_channel_base = 64




