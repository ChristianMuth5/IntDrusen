import sys

from datagen import gen_data
from trainer import train
from latent_discovery import find_latent
import logging
import os


def run_manager(runs, random_seed):
    # Init the logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('mainlog.txt')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    for run in runs:

        if run is None:
            logger.info("Encountered early stop")
            break
        # Set values that are equal for all
        if "random_seed" not in run:
            run["random_seed"] = random_seed

        logger.info(f"Run: {run}")

        # Generate data
        gen_data(run, logger)

        # Train model
        G, folder_out = train(run, logger)
        if G is None:  # If training failed, continue to next run
            logger.info("Training failed, G is None, no need to search for directions")
            continue

        # Find directions
        find_latent(run, G, folder_out, logger)

    logger.info("Finished")
    file_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    return


def setup_runs(generate_all_paths_for_int_drusen=False):

    # data_source: Currently supports Duke and Bonn, describes which dataset to use, and the names of the
    # calculated lines with layer annotation

    # Valid method values:
    # noise, for no noise removal
    # GaussX, for Gauss filter with X being Sigma
    # FFDNet, noise removal with FFDNet
    # mnist, use mnist dataset
    # median, for median filterin
    # BiM, for Bilaterial filtering plus Median filtering
    # overview: not fully tested and was used for creating overview images for data exploration

    # remove_below_line: Only useful when not using rectified data, it removes everything 10 pixel below the BM line

    # image_size: Only tested with 128

    # minimum_drusen_height: Minimum height for calculating the drusen, minimum is 3

    # rectify: Whether to use rectangular data and rectify the BM line

    # watershed: Whether to use something similar to watershed for segmenting singular drusen rather than
    # using binary erosion. The value corresponds to the minimum distance between drusen peaks, use
    # -1 for binary erosion.
    # Only tested for rectify = True

    # single_drusen: Whether to have only a single druse in the image or multiple, blurs out the rest

    # scaling: if not included there will be no scaling of the infos, otherwise if =1, use histogram matching,
    # if larger than 1 scale by the factor, will spread the values over [0,1]
    dataset_noise_s = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                       "minimum_drusen_height": 5, "rectify": True, "watershed": 10, "single_drusen": True}
    dataset_noise_s_10 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                          "minimum_drusen_height": 5, "rectify": True, "watershed": 10, "single_drusen": True}
    dataset_noise_s_20 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                          "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}
    dataset_noise_s_200 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                           "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True,
                           "scaling": 200}
    dataset_noise_s_1 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                         "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True,
                         "scaling": 1}
    dataset_noise_m = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                       "minimum_drusen_height": 5, "rectify": True, "watershed": 10, "single_drusen": False}
    dataset_noise_m_10 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                          "minimum_drusen_height": 5, "rectify": True, "watershed": 10, "single_drusen": False}
    dataset_noise_m_20 = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                          "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": False}
    dataset_ffdnet = {"data_source": "Bonn", "method": "FFDNet", "remove_below_line": False, "image_size": 128,
                      "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}
    dataset_gauss = {"data_source": "Bonn", "method": "Gauss2", "remove_below_line": False, "image_size": 128,
                     "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}
    dataset_bim = {"data_source": "Bonn", "method": "BiM", "remove_below_line": False, "image_size": 128,
                   "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}
    dataset_overview = {"data_source": "Bonn", "method": "overview", "remove_below_line": False, "image_size": 128,
                        "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}
    # dataset5 = {"data_source": "Duke", "method": "median", "remove_below_line": False, "image_size": 128,
    #            "minimum_drusen_height": 5, "rectify": True, "watershed": 20, "single_drusen": True}

    # Valid train_method values:
    # GAN, classic simple GAN architecture
    # ComplexGAN, has more layers, more trainable parameters
    # StyleGAN, use StyleGAN

    # num_epochs: How many epochs to train GAN for

    # latent size: size of the latent vector

    # augment_data: whether to augment the data, currently only vertical flips, as others could create unwanted features

    # mix: mix real and fake batch during training

    # loss: which loss function to use for the GAN, supports currently bce and mse

    # batch_size: batch size to be used

    # conditions: use conditional informations given as list of indices, with height=0, width=1, volume=2
    # leave list empty for non-conditional, has been changed to only be volume

    # weight_samples: optional, if included and True, sample weighter by class, because of this currently only
    # works for conditions==[2], so set to volume only
    model_gan_10_10 = {"train_method": "GAN", "num_epochs": 10, "latent_size": 10, "augment_data": True, "mix": False,
                       "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_10_20 = {"train_method": "GAN", "num_epochs": 10, "latent_size": 20, "augment_data": True, "mix": False,
                       "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_10_20_cond = {"train_method": "GAN", "num_epochs": 10, "latent_size": 20, "augment_data": True,
                            "mix": False,
                            "loss": "bce", "batch_size": 128, "conditions": [2]}
    model_gan_50_20 = {"train_method": "GAN", "num_epochs": 50, "latent_size": 20, "augment_data": True, "mix": False,
                       "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_100_10 = {"train_method": "GAN", "num_epochs": 100, "latent_size": 10, "augment_data": True, "mix": False,
                        "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_100_20_cond = {"train_method": "GAN", "num_epochs": 100, "latent_size": 20, "augment_data": True,
                             "mix": False,
                             "loss": "bce", "batch_size": 128, "conditions": [2]}
    model_gan_200_10 = {"train_method": "GAN", "num_epochs": 200, "latent_size": 10, "augment_data": True, "mix": False,
                        "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_200_20 = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_200_20_mix = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                            "mix": True,
                            "loss": "bce", "batch_size": 128, "conditions": []}
    model_gan_50_20_cond = {"train_method": "GAN", "num_epochs": 50, "latent_size": 20, "augment_data": True,
                            "mix": False,
                            "loss": "bce", "batch_size": 128, "conditions": [2], "weight_samples": True}
    model_gan_200_20_cond = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                             "mix": False,
                             "loss": "bce", "batch_size": 128, "conditions": [2], "weight_samples": True}
    model_gan_50_20_ws = {"train_method": "GAN", "num_epochs": 50, "latent_size": 20, "augment_data": True,
                          "mix": False,
                          "loss": "bce", "batch_size": 128, "conditions": [], "weight_samples": True}
    model_gan_200_20_ws = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                           "mix": False,
                           "loss": "bce", "batch_size": 128, "conditions": [], "weight_samples": True}
    model_gan_100_20_ws = {"train_method": "GAN", "num_epochs": 100, "latent_size": 20, "augment_data": True,
                           "mix": False,
                           "loss": "bce", "batch_size": 128, "conditions": [], "weight_samples": True}
    model_gan_100_20 = {"train_method": "GAN", "num_epochs": 100, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce", "batch_size": 128, "conditions": [], "weight_samples": False}
    model_gan_200_20_mix_cond = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                                 "mix": True,
                                 "loss": "bce", "batch_size": 128, "conditions": [2]}

    model_cgan_10_20 = {"train_method": "ComplexGAN", "num_epochs": 10, "latent_size": 20, "augment_data": True,
                        "mix": False,
                        "loss": "bce", "batch_size": 128, "conditions": []}
    model_cgan_200_20 = {"train_method": "ComplexGAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                         "mix": False,
                         "loss": "bce", "batch_size": 128, "conditions": []}
    model_cgan_200_20_mix = {"train_method": "ComplexGAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                             "mix": True,
                             "loss": "bce", "batch_size": 128, "conditions": []}

    model_stylegan3_05 = {"train_method": "StyleGAN3", "num_epochs": 5000, "augment_data": True,
                          "StyleGAN_Settings": "--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=0.5 --cbase=16384",
                          "latent_size": 30}
    model_stylegan3_02 = {"train_method": "StyleGAN3", "num_epochs": 500, "augment_data": True,
                          "StyleGAN_Settings": "--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=0.5 --cbase=16384",
                          "latent_size": 10}
    model_stylegan2 = {"train_method": "StyleGAN2", "num_epochs": 500, "augment_data": True,
                       "StyleGAN_Settings": "--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384",
                       "latent_size": 10}
    model_stylegan2_20 = {"train_method": "StyleGAN2", "num_epochs": 20, "augment_data": True,
                          "StyleGAN_Settings": "--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384",
                          "latent_size": 10}
    model_stylegan2_1000 = {"train_method": "StyleGAN2", "num_epochs": 1000, "augment_data": True,
                            "StyleGAN_Settings": "--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384",
                            "latent_size": 10}
    model_stylegan2_1000_ws = {"train_method": "StyleGAN2", "num_epochs": 1000, "augment_data": True,
                               "StyleGAN_Settings": "--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384",
                               "latent_size": 10, "weight_samples": True}
    model_stylegan2_2000 = {"train_method": "StyleGAN2", "num_epochs": 2000, "augment_data": True,
                            "StyleGAN_Settings": "--cfg=stylegan2 --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384",
                            "latent_size": 10}

    # Valid values for latent discovery
    # method: linear, warp, pde
    # latent_steps, around 100-200k seems to be fine
    # deformator_type: used for linear method
    # directions: amount of directions
    latent_linear_1k = {"method": "linear", "latent_steps": 1000, "deformator_type": "ORTHO", "directions": 20,
                        "num_samples": 20}
    latent_linear_100 = {"method": "linear", "latent_steps": 50, "deformator_type": "ORTHO", "directions": 20,
                         "num_samples": 20}
    latent_linear_50_w = {"method": "linear", "latent_steps": 50, "deformator_type": "ORTHO", "directions": 20,
                           "num_samples": 20, "shift_in_w_space": True}
    latent_linear_100_w = {"method": "linear", "latent_steps": 100, "deformator_type": "ORTHO", "directions": 20,
                           "num_samples": 20, "shift_in_w_space": True}
    latent_linear_1k_w = {"method": "linear", "latent_steps": 1000, "deformator_type": "ORTHO", "directions": 20,
                          "num_samples": 20, "shift_in_w_space": True}
    latent_linear_20k = {"method": "linear", "latent_steps": 20000, "deformator_type": "ORTHO", "directions": 20,
                        "num_samples": 20}
    latent_linear_20k_w = {"method": "linear", "latent_steps": 20000, "deformator_type": "ORTHO", "directions": 20,
                          "num_samples": 20,
                          "shift_in_w_space": True}
    latent_linear_3k = {"method": "linear", "latent_steps": 3000, "deformator_type": "ORTHO", "directions": 20,
                        "num_samples": 20}
    latent_linear_3k_w = {"method": "linear", "latent_steps": 3000, "deformator_type": "ORTHO", "directions": 20,
                          "num_samples": 20,
                          "shift_in_w_space": True}
    latent_linear_200_w = {"method": "linear", "latent_steps": 200, "deformator_type": "ORTHO", "directions": 20,
                           "num_samples": 20, "shift_in_w_space": True}
    latent_linear_10k = {"method": "linear", "latent_steps": 10000, "deformator_type": "ORTHO", "directions": 20,
                         "num_samples": 20}
    latent_linear_100k = {"method": "linear", "latent_steps": 100000, "deformator_type": "ORTHO", "directions": 20,
                          "num_samples": 20}
    latent_linear_50k = {"method": "linear", "latent_steps": 50000, "deformator_type": "ORTHO", "directions": 20,
                         "num_samples": 20}
    latent_warp_100 = {"method": "warp", "latent_steps": 100, "directions": 20, "num_samples": 20}
    latent_warp_100_w = {"method": "warp", "latent_steps": 100, "directions": 20, "num_samples": 20,
                         "shift_in_w_space": True}
    latent_warp_1k = {"method": "warp", "latent_steps": 1000, "directions": 20, "num_samples": 20}
    latent_warp_50k = {"method": "warp", "latent_steps": 50000, "directions": 20, "num_samples": 20}
    latent_warp_50k_w = {"method": "warp", "latent_steps": 50000, "directions": 20, "num_samples": 20,
                         "shift_in_w_space": True}
    latent_warp_1k_w = {"method": "warp", "latent_steps": 1000, "directions": 20, "num_samples": 20,
                        "shift_in_w_space": True}
    latent_warp_5k = {"method": "warp", "latent_steps": 5000, "directions": 20, "num_samples": 20}
    latent_warp_100k_20 = {"method": "warp", "latent_steps": 100000, "directions": 20, "num_samples": 20}
    latent_warp_200k_20 = {"method": "warp", "latent_steps": 200000, "directions": 20, "num_samples": 20}
    latent_warp_200k_40 = {"method": "warp", "latent_steps": 200000, "directions": 40, "num_samples": 20}
    latent_warp_200k_60 = {"method": "warp", "latent_steps": 200000, "directions": 60, "num_samples": 20}
    latent_pde_50_20 = {"method": "pde", "latent_steps": 50, "directions": 20, "num_samples": 20}
    latent_pde_50_20_w = {"method": "pde", "latent_steps": 50, "directions": 20, "num_samples": 20,
                          "shift_in_w_space": True}
    latent_pde_200_20 = {"method": "pde", "latent_steps": 200, "directions": 20, "num_samples": 20}
    latent_pde_20k_20 = {"method": "pde", "latent_steps": 20000, "directions": 20, "num_samples": 20}
    latent_pde_50k_20 = {"method": "pde", "latent_steps": 50000, "directions": 20, "num_samples": 20}
    latent_pde_5k_20 = {"method": "pde", "latent_steps": 5000, "directions": 20, "num_samples": 20}
    latent_pde_5k_20_w = {"method": "pde", "latent_steps": 5000, "directions": 20, "num_samples": 20,
                          "shift_in_w_space": True}
    latent_pde_20k_20_w = {"method": "pde", "latent_steps": 20000, "directions": 20, "num_samples": 20,
                          "shift_in_w_space": True}
    latent_pde_20k_20 = {"method": "pde", "latent_steps": 20000, "directions": 20, "num_samples": 20}
    latent_pde_1k_20_w = {"method": "pde", "latent_steps": 1000, "directions": 20, "num_samples": 20,
                          "shift_in_w_space": True}
    latent_pde_1k_20 = {"method": "pde", "latent_steps": 1000, "directions": 20, "num_samples": 20}

    runs = []

    # because they have already been trained, might as well look at them
    #runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": None,
    #             "drusen": [[8, 34, 51], [5, 152, 209], [40, 49, 75], [1, 16, 65, 19, 111], [28, 181]]})
    #runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_m_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k_w,
                 "paths": []})

    #runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": None,
    #             "drusen": [[98, 101], [147], [3, 22], [18, 94], [213]]})
    #runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_s_10, "model": model_stylegan2_2000, "latent_method": latent_warp_50k_w,
                 "paths": []})

    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20, "latent_method": None,
    #             "drusen": [[38, 152, 157], [25, 109, 143, 203], [101, 144], [50, 200], [87, 198]]})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20, "latent_method": latent_linear_50k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20, "latent_method": latent_warp_100k_20,
    #             "paths": []})

    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20_ws, "latent_method": None,
    #             "drusen": [[3, 14, 24], [2, 7, 19, 58, 61], [0, 10, 27, 31, 45], [4, 44, 66], [8, 36, 62]]})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20_ws, "latent_method": latent_linear_50k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20_ws, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_gan_100_20_ws, "latent_method": latent_warp_100k_20,
    #             "paths": []})

    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": None,
    #             "drusen": [[2, 16, 114], [134, 198, 227], [46, 9], [30, 262], [42, 64]]})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k_w,
                 "paths": []})

    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": None,
    #             "drusen": [[2, 355], [147, 201, 224], [32, 67, 106], [161, 399], [48, 56]]})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_m_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k_w,
                 "paths": []})

    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20, "latent_method": None,
    #             "drusen": [[20, 227, 196], [37, 205], [108, 130, 176, 185], [85], [217, 246, 219]]})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20, "latent_method": latent_linear_50k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20, "latent_method": latent_warp_100k_20,
    #             "paths": []})

    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20_ws, "latent_method": None,
    #             "drusen": [[23, 67, 103], [32, 45, 86, 204, 222], [21, 72], [77, 114, 175], [79, 232]]})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20_ws, "latent_method": latent_linear_50k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20_ws, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_gan_100_20_ws, "latent_method": latent_warp_100k_20,
    #             "paths": []})

    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": None,
    #             "drusen": [[217, 553], [571, 711], [57, 84, 212], [237, 220], [207, 377]]})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000, "latent_method": latent_warp_50k_w,
                 "paths": []})

    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": None,
    #             "drusen": [[300, 414], [260, 281], [120, 231], [36, 62], [9, 293, 393]]})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_linear_20k,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_linear_20k_w,
    #             "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20_w})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_pde_20k_20_w,
                 "paths": []})
    #runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k,
    #             "paths": []})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k_w})
    runs.append({"dataset": dataset_noise_s_20, "model": model_stylegan2_1000_ws, "latent_method": latent_warp_50k_w,
                 "paths": []})

    # runs.append({"dataset": dataset_noise_s, "model": model_stylegan2_1000, "latent_method": latent_warp_100_w})  # about 0.5s per step
    # runs.append({"dataset": dataset_noise_s, "model": model_stylegan2_1000, "latent_method": latent_pde_50_20_w})  # about 1.7s per step
    # runs.append({"dataset": dataset_noise_s, "model": model_stylegan2_1000, "latent_method": latent_linear_200_w})  # about 1.5s per step

    runs.append(None)  # Early stop / End

    if generate_all_paths_for_int_drusen:
        for i in range(len(runs)):
            if runs[i] is not None and runs[i]["latent_method"] is not None:
                runs[i]["paths"] = []

    return runs

def main():
    # Change to a fixed value for reproducibility, -1 for random
    random_seed = -1
    #runs = setup_runs(generate_all_paths_for_int_drusen=True)
    runs = setup_runs(generate_all_paths_for_int_drusen=False)

    run_manager(runs, random_seed)

    #run_manager(setup_runs(True), random_seed)  # Generate all path images for all interesting drusen


if __name__ == '__main__':
    # Warped GAN Space seems to work here only on 1 device
    #os.environ["CUDA_HOME"] = "C:\\Users\\cmuth\\anaconda3\\pkgs\\cuda-nvcc-12.2.91-0\\bin"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
