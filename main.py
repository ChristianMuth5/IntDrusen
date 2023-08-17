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
            logger.info("Training failed, G is None")
            continue
        # Find directions
        find_latent(run, G, folder_out, logger)

    logger.info("Finished")
    file_handler.close()
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
    return


def main():
    # Change to a fixed value for reproducibility, -1 for random
    random_seed = -1

    # Valid dataset values:
    # noise, for no noise removal
    # GaussX, for Gauss filter with X being Sigma
    # FFDNet, noise removal with FFDNet
    # mnist, use mnist dataset
    # median, for median filterin
    # BiM, for Bilaterial filtering plus Median filtering
    dataset_noise = {"data_source": "Bonn", "method": "noise", "remove_below_line": False, "image_size": 128,
                     "minimum_drusen_height": 5, "rectify": True}
    dataset_ffdnet = {"data_source": "Bonn", "method": "FFDNet", "remove_below_line": False, "image_size": 128,
                      "minimum_drusen_height": 5, "rectify": True}
    dataset_gauss = {"data_source": "Bonn", "method": "Gauss2", "remove_below_line": False, "image_size": 128,
                     "minimum_drusen_height": 5, "rectify": True}
    dataset_bim = {"data_source": "Bonn", "method": "BiM", "remove_below_line": False, "image_size": 128,
                   "minimum_drusen_height": 5, "rectify": True}
    dataset_overview = {"data_source": "Bonn", "method": "overview", "remove_below_line": False, "image_size": 128,
                        "minimum_drusen_height": 5, "rectify": True}
    # dataset5 = {"data_source": "Duke", "method": "median", "remove_below_line": False, "image_size": 128,
    #            "minimum_drusen_height": 5}

    # Valid train_method values:
    # GAN, classic simple GAN architecture
    # ComplexGAN, has more layers, more trainable parameters
    # StyleGAN, use StyleGAN
    model_gan_10_10 = {"train_method": "GAN", "num_epochs": 10, "latent_size": 10, "augment_data": True, "mix": False,
                       "loss": "bce"}
    model_gan_200_20_mix_bce = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                                "mix": True, "loss": "bce"}
    model_gan_200_20_mix_mse = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True,
                                "mix": True, "loss": "mse"}
    model_gan_100_10 = {"train_method": "GAN", "num_epochs": 100, "latent_size": 10, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_gancomplex_100_10 = {"train_method": "ComplexGAN", "num_epochs": 100, "latent_size": 10, "augment_data": True,
                               "mix": False, "loss": "bce"}
    model_gan_100_20 = {"train_method": "GAN", "num_epochs": 100, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_gan_200_10 = {"train_method": "GAN", "num_epochs": 200, "latent_size": 10, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_gan_200_20 = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_cgan_200_20 = {"train_method": "ComplexGAN", "num_epochs": 200, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_cgan_10_20 = {"train_method": "ComplexGAN", "num_epochs": 10, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_gan_10_20 = {"train_method": "GAN", "num_epochs": 10, "latent_size": 20, "augment_data": True, "mix": False,
                       "loss": "bce"}
    model_gan_200_20 = {"train_method": "GAN", "num_epochs": 200, "latent_size": 20, "augment_data": True, "mix": False,
                        "loss": "bce"}
    model_stylegan_05 = {"train_method": "StyleGAN", "num_epochs": 5000, "augment_data": True,
                         "StyleGAN_Settings": "--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=0.5 --cbase=16384",
                         "latent_size": 30}
    model_stylegan_02 = {"train_method": "StyleGAN", "num_epochs": 500, "augment_data": True,
                         "StyleGAN_Settings": "--cfg=stylegan3-r --gpus=1 --batch=32 --gamma=0.5 --cbase=16384",
                         "latent_size": 10}

    # Valid values for latent discovery
    # Latent method: linear, warp
    # Latent steps, around 100-200k seems to be fine
    latent_linear_10k = {"method": "linear", "latent_steps": 10000, "deformator_type": "ORTHO"}
    latent_warp_1k = {"method": "warp", "latent_steps": 1000}
    latent_warp_10k = {"method": "warp", "latent_steps": 10000}
    latent_warp_200k = {"method": "warp", "latent_steps": 200000}
    latent_linear_100k = {"method": "linear", "latent_steps": 100000, "deformator_type": "ORTHO"}
    latent_linear_200k = {"method": "linear", "latent_steps": 200000, "deformator_type": "ORTHO"}

    runs = []
    runs.append({"dataset": dataset_noise, "model": model_gan_10_20, "latent_method": latent_warp_1k})  # test
    runs.append({"dataset": dataset_noise, "model": model_cgan_10_20, "latent_method": latent_warp_1k})  # test
    runs.append({"dataset": dataset_noise, "model": model_gan_200_20, "latent_method": latent_warp_200k})
    runs.append({"dataset": dataset_noise, "model": model_cgan_200_20, "latent_method": latent_warp_200k})
    runs.append(None)  # Early stop
    runs.append({"dataset": dataset_bim, "model": model_gan_200_20, "latent_method": latent_warp_200k})
    runs.append({"dataset": dataset_overview, "model": model_gan_100_10, "latent_method": latent_warp_1k})
    runs.append({"dataset": dataset_bim, "model": model_stylegan_02, "latent_method": latent_warp_1k})
    runs.append({"dataset": dataset_bim, "model": model_gan_200_20, "latent_method": latent_warp_1k})
    runs.append({"dataset": dataset_bim, "model": model_gan_200_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_200_20_mix_bce, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_200_20_mix_mse, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_10_10, "latent_method": latent_linear_10k})
    runs.append({"dataset": dataset_bim, "model": model_gan_10_10, "latent_method": latent_warp_1k})
    runs.append({"dataset": dataset_noise, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_gauss, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_200_10, "latent_method": latent_linear_200k})
    runs.append({"dataset": dataset_bim, "model": model_gan_200_10, "latent_method": latent_linear_200k})
    runs.append({"dataset": dataset_bim, "model": model_stylegan_05, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_10, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_bim, "model": model_gan_100_20, "latent_method": latent_linear_100k})
    runs.append({"dataset": dataset_gauss, "model": model_gan_100_10, "latent_method": latent_linear_200k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_10, "latent_method": latent_linear_200k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_10, "latent_method": latent_linear_200k})
    runs.append({"dataset": dataset_ffdnet, "model": model_gan_100_20, "latent_method": latent_linear_200k})

    run_manager(runs, random_seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

# TODO: StyleGAN should have less directions in latent deformator
