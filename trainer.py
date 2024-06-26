from __future__ import print_function

import glob
import logging
import os
import pickle
import random
import subprocess
import sys
import shutil
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import SimpleGAN
import SimpleGAN_Rect
import ComplexGAN
import ComplexGAN_Rect
import SimpleGAN_Rect_Cond


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


class CustomDataset(Dataset):
    def __init__(self, image_folder, info_dict, sorted_keys, transform=None, conditions=None):
        self.image_folder = image_folder
        self.info_dict = info_dict
        self.sorted_keys = sorted_keys
        self.transform = transform
        self.conditions = conditions
        self.is_volume = len(self.conditions) == 1 and 2 in self.conditions

    def __len__(self):
        return len(self.info_dict)

    def __getitem__(self, index):
        fname = self.sorted_keys[index]
        image_path = os.path.join(self.image_folder, "1", fname)
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        if self.conditions is None:
            return image

        if self.is_volume:
            labels = torch.tensor(int(self.info_dict[fname][2]))
        else:
            labels = torch.tensor(np.asarray(self.info_dict[fname])[self.conditions])
        return image, labels


# Loads the specified dataset
def load_dataset(droot, augmented, batch_size, workers, conditions, weight_samples):
    # dataset = dset.ImageFolder(root=droot,
    #                           transform=transforms.Compose([
    #                               transforms.Grayscale(),
    #                               transforms.ToTensor(),
    #                           ]))
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=workers)
    #
    # mean, std = get_mean_and_std(dataloader)
    mean = 0.5
    std = 0.5

    if augmented:
        transformations = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
        ])
    else:
        transformations = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    with open(os.path.join(droot, "info_dict.json")) as f:
        info = json.load(f)

    # Create the dataloader
    if weight_samples:
        dataset = CustomDataset(droot, info, sorted(info.keys()), transformations, [2])

        #vals = np.asarray(list(info.values()))[:, 2]
        #class_counts = np.unique(vals, return_counts=True)[1]
        #class_weights = 1 / class_counts
        # Manual set to probabilities: p, 2*p, 4*p, 8*p, 16*p
        #class_weights = class_weights * np.asarray([1, 4 / 6, 8 / 29, 16 / 111, 32 / 775])
        class_weights = np.asarray([1 / 2 ** n for n in range(5, 0, -1)])

        sample_weights = torch.tensor([class_weights[x[1]] for x in dataset])

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, sampler=sampler)
    else:
        dataset = CustomDataset(droot, info, sorted(info.keys()), transformations, conditions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader


def show_current(dataloader, img_list, imgs_folder, epoch, device):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                            (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(os.path.join(imgs_folder, f"fake_imgs_{epoch}.png"))
    # plt.show()
    plt.close()


def get_generator(train_method, nz, rectangular, ngpu, conditions):
    if conditions:
        if rectangular:
            if train_method == "ComplexGAN":
                G = None
            else:
                G = SimpleGAN_Rect_Cond.get_generator(ngpu, nz, conditions)
        else:
            if train_method == "ComplexGAN":
                G = None
            else:
                G = None
    else:
        if rectangular:
            if train_method == "ComplexGAN":
                G = ComplexGAN_Rect.get_generator(ngpu, nz)
            else:
                G = SimpleGAN_Rect.get_generator(ngpu, nz)
        else:
            if train_method == "ComplexGAN":
                G = ComplexGAN.get_generator(ngpu, nz)
            else:
                G = SimpleGAN.get_generator(ngpu, nz)
    return G


def get_discriminator(train_method, rectangular, ngpu, conditions):
    if conditions:
        if rectangular:
            if train_method == "ComplexGAN":
                D = None
            else:
                D = SimpleGAN_Rect_Cond.get_discriminator(ngpu, conditions)
        else:
            if train_method == "ComplexGAN":
                D = None
            else:
                D = None
    else:
        if rectangular:
            if train_method == "ComplexGAN":
                D = ComplexGAN_Rect.get_discriminator(ngpu)
            else:
                D = SimpleGAN_Rect.get_discriminator(ngpu)
        else:
            if train_method == "ComplexGAN":
                D = ComplexGAN.get_discriminator(ngpu)
            else:
                D = SimpleGAN.get_discriminator(ngpu)
    return D


def feed_D(D, input, labels):
    if labels is None:
        return D(input).view(-1)
    return D(input, labels).view(-1)


def train_own_model(run, folder_out, logger: logging.Logger):
    model_param = run["model"]
    num_epochs = model_param["num_epochs"]
    train_method = model_param["train_method"]
    augmented = model_param["augment_data"]
    latent_size = model_param["latent_size"]
    random_seed = run["random_seed"]
    dataroot = run["dataroot"]
    mix = model_param["mix"]
    loss = model_param["loss"]
    conditions = model_param["conditions"]
    conditional = len(conditions) > 0
    dataset = run["dataset"]
    rectangular = dataset["rectify"]
    weight_samples = model_param["weight_samples"] if "weight_samples" in model_param else False
    # Number of workers for dataloader
    workers = 16
    # Number of channels in the training images. For color images this is 3
    nc = 1
    # Size of z latent vector (i.e. size of generator input)
    nz = latent_size
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # Threshold to decide when the loss of the Discriminator is too low, restarts training
    loss_d_threshold = 0.0002
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Check status of GAN every x-th epoch
    check_status_every_epoch = 5
    # Save G and D every x-th epoch
    save_networks_every_epoch = num_epochs // 5
    # Learning rate for optimizers
    lr_G = 0.0002
    lr_D = 0.0001
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # Batch size during training
    batch_size = model_param["batch_size"]

    generator_path = os.path.join(folder_out, "generator.pt")
    discriminator_path = os.path.join(folder_out, "discriminator.pt")

    logger.info(f"Start training on {dataroot} with {train_method} for {num_epochs} epochs")
    if os.path.exists(generator_path):
        logger.info("Already trained with these settings")
        G = get_generator(train_method, nz, rectangular, ngpu, conditions)
        G.load_state_dict(torch.load(generator_path))
        G.eval()
        return G, folder_out

    # GAN Training Code
    def train_loop(run_number, seed):
        # Set random seed for reproducibility
        if seed == -1:
            seed = random.randint(1, 10000)

        logger.debug(f"Random Seed: {seed}")
        random.seed(seed)
        torch.manual_seed(seed)

        dataloader = load_dataset(dataroot, augmented, batch_size, workers, conditions, weight_samples)

        # Create the generator
        netG = get_generator(train_method, nz, rectangular, ngpu, conditions)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, std=0.02.
        netG.apply(weights_init)

        # Print the model
        logger.debug(str(netG))

        # Create the Discriminator
        netD = get_discriminator(train_method, rectangular, ngpu, conditions)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, std=0.2.
        netD.apply(weights_init)

        # Print the model
        logger.debug(str(netD))

        # Initialize BCELoss function
        if loss == "mse":
            criterion = nn.MSELoss()
        else:  # bce
            criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        # Init folders
        imgs_folder = os.path.join(folder_out, f"run_{run_number}_imgs")
        log_folder = os.path.join(folder_out, f"run_{run_number}_log")
        os.makedirs(log_folder, exist_ok=True)  # Output temp models
        os.makedirs(imgs_folder, exist_ok=True)  # Output folder for images

        logger.debug("Starting Training Loop...")

        # Count how often the loss of G or D is too small in a streak
        failure_streak = 0

        # For each epoch
        epoch = 0
        while epoch < num_epochs:
            # Start time of the current epoch
            epoch_start_time = datetime.now()

            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                # TEMP:
                labels = None
                if conditional:
                    labels = data[1].unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).to(device).type(torch.float)

                # Forward pass real batch through D
                output = feed_D(netD, real_cpu, labels)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                if conditional:
                    noise = torch.cat((labels, torch.randn(b_size, nz-len(conditions), 1, 1, device=device)), dim=1)
                else:
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = feed_D(netD, fake.detach(), labels)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                # Train with real and fake batches mixed
                if mix:
                    factor = 0.0
                    while factor < 0.1 or factor > 0.9:
                        factor = torch.normal(0.5, 0.1, (1,)).data[0]
                    mixed_data = real_cpu * factor + fake.detach() * (1.0 - factor)
                    label.fill_(real_label * factor + fake_label * (1.0 - factor))
                    output = feed_D(netD, mixed_data, labels)
                    errD_mix = criterion(output, label)
                    errD_mix.backward()
                    D_mix = output.mean().item()

                # Compute error of D as sum over the fake and the real batches
                if mix:
                    errD = errD_real + errD_fake + errD_mix
                else:
                    errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = feed_D(netD, fake, labels)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Output training stats and check loss
                if i % 100 == 0:
                    logger.debug(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}]\tLoss_D: {errD.item():.4f}" +
                                 f"\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

                    # See if the training might fail
                    if errD.item() < loss_d_threshold or errD.item() > 100 - loss_d_threshold:
                        failure_streak += 1
                    else:
                        failure_streak = 0
                    if failure_streak > 9:  # stop stopping
                        logger.debug(f"Loss of D is smaller than {loss_d_threshold}, 10 times in a row, restart training")
                        return False, None

                iters += 1
            epoch += 1
            # End time of the current epoch
            epoch_elapsed_time = datetime.now() - epoch_start_time
            logger.debug(f"Epoch {epoch:4d} took {str(epoch_elapsed_time)}")

            # Save G and D
            if epoch % save_networks_every_epoch == 0 and epoch < num_epochs:
                logger.debug("Saving G and D")
                torch.save(netG.state_dict(), os.path.join(log_folder, f"temp_generator_{epoch}.pt"))
                torch.save(netD.state_dict(), os.path.join(log_folder, f"temp_discriminator_{epoch}.pt"))
            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % check_status_every_epoch == 0) or (epoch == num_epochs - 1):
                logger.debug("Checking how the generator is doing")
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                logger.debug("Saving image output of G")
                show_current(dataloader, img_list, imgs_folder, epoch, device)

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(imgs_folder, f"losses.png"))
        plt.close()
        # Saving the model
        logger.debug("Saving Final G and D")
        torch.save(netG.state_dict(), generator_path)
        torch.save(netD.state_dict(), discriminator_path)
        logger.debug("Training finished")
        return True, netG

    finished_training = False
    G = None
    i = 0
    while not finished_training:
        i += 1
        if i == 10:
            logger.info("Too many failures, quit trying")
            return None, folder_out
        finished_training, G = train_loop(i, random_seed)

    model_path = "SimpleGAN"
    if train_method == "ComplexGAN":
        model_path = "ComplexGAN"
    if rectangular:
        model_path += "_Rect"
    model_path += ".py"
    shutil.copy(model_path, os.path.join(folder_out, "model.py"))

    logger.info("Finished training")
    return G, folder_out


def get_matching_stylegan(folder_out, settings: str, datafolder_name):
    all_files = glob.glob(os.path.join(folder_out, "*"))
    folders = [os.path.split(file)[-1] for file in all_files if os.path.isdir(file)]
    folders = [folder for folder in folders if "-" + datafolder_name + "-" in folder]

    tracked_settings = ["gpus", "batch", "gamma"]
    for setting in settings.split(" "):
        if setting.startswith("--cgf="):
            folders = [folder for folder in folders if setting.split("=")[1] in folder]
            continue
        k, v = setting.split("=")
        k = k[1:]
        if k[1:] in tracked_settings:
            folders = [folder for folder in folders if k + v in folder]

    folders = [folder for folder in folders if glob.glob(os.path.join(folder_out, folder, "*.pkl"))]
    if not folders:
        return None
    g_files = glob.glob(os.path.join(folder_out, folders[-1], "*.pkl"))
    return sorted(g_files)[-1]


def train_stylegan(run, folder_out, logger: logging.Logger):
    # https://stackoverflow.com/questions/69169145/how-to-make-python-libraries-accessible-in-pythonpath
    # save the literal filepath to both directories as strings
    top_path = os.path.abspath(os.path.join("stylegan3"))
    tu_path = os.path.abspath(os.path.join('stylegan3', 'torch_utils'))
    dnnlib_path = os.path.abspath(os.path.join('stylegan3', 'dnnlib'))
    # add those strings to python path
    if top_path not in sys.path:
        sys.path.append(top_path)
    if tu_path not in sys.path:
        sys.path.append(tu_path)
    if dnnlib_path not in sys.path:
        sys.path.append(dnnlib_path)

    dataroot = run["dataroot"]
    model_param = run["model"]

    datafolder_name = os.path.split(dataroot)[-1]

    settings = model_param["StyleGAN_Settings"]
    path_to_g = get_matching_stylegan(folder_out, settings, datafolder_name)

    if path_to_g and path_to_g is not None:
        logger.info("Already trained with these settings")
    else:
        filename = os.path.join("stylegan3", "train.py")

        cmd = ["python", filename,
               "--outdir=" + folder_out,
               "--data=" + dataroot + ".zip"]

        cmd += [setting for setting in settings.split(" ")]

        if "--kimg=" not in settings:
            cmd += ["--kimg=" + str(model_param["num_epochs"])]
        # if "--cmax=" not in settings:
        #    cmd += ["--cmax=" + str(run["latent_size"])]
        if "--mirror=" not in settings and model_param["augment_data"]:
            cmd += ["--mirror=1"]
        if "--snap=" not in settings:
            cmd += ["--snap=" + str(model_param["num_epochs"] // 20)]

        logger.debug(f"Command: {cmd}")
        logger.info(f"Start training {model_param['train_method']}")
        subprocess.run(cmd)
        logger.info(f"Finished training {model_param['train_method']}")
        path_to_g = get_matching_stylegan(folder_out, settings, datafolder_name)

    with open(path_to_g, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
    # z = torch.randn([1, G.z_dim]).cuda()  # latent codes
    # c = None  # class labels (not used in this example)
    # img = G(z, c)  # NCHW, float32, dynamic range [-1, +1], no truncation
    run["model"]["path_to_g"] = path_to_g
    return G, folder_out


# Training Loop
# run: params for the trainer
# dataroot: Root directory for dataset, should have a subfolder for each class
def train(run, logger: logging.Logger):
    plt.switch_backend('agg')  # does this fix the tkinter bug? is it enough to put this here?
    model_param = run["model"]
    num_epochs = model_param["num_epochs"]
    train_method = model_param["train_method"]
    augmented = model_param["augment_data"]
    is_stylegan = train_method in ["StyleGAN2", "StyleGAN3"]
    latent_size = model_param["latent_size"] if not is_stylegan else None
    mix = model_param["mix"] if not is_stylegan else False
    loss = model_param["loss"] if not is_stylegan else None
    conditions = model_param["conditions"] if not is_stylegan else []
    dataroot = run["dataroot"]
    weight_samples = model_param["weight_samples"] if "weight_samples" in model_param else False

    # Where to save output
    folder_out = f"{dataroot}"
    if augmented:
        folder_out += "_aug"
    if mix:
        folder_out += "_mix"
    if loss:
        folder_out += f"_{loss}"
    if conditions:
        folder_out += "_c="
    if 0 in conditions:
        folder_out += "h"
    if 1 in conditions:
        folder_out += "w"
    if 2 in conditions:
        folder_out += "v"
    if weight_samples:
        folder_out += "_ws"
    folder_out += f"_results_{train_method}_{num_epochs}epochs"
    if not is_stylegan:
        folder_out += f"_{latent_size}nz"
    os.makedirs(folder_out, exist_ok=True)  # General output folder

    # Init the logger
    file_handler = logging.FileHandler(os.path.join(folder_out, 'mainlog.txt'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def close_logger():
        file_handler.close()
        logger.removeHandler(file_handler)

    if train_method == "GAN" or train_method == "ComplexGAN":
        G, folder_out = train_own_model(run, folder_out, logger)
        close_logger()

    elif is_stylegan:
        G, folder_out = train_stylegan(run, folder_out, logger)
        close_logger()
    else:
        G, folder_out = None, None

    # drusen dir will not exists for newly trained models, thus preventing this code from running when no human has yet
    # looked at possibly interesting drusen created by this model
    drusen_dir = os.path.join(folder_out, "drusen")
    if "drusen" in run and os.path.exists(drusen_dir):
        drusen = run["drusen"]
        int_drusen_dir = os.path.join(folder_out, "interesting_drusen")
        if os.path.exists(int_drusen_dir):
            shutil.rmtree(int_drusen_dir)
            os.makedirs(int_drusen_dir)
        for i, name in [(0, "Reticular Pseudo Drusen"), (1, "Drusenoid PED"), (2, "Small hard Drusen"), (3, "Large soft Drusen"), (4, "Other")]:
            sub_dir = os.path.join(int_drusen_dir, name)
            os.makedirs(sub_dir)
            for drusen_i in drusen[i]:
                shutil.copy(os.path.join(drusen_dir, f"image_{drusen_i}.jpg"), sub_dir)
                shutil.copy(os.path.join(drusen_dir, f"latent_code_{drusen_i}.pt"), sub_dir)

    return G, folder_out
