# IMPORTS
# For FFDNet
import os

import cv2
# For data generation
import eyepy as ep  # Documentation: https://medvisbonn.github.io/eyepy/
import numpy
import numpy as np
import glob
import torch
import torch.nn as nn
from scipy.ndimage import label, center_of_mass, binary_erosion, gaussian_filter, median_filter
from torch.autograd import Variable
import logging
import functions
import subprocess
import multiprocessing as mp
from functools import partial
import shutil


# FFDNet Models
class UpSampleFeatures(nn.Module):
    r"""Implements the last layer of FFDNet
    """

    def __init__(self):
        super(UpSampleFeatures, self).__init__()

    def forward(self, x):
        return functions.upsamplefeatures(x)


class IntermediateDnCNN(nn.Module):
    r"""Implements the middel part of the FFDNet architecture, which
    is basically a DnCNN net
    """

    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        if self.input_features == 5:
            self.output_features = 4  # Grayscale image
        elif self.input_features == 15:
            self.output_features = 12  # RGB image
        else:
            raise Exception('Invalid number of input features')

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features, out_channels=self.middle_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features,
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.output_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        self.itermediate_dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_dncnn(x)
        return out


class FFDNet(nn.Module):
    r"""Implements the FFDNet architecture
    """

    def __init__(self, num_input_channels):
        super(FFDNet, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 96
            self.num_conv_layers = 12
            self.downsampled_channels = 15
            self.output_features = 12
        else:
            raise Exception('Invalid number of input features')

        self.intermediate_dncnn = IntermediateDnCNN(
            input_features=self.downsampled_channels,
            middle_features=self.num_feature_maps,
            num_conv_layers=self.num_conv_layers)
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):
        concat_noise_x = functions.concatenate_input_noise_map(x.data, noise_sigma.data)
        concat_noise_x = Variable(concat_noise_x)
        h_dncnn = self.intermediate_dncnn(concat_noise_x)
        pred_noise = self.upsamplefeatures(h_dncnn)
        return pred_noise


# FFDNet Functions
def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


def is_rgb(im_path):
    r""" Returns True if the image in im_path is an RGB image
    """
    from skimage.io import imread
    rgb = False
    im = imread(im_path)
    if len(im.shape) == 3:
        if not (np.allclose(im[..., 0], im[..., 1]) and np.allclose(im[..., 2], im[..., 1])):
            rgb = True
    print("rgb: {}".format(rgb))
    print("im shape: {}".format(im.shape))
    return rgb


def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data / 255.)


# FUNCTION FOR ONE IMAGE RIGHT NOW - seems to be fast enough no need for batches right now
def ffdnet_one(image_in, datatype, noise_sigma, model):
    # imorig = cv2.imread(file_in, cv2.IMREAD_GRAYSCALE)
    imorig = image_in
    imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)
    imnoisy = imorig.clone()

    with torch.no_grad():  # PyTorch v0.4.0
        imorig, imnoisy = Variable(imorig.type(datatype)), Variable(imnoisy.type(datatype))
        nsigma = Variable(torch.FloatTensor([noise_sigma]).type(datatype))

    # Estimate noise and subtract it to the input image
    im_noise_estim = model(imnoisy, nsigma)
    outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
    return outim.detach().cpu().squeeze().numpy()


def get_drusen_folder(run):
    dataset = run["dataset"]
    data_src = dataset["data_source"]
    target_image_size = dataset["image_size"]
    minimum_drusen_height = dataset["minimum_drusen_height"]
    remove_below_line = dataset["remove_below_line"]
    rectify = dataset["rectify"]

    target_dir = f"{data_src}_{target_image_size}_{minimum_drusen_height}"
    if remove_below_line:
        target_dir += "_RBL"
    if rectify:
        target_dir += "_rect"
    return os.path.join("Data", target_dir, "1")


def get_gauss_sigma(run):
    return int(run["dataset"]["method"][5:])


def rbl(im, ind):
    mask = np.zeros_like(im)
    line_values = ind + 10
    rows = np.arange(im.shape[0])[:, np.newaxis]
    mask[rows < line_values] = 1
    return im * mask


def move_down(im, line):
    shift_values = np.max(line) - line
    shifted_im = np.zeros_like(im)
    #shifted_im[(slice(None), np.arange(im.shape[1]))] = im[(shift_values, np.arange(im.shape[1]))]
    for col in range(im.shape[1]):
        shifted_im[:, col] = np.roll(im[:, col], shift_values[col])
    return shifted_im


def process_scan(data, i, erosion_kernel_width, min_pixels_threshold,
                 target_image_size, target_dir, remove_below_line, target_filename, method, is_duke, rectify):

    # Original line
    line = np.asarray(data.layers["BM_2c41ukad" if is_duke else "BM"].data[i])
    # Remove nans from line
    nan_indices = np.argwhere(np.isnan(line)).ravel()
    if len(nan_indices) > 0:
        is_at_start = nan_indices[0] == 0
        last = 0
        if is_at_start:
            for index in nan_indices:
                if index == last:
                    last = index + 1
                else:
                    break
        for index in nan_indices[last:]:
            line[index] = line[index - 1]
        while last > 0:
            line[last - 1] = line[last]
            last -= 1
    line = line.astype(int)

    # Compute smooth line
    smooth_line = gaussian_filter(line, sigma=5)  # 5 seems to be good, 10 is not smoother
    im = rbl(data.data[i], smooth_line) if remove_below_line else data.data[i]
    if method == 6:  # overview
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im[line, np.arange(im.shape[1])] = np.array([0, 0, 255], dtype=int)
        im[smooth_line, np.arange(im.shape[1])] = np.array([0, 255, 0], dtype=int)

    if rectify:
        #filepath = os.path.join(target_dir, f"{target_filename}_scan{i:02d}.png")
        #cv2.imwrite(filepath, im)
        im = move_down(im, smooth_line)
        #filepath = os.path.join(target_dir, f"{target_filename}_scan{i:02d}_.png")
        #cv2.imwrite(filepath, im)

    # Drusen calculation
    scan = data.volume_maps["drusen"].data[i]
    if rectify:
        scan = move_down(scan, smooth_line)
    scan = binary_erosion(scan, iterations=1, structure=np.ones((erosion_kernel_width, erosion_kernel_width)))
    # Calculate the connected components
    labeled_scan, num_drusen = label(scan)
    # Calculate the center of each cluster that is large enough
    centers = []
    for drusen_id in range(1, num_drusen + 1):
        if np.count_nonzero(labeled_scan == drusen_id) < min_pixels_threshold:
            continue
        cluster_center = center_of_mass(scan, labeled_scan, drusen_id)
        centers.append((int(cluster_center[0]), int(cluster_center[1])))
    if not centers:
        return

    # Creating patches
    height, width = scan.shape
    for j, (row, col) in enumerate(centers):
        # Calculate start and end values for row indices
        start_row = row - target_image_size // 2
        end_row = start_row + target_image_size

        # Check and adjust for boundary cases
        if start_row < 0:
            start_row = 0
            end_row = target_image_size
        elif end_row > height:
            start_row = height - target_image_size
            end_row = height

        # Calculate start and end values for column indices
        start_col = col - target_image_size // 2
        end_col = start_col + target_image_size

        # Check and adjust for boundary cases
        if start_col < 0:
            start_col = 0
            end_col = target_image_size
        elif end_col > width:
            start_col = width - target_image_size
            end_col = width

        # Adjust rows for smooth line
        if rectify:
            diff = end_row - np.max(smooth_line) - 10
            end_row -= diff
            start_row -= diff
            start_row += 64  # Make it rectangular

        filepath = os.path.join(target_dir, f"{target_filename}_scan{i:02d}_drusen{j:02d}.png")

        if method == 6:
            to_save = im[start_row:end_row, start_col:end_col, :]  # numpy ndarray (128,128, 3)
        else:
            to_save = im[start_row:end_row, start_col:end_col]  # numpy ndarray (128,128)

        cv2.imwrite(filepath, to_save)


def process_file(filename, file_regex, minimum_drusen_height, erosion_kernel_width, min_pixels_threshold,
                 target_image_size, target_dir, remove_below_line, method, is_duke, rectify):
    target_filename = os.path.split(filename)[-1][:-len(file_regex) + 1]

    # Get eroded data
    data = ep.EyeVolume.load(filename)
    drusen5 = ep.drusen(data.layers["RPE"].data, data.layers["BM"].data, data.shape,
                        minimum_height=minimum_drusen_height)
    data.add_pixel_annotation(drusen5, name="drusen5")
    for i in range(len(data)):
        process_scan(data, i, erosion_kernel_width, min_pixels_threshold,
                     target_image_size, target_dir, remove_below_line, target_filename, method, is_duke, rectify)
    return


def prepare_ffdnet():
    # For FFDNet
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Prepare model
    in_ch = 1
    model_fn = 'models/FFDNet/net_gray.pth'
    net = FFDNet(num_input_channels=in_ch)
    state_dict = torch.load(model_fn)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(state_dict)
    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()
    return model


def filter_file(filepath, method, target_dir, gauss_sigma):
    filename = os.path.join(target_dir, os.path.split(filepath)[-1])
    g = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if method == 1:
        g = gaussian_filter(g, sigma=gauss_sigma)
    elif method == 4:
        g = cv2.bilateralFilter(g, 9, 25, 25)
        g = median_filter(g, size=2)
    elif method == 5:
        g = median_filter(g, size=2)
    cv2.imwrite(filename, g)


def generate(run, target_dir, method):

    # Only raw drusen should be generated
    if method == 0:
        generate_drusen(run, method)
        return

    # INIT Folders
    os.makedirs(target_dir, exist_ok=True)
    target_dir = os.path.join(target_dir, "1")  # Add a sub folder for data loader
    os.makedirs(target_dir, exist_ok=True)

    src_dir = get_drusen_folder(run)

    # Preparation for methods
    # Gauss
    gauss_sigma = 2  # Default value
    if method == 1:
        gauss_sigma = get_gauss_sigma(run)

    # FFDNet
    model = None
    if method == 2:
        model = prepare_ffdnet()

    # Get all images
    files = glob.glob(os.path.join(src_dir, "*"))

    if method == 2:  # One file after another need to be processed
        for filepath in files:
            filename = os.path.join(target_dir, os.path.split(filepath)[-1])
            g = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # FFDNet
            if method == 2:
                g = ffdnet_one(image_in=g, datatype=torch.cuda.FloatTensor, noise_sigma=75 / 255, model=model)
                g = (g * 255).astype(np.uint8)

            cv2.imwrite(filename, g)
    else:
        pool = mp.Pool(mp.cpu_count())

        partial_process_item = partial(filter_file, method=method, target_dir=target_dir, gauss_sigma=gauss_sigma)
        pool.map(partial_process_item, files)

        pool.close()
        pool.join()

    return


def generate_drusen(run, method):
    dataset = run["dataset"]
    target_image_size = dataset["image_size"]
    remove_below_line = dataset["remove_below_line"]
    minimum_drusen_height = dataset["minimum_drusen_height"]
    rectify = dataset["rectify"]

    target_dir = get_drusen_folder(run)
    if method == 6:
        target_dir = target_dir + "_ov"
        if os.path.exists(target_dir):  # redo if it already exists
            shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    is_duke = dataset["data_source"] == "Duke"

    folder = os.path.join("Data", "Duke_processed" if is_duke else "Bonn")
    file_regex = "*.eye"
    erosion_kernel_width = 7
    min_pixels_threshold = 20
    files = glob.glob(os.path.join(folder, file_regex))
    pool = mp.Pool(mp.cpu_count())
    partial_process_item = partial(process_file, file_regex=file_regex, minimum_drusen_height=minimum_drusen_height,
                                   erosion_kernel_width=erosion_kernel_width,
                                   min_pixels_threshold=min_pixels_threshold,
                                   target_image_size=target_image_size,
                                   target_dir=target_dir, remove_below_line=remove_below_line, method=method,
                                   is_duke=is_duke, rectify=rectify)
    pool.map(partial_process_item, files)

    pool.close()
    pool.join()


# generates the data
def gen_data(run, logger: logging.Logger):
    dataset = run["dataset"]
    data_src = dataset["data_source"]
    noise_method = dataset["method"]
    target_image_size = dataset["image_size"]
    remove_below_line = dataset["remove_below_line"]
    minimum_drusen_height = dataset["minimum_drusen_height"]
    rectify = dataset["rectify"]

    is_for_stylegan = run["model"]["train_method"] == "StyleGAN"

    gauss_sigma = 2
    method = 0  # none
    target_dir = f"{data_src}_{target_image_size}_{minimum_drusen_height}"
    if remove_below_line:
        target_dir += "_RBL"
    if rectify:
        target_dir += "_rect"
    if noise_method.startswith("Gauss"):
        method = 1
        gauss_sigma = get_gauss_sigma(run)
        noise_method = "Gauss"
        target_dir = target_dir + f"_gauss{gauss_sigma}"
    if noise_method == "FFDNet":
        method = 2
        target_dir = target_dir + "_ffdnet"
    if noise_method == "mnist":
        method = 3
        target_dir = "mnist_training"
    if noise_method == "BiM":
        method = 4
        target_dir = target_dir + "_bim"
    if noise_method == "median":
        method = 5
        target_dir = target_dir + "_median"
    if noise_method == "overview":
        method = 6
        target_dir = target_dir + "_ov"


    target_dir = os.path.join("Data", target_dir)
    run["dataroot"] = target_dir

    # Don't generate if already exists:
    if os.path.exists(target_dir):
        logger.info("Data already exists")
    elif method == 6:  # special case for generating overview
        generate_drusen(run, method)
    # Don't generate for mnist
    elif method != 3:
        if method != 0 and not os.path.exists(get_drusen_folder(run)):
            logger.info("Need to generate drusen first")
            generate_drusen(run, method)
            logger.info("Finished generating drusen")
        logger.info(f"Start generating data of size {target_image_size} with method {noise_method}"
                    + (f", with sigma={gauss_sigma}" if noise_method == "Gauss" else "")
                    + (f" and remove below line" if remove_below_line else ""))
        generate(run, target_dir, method)

    if is_for_stylegan:
        src_dir = target_dir
        target_zip = target_dir + ".zip"
        if os.path.exists(target_zip):
            logger.info("Data is already converted to StyleGAN")
            return
        filename = os.path.join("stylegan3", "dataset_tool.py")
        cmd = ["python", filename,
               "--source=" + src_dir,
               "--dest=" + target_zip]
        logger.info(f"Converting dataset to StyleGAN with command{cmd}")
        subprocess.run(cmd)
        logger.info("Finished converting")
    return
