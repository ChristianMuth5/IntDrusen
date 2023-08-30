import glob
import json
import os
import filecmp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp
from functools import partial
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.utils.data import WeightedRandomSampler
from SimpleGAN_Rect import get_generator


def compare_image_folders(f1, f2):
    files_f1 = [f for f in os.listdir(f1) if os.path.isfile(os.path.join(f1, f))]
    files_f2 = [f for f in os.listdir(f2) if os.path.isfile(os.path.join(f2, f))]

    if len(files_f1) != len(files_f2):
        return False

    for file in files_f1:
        path_f1 = os.path.join(f1, file)
        path_f2 = os.path.join(f2, file)
        if not filecmp.cmp(path_f1, path_f2, shallow=False):
            return False

    return True


def check_folder_equals():
    f1 = os.path.join("Data", "Duke_drusen_128_5_ffdnet_RBL_forloop", "1")
    f2 = os.path.join("Data", "Duke_drusen_128_5_ffdnet_RBL", "1")
    print(compare_image_folders(f1, f2))


def generate_overview(folder_path):
    file_regex = "*"
    files = glob.glob(os.path.join(folder_path, file_regex))
    files = np.random.choice(files, 50, replace=False)
    # Define the grid layout
    rows = 5
    columns = 10

    fig, axes = plt.subplots(rows, columns, figsize=(20, 10))

    for i, ax in enumerate(axes.flatten()):
        filename = files[i]
        image = mpimg.imread(filename)
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("Overview.png")
    plt.close()


def analyze_hist(path):
    with open(path, "r") as f:
        data = json.load(f)
        vals = np.asarray(list(data.values()))

        for i in range(3):
            print(["Height", "Width", "Volume"][i])
            print(np.histogram(vals[:, i], bins=5))


def analyze_hists(folder):
    print("Original Data:")
    analyze_hist(os.path.join(folder, "info_dict_original.json"))
    print("Scaled Data:")
    analyze_hist(os.path.join(folder, "info_dict.json"))


def create_one_overview(data):
    drusen = data[0]
    volumes = data[1]
    # Define the grid layout
    rows = 5
    columns = 10

    fig, axes = plt.subplots(rows, columns, figsize=(20, 10))

    for i in range(50):
        ax = axes[i//10][i%10]
        if i < len(volumes):
            image = mpimg.imread(drusen[i])
            ax.imshow(image, cmap='gray')
            ax.set_title(f"{volumes[i]}")
        ax.axis('off')

    plt.tight_layout()
    folder = os.path.join(os.path.split(drusen[0])[0], os.pardir, "Overview_original")
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f"Overview_{data[2]:04d}.png"))
    plt.close()


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
        return image, labels, index


def analyze_volume_classes(folder):
    with open(os.path.join(folder, "info_dict.json")) as f:
        info = json.load(f)

    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(0.5, 0.5)
    ])
    droot = os.path.join("Data", "Bonn_128_5_rect_w10_s")
    dataset = CustomDataset(droot, info, sorted(info.keys()), transformations, [2])

    # Create the dataloader

    vals = np.asarray(list(info.values()))[:, 2]
    class_counts = np.unique(vals, return_counts=True)[1]
    print("Original class counts")
    print(class_counts)
    class_weights = 1 / class_counts
    print(class_weights)
    class_weights = class_weights * np.asarray([1, 4/6, 8/29, 16/111, 32/775])
    class_weights = np.asarray([1 / 2 ** n for n in range(5, 0, -1)])
    print(class_weights)
    sample_weights = torch.tensor([class_weights[x[1]] for x in dataset])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

    dataloader = DataLoader(dataset, batch_size=128, num_workers=16, sampler=sampler)

    class_counter = np.asarray([0, 0, 0, 0, 0])
    indices = np.zeros(len(dataset))
    for i, data in enumerate(dataloader, 0):
        classes = data[1]
        index = data[2]
        a, b = np.unique(classes.numpy(), return_counts=True)
        for i in range(len(a)):
            class_counter[a[i]] = class_counter[a[i]] + b[i]
        for i in index:
            indices[i] = indices[i] + 1
    print("Moditied sampling class counts")
    print(class_counter)  # [14827 14936 14927 14606 14847]
    print(np.sum(class_counter))  # 74143
    print(np.sum(indices))
    print(np.unique(indices, return_counts=True))

"""
(array([0.00000, 1.00000, 2.00000, 3.00000, 4.00000, 5.00000, 6.00000,
       7.00000, 8.00000, 9.00000, 10.00000, 11.00000, 12.00000, 13.00000,
       14.00000, 15.00000, 16.00000, 17.00000, 18.00000, 20.00000,
       21.00000]), array([31403, 24818, 10972,  3989,  1530,   653,   323,   166,    90,
          67,    36,    26,    18,    24,     9,    10,     5,     1,
           1,     1,     1]))
"""

def create_volume_overview(folder):
    #drusen = np.asarray(glob.glob(os.path.join(folder, "Segmentations", "*.png")))
    info_dict = None
    with open(os.path.join(folder, "info_dict_original.json"), "r") as f:
        info_dict = json.load(f)
    volumes = np.asarray(list(info_dict.values()))[:, 2]
    drusen = np.asarray([os.path.join(folder, "1", x) for x in info_dict.keys()])
    nums = np.argsort(volumes)

    indices = [nums[i:min(i+50, nums.shape[0])] for i in range(0, nums.shape[0], 50)]
    data = [(drusen[indices[i]], volumes[indices[i]], i) for i in range(len(indices))]
    pool = mp.Pool(mp.cpu_count())
    partial_process_item = partial(create_one_overview)
    pool.map(partial_process_item, data)
    pool.close()
    pool.join()

def main():
    np.set_printoptions(formatter={'float_kind': "{:.5f}".format})
    #check_folder_equals()
    #generate_overview(os.path.join("Data", "Bonn_128_5_RBL", "1_ov"))

    folder = os.path.join("Data", "Bonn_128_5_rect_w10_s")
    #analyze_hists(folder)
    #analyze_volume_classes(folder)
    path = os.path.join("Data", "Bonn_128_5_rect_w10_s_aug_bce_ws_results_GAN_200epochs_20nz_warp1000_20",
                        "experiments", "complete", "GAN128-LeNet-K20-D5-LearnGammas-eps0.15_0.25", "results",
                        "GAN128_20", "32_0.2_6.4")
    drusen_folder = sorted(glob.glob(os.path.join(path, "4d12ee4688793d6d12e4d8b9a318943685f77af6")))
    #path = os.path.join("Data", "Bonn_128_5_rect_w10_s_aug_bce_ws_results_GAN_200epochs_20nz_pde200_20",
    #                    "experiments", "complete", "GAN128-LeNet-K20-D10", "results",
    #                    "GAN128_20", "64_0.2_12.8")
    #drusen_folder = sorted(glob.glob(os.path.join(path, "4a12aeae2d0678fd35dbcafea14c8bd2e4791c96")))
    G = get_generator(1, 20)
    g_path = os.path.join("Data", "Bonn_128_5_rect_w10_s_aug_bce_ws_results_GAN_200epochs_20nz", "generator.pt")
    G.load_state_dict(torch.load(g_path))
    G.eval()
    for df in drusen_folder:
        latent_codes = torch.load(os.path.join(df, "paths_latent_codes.pt"), map_location=lambda storage, loc: storage)
        path_folder = os.path.join(df, "paths_images", "path_000")
        #path_folder = os.path.join(df, "paths_images", "path_001")
        path_images = sorted(glob.glob(os.path.join(path_folder, "*.jpg")))
        print(df)
        fig, axes = plt.subplots(2, 10, figsize=(20, 10))
        for i in range(10):
            image_latent = latent_codes[0][i].unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            ax = axes[0][i]
            image = mpimg.imread(path_images[i])
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Orig path")
            ax.axis('off')
            ax = axes[1][i]
            image = G(image_latent).cpu().detach().numpy().squeeze()
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Latent in G")
            ax.axis('off')
        plt.savefig(os.path.join("Data", "Comp.jpg"))
        plt.close()
        return

    #middle_drusen_code = latent_codes[0][latent_codes.shape[1]//2]
    # Format: path_i, image_in_path_i, vector


if __name__ == "__main__":
    main()
