import glob
import os
import json
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import multiprocessing as mp
from functools import partial
from skimage.exposure import match_histograms


def normalize_info(target_dir, scaling=1):
    # If the directory does not exist, return instead of crashing
    if not os.path.exists(os.path.join(target_dir, "temp")):
        return

    npy_files = glob.glob(os.path.join(target_dir, "temp", "*.npy"))
    np_arrs = []
    for f in npy_files:
        np_arrs.append(np.load(f))
    data = np.concatenate(np_arrs, axis=1)
    mi = np.min(data, axis=1)
    ma = np.max(data, axis=1) - mi

    json_files = glob.glob(os.path.join(target_dir, "temp", "*.json"))
    info_dict = dict()
    for fname in json_files:
        with open(fname, "r") as f:
            info = json.load(f)
            #l_scale = np.log(scaling+1)
            for k, (a, b, c) in info.items():
                #h = scaling * (a - mi[0]) / ma[0]
                #w = scaling * (b - mi[1]) / ma[1]
                h = (a - mi[0]) / ma[0]
                w = (b - mi[1]) / ma[1]
                v = c#(c - mi[2]) / ma[2]
                #if scaling > 1:
                #    info_dict[k] = (np.log(h+1)/l_scale, np.log(w+1)/l_scale)
                #else:
                info_dict[k] = (h, w, v)

    with open(os.path.join(target_dir, "info_dict_original.json"), "w") as f:
        json.dump(info_dict, f)

    if scaling is not None:
        # Either scale or match histogram
        if scaling > 1:
            l_scale = np.log(scaling + 1)
            for k, (h, w, v) in info_dict.items():
                info_dict[k] = (np.log(scaling * h + 1) / l_scale,
                                np.log(scaling * w + 1) / l_scale,
                                v)#np.log(scaling * v + 1) / l_scale)
        else:
            a = np.asarray(list(info_dict.values()))
            a = a + np.random.normal(0, 1/1000, a.shape)
            a0 = match_histograms(a[:, 0], np.arange(0, 1, 1 / 10000))
            a1 = match_histograms(a[:, 1], np.arange(0, 1, 1 / 10000))
            a2 = a[:, 2]#match_histograms(a[:, 2], np.arange(0, 1, 1 / 10000))

            keys = list(info_dict.keys())
            for i in range(len(info_dict)):
                info_dict[keys[i]] = (a0[i], a1[i], a2[i])

    # Save info_dict
    fname = os.path.join(target_dir, "info_dict.json")
    print(f"{len(info_dict)} drusen created")
    with open(fname, "w") as f:
        json.dump(info_dict, f)

    shutil.rmtree(os.path.join(target_dir, "temp"))
    show_modified_histograms(target_dir)


def show_50_drusen(data):
    files = data[0]
    out_dir = data[1]
    sums = data[2]
    indices = data[3]
    j = data[4]

    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    for i in range(50):
        plt_row = i // 10
        plt_col = i % 10

        ax = axes[plt_row][plt_col]
        ax.axis('off')
        if i < len(indices):
            im = mpimg.imread(files[indices[i]])
            ax.set_title(f"Sum = {sums[indices[i]]}")
            ax.imshow(im, cmap='gray')

    plt.savefig(os.path.join(out_dir, f"Drusen_{j:03d}.jpg"))
    plt.close()


def show_all_drusen(folder):
    out_dir = folder + "_overview"
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(folder, "1", "*"))
    arr = np.zeros((len(files), 64, 128))
    for i in range(len(files)):
        arr[i] = Image.open(files[i])
    sums = np.sum(arr, axis=(1, 2))
    indices = np.argsort(sums)

    indices = [indices[i:min(i+50, len(indices))] for i in range(0, len(indices), 50)]

    data = [(files, out_dir, sums, indices[i], i) for i in range(len(indices))]

    pool = mp.Pool(mp.cpu_count())
    partial_process_item = partial(show_50_drusen)
    pool.map(partial_process_item, data)
    pool.close()
    pool.join()


def show_hist_infos(folder, bins):
    files = sorted(glob.glob(os.path.join(folder, "info_dict*.json")))
    nums = []
    for f in files:
        fname = os.path.split(f)[-1].split(".")[0]
        if len(fname) == 9:
            nums.append(0)
        else:
            nums.append(int(fname[10:]))
    files = np.asarray(files)[np.argsort(nums)]
    nums = np.sort(nums)
    data = []
    for fname in files:
        with open(fname, "r") as f:
            info = json.load(f)
            data_temp = []
            for k, v in info.items():
                data_temp.append(v)
            data.append(np.asarray(data_temp))
    data = np.asarray(data)
    rows = data.shape[0]
    cols = data.shape[2]

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    for row in range(rows):
        for col in range(cols):
            ax = axes[row][col]
            ax.hist(data[row, :, col], bins=bins)
            var = ["Height", "Width", "Volume"][col]
            ax.set_title(f"Scale={nums[row]}, Var={var}")
    plt.savefig(os.path.join(folder, "hists.jpg"))
    plt.close()


def show_scalings(folder):
    scalings = [1, 10, 20, 40, 50, 60]
    for scaling in scalings:
        print(scaling)
        normalize_info(os.path.join(folder))
    show_hist_infos(folder, 20)


def show_modified_histograms(folder):
    with open(os.path.join(folder, "info_dict_original.json"), "r") as f:
        info_orig = json.load(f)
    with open(os.path.join(folder, "info_dict.json"), "r") as f:
        info = json.load(f)
    rows = 2
    cols = 3
    row_names = ["Original", "Hist Equal"]
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    data_orig = np.asarray(list(info_orig.values()))
    data = np.asarray(list(info.values()))

    for row in range(rows):
        for col in range(cols):
            ax = axes[row][col]
            var = ["Height", "Width", "Volume"][col]
            ax.set_title(f"{row_names[row]}, Var={var}")
            b = 20
            if row == 0:
                ax.hist(data_orig[:, col], bins=b)
            elif row == 1:
                ax.hist(data[:, col], bins=b)
                #ax.hist(match_histograms(data_orig[:, col], np.arange(0, 1, 1/10000)), bins=b)
                #ax.hist(np.arange(0, 1, 1/10000), bins=100)

    plt.savefig(os.path.join(folder, f"hists_comparison.jpg"))
    plt.close()


def show_hist_for_dict(file):
    with open(file, "r") as f:
        info = json.load(f)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        a = np.asarray(list(info.values()))
        ax = axes[0][0]
        ax.hist(a[:, 0], bins=20)
        ax = axes[0][1]
        ax.hist(a[:, 1], bins=20)
        plt.savefig(os.path.join(os.path.split(file)[0], "hist_final.jpg"))
        plt.close()

def main():
    # normalize_info(os.path.join("Data", "Bonn_128_5_rect_w10_s (copy)"))
    # show_all_drusen(os.path.join("Data", "Bonn_128_5_rect_w10_s"))
    # show_scalings(os.path.join("Data", "Bonn_128_5_rect_w10_s"))
    # show_modified_histograms(os.path.join("Data", "Bonn_128_5_rect_w10_s"))
    # show_hist_for_dict(os.path.join("Data", "Bonn_128_5_rect_w10_s (copy)", "info_dict.json"))
    show_modified_histograms(os.path.join("Data", "Bonn_128_5_rect_w11_s"))



if __name__ == "__main__":
    main()
