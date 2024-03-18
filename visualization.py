import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from functools import partial
from PIL import Image
from scipy.stats import chi2
import torch


def fig2image(fig):
    fig.canvas.draw()
    #(w, h) = fig.canvas.get_width_height()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #im = Image.fromarray(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4)))
    plt.close()
    return im


def get_chi_squ(drusen_folder, path_i, image_in_path_i):
    lc = torch.load(os.path.join(drusen_folder, "paths_latent_codes.pt"), map_location=lambda storage, loc: storage)
    i = image_in_path_i if image_in_path_i > -1 else lc.shape[1]//2  # Take middle druse if -1
    lc = lc[path_i][i].detach().numpy()  # lc is tensor of size 20
    c = chi2(int(lc.shape[0]))  # Degrees of freedom set to k, as we are not estimating loc or scale
    # We can use x² because mu=0, std=1, so (x-mu)^T * S^-1 * (x-mu) = x^T * S * x = x^T * x
    return f"{1-c.cdf(np.linalg.norm(lc)**2):.03f}"


def show_10paths_for_5drusen(data):
    folder = data[1]
    drusen = data[2]
    path_indices = data[3]
    drusen_indices = data[4]

    def animation_function(path_image_i):
        fig, axes = plt.subplots(5, 11, figsize=(22, 10))
        for i in range(5):
            plt_row = i
            drusen_index = drusen_indices[i]
            drusen_folder = drusen[drusen_index]

            # original druse
            druse = glob.glob(os.path.join(drusen_folder, "original_image.jpg"))[0]
            image = mpimg.imread(druse)
            ax = axes[plt_row][0]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Druse {drusen_index}, Ch2={get_chi_squ(drusen_folder, 0, -1)}")
            ax.axis('off')

            for path_i in range(10):
                plt_col = path_i + 1
                path_index = path_indices[path_i]
                # path image
                druse_path = sorted(glob.glob(os.path.join(drusen_folder, "paths_images",
                                                           f"path_{path_index:03d}", "*.jpg")))[path_image_i]
                image = mpimg.imread(druse_path)
                ax = axes[plt_row][plt_col]
                ax.imshow(image, cmap='gray')
                ax.set_title(f"Path {path_index}, Ch2={get_chi_squ(drusen_folder, path_index, path_image_i)}")
                ax.axis('off')
        plt.tight_layout()
        return fig2image(fig)

    images = []
    for i in range(33):
        images.append(animation_function(i))
    #anim_created = FuncAnimation(fig, animation_function, frames=33, interval=125)

    drusen_index = drusen_indices[0] // 5
    path_index = path_indices[0] // 10
    fname = os.path.join(folder, f"DrusenOverview_Drusen{drusen_index:02d}_Paths{path_index:02d}.gif")

    #anim_created.save(os.path.join(folder, fname), writer='imagemagick', fps=8)
    images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=125, loop=0)


def show_20drusen_for_one_path(data):
    folder = data[1]
    drusen = data[2]
    path_i = data[3]
    drusen_indices = data[4]

    def animation_function(path_image_i):
        fig, axes = plt.subplots(5, 10, figsize=(20, 10))
        for i in range(20):
            plt_col = i % 10
            plt_row = 0 if i < 10 else 3

            # original druse
            ax = axes[plt_row][plt_col]
            drusen_index = drusen_indices[i]
            drusen_folder = drusen[drusen_index]
            if i < len(drusen_indices):
                druse = glob.glob(os.path.join(drusen_folder, "original_image.jpg"))[0]
                image = mpimg.imread(druse)
                ax.imshow(image, cmap='gray')
                ax.set_title(f"Druse {drusen_index}, Ch2={get_chi_squ(drusen_folder, 0, -1)}")
            ax.axis('off')

            # path image
            ax = axes[plt_row + 1][plt_col]
            if i < len(drusen_indices):
                fn = drusen[drusen_indices[i]]
                files = sorted(glob.glob(os.path.join(fn, "paths_images", f"path_{path_i:03d}", "*.jpg")))
                druse_path = files[path_image_i]
                image = mpimg.imread(druse_path)
                ax.set_title(f"Path {path_i}, Ch2={get_chi_squ(drusen_folder, path_i, path_image_i)}")
                ax.imshow(image, cmap='gray')
            ax.axis('off')
        # remove center line
        for i in range(10):
            ax = axes[2][i]
            ax.axis('off')
        plt.tight_layout()
        return fig2image(fig)

    images = []
    for i in range(33):
        images.append(animation_function(i))
    #anim_created = FuncAnimation(fig, animation_function, frames=33, interval=125)

    drusen_index = drusen_indices[0] // 20
    fname = os.path.join(folder, f"PathOverview_Path{path_i:03d}_Drusen{drusen_index:02d}.gif")
    #anim_created.save(os.path.join(folder, fname), writer='imagemagick', fps=8)
    images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=125, loop=0)


def get_category(drusen_dir):
    return os.path.split(drusen_dir)[1].split('_')[0]


def show_10drusen_for_5paths(data):
    folder = data[1]
    drusen_dirs = data[2]
    path_indices = data[3]
    nrow = len(path_indices)+1
    ncol = len(drusen_dirs)

    def animation_function(path_image_i):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
        categories = [get_category(d) for d in drusen_dirs]
        for i in range(ncol):
            drusen_dir = drusen_dirs[i]
            for j in range(nrow):
                ax = axes[j][i]
                ax.axis('off')

                if j == 0:  # Original Druse
                    if i == 0 or categories[i] != categories[i-1]:  # Set title for first or changing
                        ax.set_title(categories[i])
                    druse = os.path.join(drusen_dir, "original_image.jpg")
                    image = mpimg.imread(druse)
                    ax.imshow(image, cmap='gray')
                else:  # Path Image
                    files = sorted(glob.glob(os.path.join(drusen_dir, "paths_images", f"path_{path_indices[j-1]:03d}", "*.jpg")))
                    druse_path = files[path_image_i]
                    image = mpimg.imread(druse_path)
                    if i == 0:
                        ax.set_title(f"Path {path_indices[j-1]}")
                    ax.imshow(image, cmap='gray')

        plt.tight_layout()
        return fig2image(fig)

    images = []
    for i in range(33):
        images.append(animation_function(i))
    #anim_created = FuncAnimation(fig, animation_function, frames=33, interval=125)

    #fname = os.path.join(folder, f"Drusen_{data[4]} Paths_{path_indices}.gif")
    fname = os.path.join(folder, f"Paths {', '.join([str(x) for x in path_indices])}.gif")
    #anim_created.save(os.path.join(folder, fname), writer='imagemagick', fps=8)
    images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=125, loop=0)



def generate_gif(data):
    if data[0] == 1:
        show_20drusen_for_one_path(data)
    elif data[0] == 2:
        show_10paths_for_5drusen(data)
    elif data[0] == 3:
        show_10drusen_for_5paths(data)


def generate_gifs_int(folder, drusen_dir, paths):
    drusen_all = sorted([os.path.join(drusen_dir[0], x) for x in glob.glob1(drusen_dir[0], "*")])
    drusen = []
    categories = ["Reticular Pseudo Drusen", "Drusenoid PED", "Small hard Drusen", "Large soft Drusen", "Other"]
    for c in categories:
        for d in drusen_all:
            if c == os.path.split(d)[1].split('_')[0]:
                drusen.append(d)
    drusen_count = len(drusen)
    if drusen_count == 0:
        return
    drusen = np.asarray(drusen)

    directions = len(glob.glob1(os.path.join(drusen_all[0], "paths_images"), "path_*"))
    if len(paths) == 0:  # take all
        paths = [i for i in range(directions)]
    path_count = len(paths)
    paths = np.asarray(paths)

    data = []
    folder = os.path.join(folder, "interesting_gifs")
    os.makedirs(folder, exist_ok=True)
    for i in range(0, drusen_count, drusen_count):
        ind_drusen = [j for j in range(i, min(i+drusen_count, drusen_count), 1)]
        for j in range(0, len(paths), 5):
            ind_paths = [k for k in range(j, min(j+5, path_count), 1)]
            data.append((3, folder, drusen[ind_drusen], paths[ind_paths], i // 10, j // 5))

    pool = mp.Pool(mp.cpu_count())
    partial_process_item = partial(generate_gif)
    pool.map(partial_process_item, data)
    pool.close()
    pool.join()
    return


def generate_gifs(folder, drusen_dir):
    drusen = sorted([os.path.join(drusen_dir[0], x) for x in glob.glob1(drusen_dir[0], "*")])
    drusen_count = len(drusen)
    directions = len(glob.glob1(os.path.join(drusen[0], "paths_images"), "path_*"))

    times = drusen_count // 20
    times += 1 if drusen_count % 20 else 0
    data1 = []
    for j in range(times):
        indices = [x for x in range(j*20, min((j+1)*20, drusen_count))]
        data1 += [(1, folder, drusen, i, indices) for i in range(directions)]

    drusen_times = drusen_count // 5
    drusen_times += 1 if drusen_count % 5 else 0
    path_times = directions // 10
    path_times += 1 if directions % 5 else 0
    data2 = []
    for i in range(drusen_times):
        drusen_indices = [x for x in range(i*5, min((i+1)*5, drusen_count))]
        for j in range(path_times):
            path_indices = [x for x in range(j*10, min((j+1)*10, directions))]
            data2 += [(2, folder, drusen, path_indices, drusen_indices)]

    data = data1 + data2

    # It could be done even faster by having each matplotlib figure be a data entry and having the stitching be
    # its own multiprocessing, but it takes less than a minute so not important for now, also though it is
    # more time efficient, it uses more space as all images would need to be saved.
    pool = mp.Pool(mp.cpu_count())
    partial_process_item = partial(generate_gif)
    pool.map(partial_process_item, data)
    pool.close()
    pool.join()


# Works for folder structure generated by WarpedGANSpace and PDETraversal
def generate_gifs_warp(folder):
    drusen_dir = glob.glob(os.path.join(folder, "experiments", "complete", "*", "results", "*", "*"))
    generate_gifs(folder, drusen_dir)


def generate_gifs_warp_int(folder, paths):
    drusen_dirs = glob.glob(os.path.join(folder, "experiments", "complete", "*", "results", "*"))
    for drusen_dir in drusen_dirs:
        generate_gifs_int(folder, glob.glob(os.path.join(drusen_dir, "*")), paths)


def generate_gifs_linear(folder):
    drusen_dir = [os.path.join(folder, "paths")]
    generate_gifs(folder, drusen_dir)


def generate_gifs_linear_int(folder, paths):
    drusen_dir = [os.path.join(folder, "interesting_paths")]
    generate_gifs_int(folder, drusen_dir, paths)


def analyze_latent_codes(image_in_path_i = -1, path_i = 0):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    names = ["warp_z", "warp_w", "linear"]
    for name_i in range(len(names)):
        name = names[name_i]
        print(name)
        path = os.path.join("Data", "latent_vector_test", name + ".pt")
        lc = torch.load(path, map_location=lambda storage, loc: storage)
        if lc.shape[1] == 512:
            lc = lc[0].detach().numpy()
        else:
            i = image_in_path_i if image_in_path_i > -1 else lc.shape[1]//2  # Take middle druse if -1
            lc = lc[path_i][i].detach().numpy()  # lc is tensor of size 20
        print("Shape:", lc.shape)
        print("Norm:", np.linalg.norm(lc))
        print("Abs Min:", np.min(np.abs(lc)))
        print("Abs Max:", np.max(np.abs(lc)))
        print("Abs Sum:", np.sum(np.abs(lc)))
        c = chi2(int(lc.shape[0]))  # Degrees of freedom set to k, as we are not estimating loc or scale
        # We can use x² because mu=0, std=1, so (x-mu)^T * S^-1 * (x-mu) = x^T * S * x = x^T * x
        c_val = f"{1-c.cdf(np.linalg.norm(lc)**2):.03f}"
        print("Chi Squared value:", c_val)
        print()
        ax = axes[name_i]
        ax.hist(lc, bins=20, range=(-4, 4))
        ax.set_title(f"{name}")
    plt.savefig(os.path.join("Data", "latent_vector_test", "Comp.jpg"))
    plt.close()

def show_5paths_for_5drusen(data):
    folder = data[0]
    drusen_is = data[1]



def main():
    folder = os.path.join("Data", "Bonn_128_5_rect_w20_s_aug_results_StyleGAN2_1000epochs_linear4000_w_ortho")
    #generate_gifs_warp(folder)
    #folder = os.path.join("Data", "Bonn_128_5_rect_w10_s_aug_bce_c=hw_results_GAN_200epochs_20nz_linear10000_ortho")
    #generate_gifs_linear(folder)
    #show_all_drusen(os.path.join("Data", "Bonn_128_5_rect_w05", "1"))
    #generate_gifs_warp(folder)
    #analyze_latent_codes()
    #generate_gifs_warp_int(os.path.join("Data", "Bonn_128_5_rect_w20_s_aug_bce_ws_results_GAN_100epochs_20nz_warp100000_20"), [])
    #generate_gifs_warp_int(os.path.join("Data", "Bonn_128_5_rect_w20_s_aug_bce_ws_results_GAN_100epochs_20nz_warp100000_20"), [1,4,5,9,12])
    generate_gifs_linear(folder)


if __name__ == "__main__":
    main()
