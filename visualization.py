import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.animation import FuncAnimation
import multiprocessing as mp
from functools import partial
from PIL import Image


def fig2image(fig):
    fig.canvas.draw()
    #(w, h) = fig.canvas.get_width_height()
    im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    #im = Image.fromarray(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4)))
    plt.close()
    return im


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

            # original druse
            druse = glob.glob(os.path.join(drusen[drusen_index], "original_image.jpg"))[0]
            image = mpimg.imread(druse)
            ax = axes[plt_row][0]
            ax.imshow(image, cmap='gray')
            ax.axis('off')

            for path_i in range(10):
                plt_col = path_i + 1
                path_index = path_indices[path_i]
                # path image
                druse_path = sorted(glob.glob(os.path.join(drusen[drusen_index], "paths_images",
                                                           f"path_{path_index:03d}", "*.jpg")))[path_image_i]
                image = mpimg.imread(druse_path)
                ax = axes[plt_row][plt_col]
                ax.imshow(image, cmap='gray')
                ax.set_title(f"Path {path_index:03d}")
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
            if i < len(drusen_indices):
                druse = glob.glob(os.path.join(drusen[drusen_indices[i]], "original_image.jpg"))[0]
                image = mpimg.imread(druse)
                ax.imshow(image, cmap='gray')
            ax.axis('off')

            # path image
            ax = axes[plt_row + 1][plt_col]
            if i < len(drusen_indices):
                fn = drusen[drusen_indices[i]]
                files = sorted(glob.glob(os.path.join(fn, "paths_images", f"path_{path_i:03d}", "*.jpg")))
                druse_path = files[path_image_i]
                image = mpimg.imread(druse_path)
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


def generate_gif(data):
    if data[0] == 1:
        show_20drusen_for_one_path(data)
    elif data[0] == 2:
        show_10paths_for_5drusen(data)


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


def generate_gifs_linear(folder):
    drusen_dir = [os.path.join(folder, "paths")]
    generate_gifs(folder, drusen_dir)


def show_all_drusen(folder):
    files = glob.glob(os.path.join(folder, "*"))
    #data = [[files[i:i+50] for i in range()]]


def main():
    folder = os.path.join("Data", "Bonn_128_5_rect_w10_s_s=100_aug_bce_c=hw_results_GAN_200epochs_20nz_pde50000_20")
    #generate_gifs_warp(folder)
    #folder = os.path.join("Data", "Bonn_128_5_rect_w10_s_aug_bce_c=hw_results_GAN_200epochs_20nz_linear10000_ortho")
    #generate_gifs_linear(folder)
    #show_all_drusen(os.path.join("Data", "Bonn_128_5_rect_w05", "1"))
    generate_gifs_warp(folder)


if __name__ == "__main__":
    main()
