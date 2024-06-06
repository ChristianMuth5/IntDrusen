import matplotlib.colors
import torch
import os
import numpy as np
from scipy.stats import chi2
import glob
import shutil
import matplotlib.pyplot as plt
from scipy.special import expit, logit


def format_chi_sq(val):
    return f"{val}"


def chi_sq(lc, chi2_dist):
    # We can use x² because mu=0, std=1, so (x-mu)^T * S^-1 * (x-mu) = x^T * S * x = x^T * x
    return 1 - chi2_dist.cdf(np.linalg.norm(lc) ** 2)


def get_chi_squ(lc, path_i, image_in_path_i, chi2dist):
    i = image_in_path_i if image_in_path_i > -1 else lc.shape[1] // 2  # Take middle druse if -1
    lc = lc[path_i][i].detach().numpy()  # lc is tensor of size 20 or 512
    return chi_sq(lc, chi2dist)


def chi2test():
    is_gan = False
    b_size = 500
    nz = 20 if is_gan else 512
    noise = torch.randn(b_size, nz, device="cpu")
    c = chi2(nz)  # Degrees of freedom set to k, as we are not estimating loc or scale
    test_vals = []

    for i in range(noise.shape[0]):
        test_vals.append(chi_sq(noise[i], c))

    test_vals = np.array(test_vals)
    print(np.min(test_vals))


def generate_chi_square_values(out_dir):
    # Create the out_folder
    out_folder = os.path.join(out_dir)
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder)

    # its messy, but works for now, all path folders contain 0000
    folders = glob.glob(os.path.join("..", "All_Path_Imgs", "*0000*"))
    for i, data_folder in enumerate(folders):
        folder_name = os.path.split(data_folder)[-1]
        print(f"{folder_name}, {i + 1}/{len(folders)}")

        out_folder_model = os.path.join(out_folder, folder_name)
        os.makedirs(out_folder_model)

        is_stylegan = "StyleGAN" in data_folder
        c = chi2(512 if is_stylegan else 20)

        data_mode = 1
        if data_mode == 0:
            if "linear" in data_folder:
                drusen_folder = "interesting_paths"
            else:
                drusen_folder = os.path.join("experiments", "complete", "*", "results", "*", "*")
            search_path = os.path.join(data_folder, drusen_folder, "*")
        else:
            search_path = os.path.join(data_folder, "*")

        for druse in glob.glob(search_path):
            drusen_name = os.path.split(druse)[-1]
            # print(f"\t{drusen_name}:")

            out_folder_drusen = os.path.join(out_folder_model, drusen_name)
            os.makedirs(out_folder_drusen)

            # for path in glob.glob(os.path.join(druse, "paths_images", "path_*")):
            #    print(os.path.split(path)[-1])
            # print(f"\t\t{len(glob.glob(os.path.join(druse, 'paths_latent_codes.pt')))}")
            lc = torch.load(os.path.join(druse, 'paths_latent_codes.pt'), map_location=lambda storage, loc: storage)

            for path_i in range(20):
                result = []
                for image_i in range(33):
                    pre = ""
                    if image_i + 1 in [1, 9, 17, 25, 33]:
                        pre = f"{image_i + 1}: "
                    result.append(pre + format_chi_sq(get_chi_squ(lc, path_i, image_i, c)))
                with open(os.path.join(out_folder_drusen, f"path_{path_i:02d}.txt"), "w") as f:
                    f.write("\n".join(result))


def create_plot(line_dict, fname, title, only_return_norm=False):
    x_vals = [i for i in range(1, 34)]
    cm1 = plt.get_cmap('gist_ncar')
    cm2 = plt.get_cmap('gist_gray')
    NUM_COLORS = len(line_dict.keys())
    num2 = 5

    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained', dpi=1200)
    ax.set_yscale('logit')
    ax.yaxis.set_ticks_position('right')

    # Add lines
    la = 0.5
    ls = 'dotted'
    ax.axhline(y=0.05, color='r', linestyle='--', label='Threshold of 0.05')
    ax.axvline(x=17, color='b', linestyle=ls, label='Original Image', alpha=la)
    ax.axvline(x=1, color='g', linestyle=ls, label="Selected Images 1, 9, 25, 33", alpha=la)
    ax.axvline(x=9, color='g', linestyle=ls, alpha=la)
    ax.axvline(x=25, color='g', linestyle=ls, alpha=la)
    ax.axvline(x=33, color='g', linestyle=ls, alpha=la)

    # ax.set_prop_cycle(color=[cm1(1.*i/(NUM_COLORS-num2)) for i in range(NUM_COLORS-num2)]+[cm2(1.*i/num2) for i in range(num2)])
    ax.set_prop_cycle(color=[cm1(1. * i / (NUM_COLORS)) for i in range(NUM_COLORS)])
    for measurement_name, values in line_dict.items():
        ax.plot(x_vals, values, label=measurement_name.replace("_", " "))

    # Create the colorbar
    t = ax.yaxis.get_transform()
    ymin, ymax = ax.get_ylim()
    #print(ymin, ymax)
    f = t.transform_non_affine
    i = t.inverted().transform_non_affine

    def f1(p):
        #print(f"f, {p}")
        r = p - 0.05
        r[r < 0] *= 20
        r[r > 0] += 0.05
        r *= 1.5
        r = expit(r)
        return r

    def i1(v):
        #print(f"i, {v}")
        r = logit(v)
        r /= 1.5
        r[r > 0.05] -= 0.05
        r[r < 0] /= 20
        r += 0.05
        return r

    vals = np.array([ymin, 0.05, 0.5, ymax])

    #print(vals)
    #print(f(vals))
    #print(i(vals))
    #print(i(f(vals)))

    #print(norm(vals))
    #norm = matplotlib.colors.FuncNorm(functions=(i, f), vmin=ymin, vmax=ymax)

    #print(norm(vals))
    #norm = plt.Normalize(vmin=f(ymin), vmax=f(ymax))
    #print(norm(vals))
    norm = matplotlib.colors.FuncNorm(functions=(f, i), vmin=ymin, vmax=ymax)
    if only_return_norm:
        return norm
    #print(norm(vals))

    #print("Creating plot")

    # 0 == ymin
    # 1 == ymax
    # 0.5 = 0.05

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('RdYlGn'), norm=norm)
    sm.set_array([])  # This is needed to generate the colorbar

    cb = plt.colorbar(sm, label='Color mapping of path images', ax=ax, location='right', format='%0.2f',
                      pad=0.01)  # adjust format string as needed
    cb.ax.set_yscale('logit')
    cb.ax.yaxis.set_ticks_position('left')
    cb.set_ticks([])

    # Add labels and title
    ax.set_xlabel('Index of image in the Path')
    ax.set_ylabel('p-value (logit scaling)')
    ax.set_title(title)

    fig.legend(loc='outside lower center', borderaxespad=0.5, fancybox=True, shadow=True, ncol=3)

    ax.grid(axis='y')

    fig.savefig(fname=fname, bbox_inches='tight')
    plt.close(fig)


def process_chisqu_file(fname):
    with open(fname, "r") as f:
        text = f.read()
    text = text.split("\n")
    for i, t in enumerate(text):
        if ":" in t:
            text[i] = float(t.split(" ")[-1])
        else:
            text[i] = float(text[i])
    return np.array(text)


def create_plots(out_folder):
    # generate_chi_square_values(out_folder)
    models = glob.glob(os.path.join(out_folder, "*"))

    for i, model in enumerate(models):
        model_name = os.path.split(model)[-1]
        print(f"{model_name}   {i + 1}/{len(models)}")

        path_vals = dict()
        for druse in glob.glob(os.path.join(model, "*")):
            druse_name = os.path.split(druse)[-1]
            for path in glob.glob(os.path.join(druse, "*")):
                path_i = int(os.path.split(path)[-1].split("_")[-1].split(".")[0])
                if path_i not in path_vals:
                    path_vals[path_i] = {}

                path_vals[path_i][druse_name] = process_chisqu_file(path)

        for p in path_vals:
            create_plot(line_dict=path_vals[p],
                        fname=os.path.join(model, f'{p}.png'),
                        title=f'Chi Squared Values for Path {p}')


def main():
    out_folder = os.path.join("..", "ChiSqu2")
    generate_chi_square_values(out_folder)
    create_plots(out_folder)


if __name__ == "__main__":
    main()
