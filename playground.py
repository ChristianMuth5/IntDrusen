import glob
import os
import filecmp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def main():
    #check_folder_equals()
    generate_overview(os.path.join("Data", "Bonn_128_5_RBL", "1_ov"))


if __name__ == "__main__":
    main()
