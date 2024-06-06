import glob

import numpy as np
import cv2
import os
import pickle

from data_support import create_plot, process_chisqu_file


def compare_images(image1, image2):
    # Resize images to the same dimensions for comparison
    if image2.shape != image1.shape:
        image2 = image2[32:-32, :]

    # Compute the Mean Squared Error (MSE) between the two images
    mse = np.mean((image1 - image2) ** 2)

    # If the MSE is below a certain threshold, consider the images as similar
    return mse < 1


def load_images_from_folder(folder, check_file_number):
    images = []
    paths = {}

    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(('.jpg')):
                # 16 removed because it is the middle image
                if check_file_number and filename[-6:-4] not in ["00", "08", "24", "32"]:
                    continue
                # Skip middle image, it occurs more often
                elif not check_file_number and filename[-6:-4] in ["_2"]:
                    continue
                img_path = os.path.join(root, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    paths[len(images) - 1] = img_path

    return np.array(images), paths


def process_folders(path_folder, model_folder):
    first_folder = os.path.join(path_folder)
    second_folder = os.path.join("..", "All_Path_Imgs", model_folder)

    a1, b1 = load_images_from_folder(first_folder, False)
    a2, b2 = load_images_from_folder(second_folder, True)

    matches = []

    def handle_match_not_equal(text):
        print(f"Match error: {text}")
        for match in matches:
            print(f"\t{b1[match[0]]} - {b2[match[1]]}")

    for i in range(a1.shape[0]):
        match_found = False
        for j in range(a2.shape[0]):
            if compare_images(a1[i], a2[j]):
                if match_found:
                    handle_match_not_equal(f"More than one for {b1[i]}")
                    return []
                matches.append((i, j))
                match_found = True
        if not match_found:
            handle_match_not_equal(f"None for {b1[i]}")
            return []

    path_drusen = []
    for match in matches:
        m1 = b1[match[0]]
        m2 = b2[match[1]]
        path_i = os.path.split(m2)[0][-2:]
        drusen_name = os.path.split(m2)[0]
        drusen_name = os.path.split(drusen_name)[-2]
        drusen_name = os.path.split(drusen_name)[-2]
        drusen_name = os.path.split(drusen_name)[-1]
        m1 = os.path.split(m1)[-1].split(".")[0][:-2]
        m1 = int(m1.split("_")[-1])
        path_drusen.append((path_i, drusen_name, m1))

    line_info = []
    path_folder = path_folder.replace("_", " ")
    for path_i, drusen_name, image_name in path_drusen:
        fname = os.path.join("..", "ChiSqu", model_folder, drusen_name, f"path_{path_i}.txt")
        line_info.append((f"{path_folder}, {image_name}", fname))
    return line_info, [(b1[match[0]], b2[match[1]]) for match in matches]


def main():
    dict_path = 'selected_p_values.pkl'

    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            line_dict = pickle.load(f)
    else:
        name_file_pairs = []
        folder_combinations = []

        folder_combinations.append(("Grow_Multiple",
                                    "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_linear20000_ortho"))

        folder_combinations.append(("Grow_Single",
                                    "Bonn_128_5_rect_w20_s_aug_bce_ws_results_GAN_100epochs_20nz_pde20000_20"))

        folder_combinations.append(("Growth and Merging",
                                    "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_warp50000_20"))

        folder_combinations.append(("Growth_and_Merging",
                                    "Bonn_128_5_rect_w20_s_aug_bce_ws_results_GAN_100epochs_20nz_pde20000_20"))

        folder_combinations.append(("Shift_In_Sizes",
                                    "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_linear20000_ortho"))

        folder_combinations.append(("Lighting",
                                    "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_warp50000_20"))

        # These two are in W space, the values of shift are not interesting to us
        # folder_combinations.append(("Almost_No_Change",
        #                            "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_warp50000_w_20"))

        # folder_combinations.append(("Abrupt_Change",
        #                            "Bonn_128_5_rect_w20_m_aug_ws_results_StyleGAN2_1000epochs_pde20000_w_20"))

        folder_combinations.append(("Mode_Collapse",
                                    "Bonn_128_5_rect_w10_m_aug_results_StyleGAN2_2000epochs_pde20000_20"))

        # folder_combinations = [("Abrupt_Change", os.path.split(f)[-1]) for f in glob.glob(os.path.join("..", "All_Path_Imgs", "*"))]

        matches = []
        for path_folder, model_folder in folder_combinations:
            print(f"{path_folder}, {model_folder}")
            a, b = process_folders(path_folder, model_folder)
            name_file_pairs.extend(a)
            matches.extend(b)
        line_dict = dict()
        for name, file_name in name_file_pairs:
            line_dict[name] = process_chisqu_file(file_name)

        # Dump dict into pickle file, to save p-values for borders of images
        with open(dict_path, 'wb') as f:
            pickle.dump(line_dict, f)

        with open("matches.pkl", "wb") as f:
            pickle.dump(matches,f)

    create_plot(line_dict=line_dict,
                fname="FinalChiSqu.png",
                title=r'p-values of $\chi^2$ test for Images in selected Paths')


if __name__ == "__main__":
    main()
