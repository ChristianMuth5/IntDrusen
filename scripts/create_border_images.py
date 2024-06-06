import pickle
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.special import logit, expit

from data_support import create_plot


def border_image(folder_in, folder_out, fname, color_map, p_value, f):
    f_in = os.path.join(folder_in, fname)
    if not os.path.exists(f_in):
        f_in = os.path.join(folder_in.replace('_', ' '), fname)
    f_out = os.path.join(folder_out, fname)
    width = 8

    image = cv2.imread(f_in, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    v = f(p_value)
    c = np.array(color_map(v)[:3]) * 255
    #print(f_in, p_value, v, c)

    #print(f_in)
    #print(p_value, expit(p_value))

    image_out = np.zeros((image.shape[0]+width, image.shape[1], image.shape[2]))
    image_out[width:image.shape[0]+width, 0:image.shape[1], :] = image
    image_out[0:width, 0:image.shape[1]] = c

    image_out = np.float32(image_out)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f_out, image_out)


def main():
    colored_folder = os.path.join("bordered")
    if os.path.exists(colored_folder):
        shutil.rmtree(colored_folder)
    os.makedirs(colored_folder)

    with open('selected_p_values.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    cm = plt.get_cmap('RdYlGn')
    #f = create_plot(loaded_dict, "", "", True)

    def f(val):
        res = 0
        if 0.001 <= val < 0.01:
            res = 1/3
        elif 0.01 <= val < 0.05:
            res = 2/3
        elif 0.05 <= val:
            res = 1
        print(val, res)
        return res*255

    for name, value in loaded_dict.items():
        folder_in = os.path.join(name.split(', ')[0].replace(' ', '_'))
        folder_out = os.path.join(colored_folder, folder_in)
        os.makedirs(folder_out, exist_ok=True)

        f_index = name.split(', ')[1]
        for i in range(5):
            #print(value)
            border_image(folder_in, folder_out, f"Image_{f_index}_{i}.jpg", cm, value[i*8], f)



if __name__ == "__main__":
    main()



