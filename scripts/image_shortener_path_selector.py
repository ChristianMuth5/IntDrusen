import numpy as np
from PIL import Image
import os
import glob


def crop_images(filepaths, do_crop=True, do_remove=False, num=0):
    for filepath in filepaths:
        n = 0
        if filepath.endswith("000000.jpg"):
            n = 0
        elif filepath.endswith("000008.jpg"):
            n = 1
        elif filepath.endswith("000016.jpg"):
            n = 2
        elif filepath.endswith("000024.jpg"):
            n = 3
        elif filepath.endswith("000032.jpg"):
            n = 4
        else:
            continue
        try:
            # Open image
            img = Image.open(filepath)

            # Convert image to numpy array
            img_array = np.array(img)

            # Crop image array
            if do_crop:
                img_array = img_array[32:-32, :]

            # Convert numpy array back to image
            cropped_img = Image.fromarray(img_array)

            # Save the cropped image, overwriting the original file
            cropped_img.save(f"Image_{num}_{n}.jpg")

            print(f"Cropped and saved {os.path.basename(filepath)} successfully.")
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}:", e)

    if do_remove:
        for filepath in filepaths:
            os.remove(filepath)



# Example usage
filepaths = glob.glob(os.path.join("Data", "*.j*"))  # List of filepaths to your images
crop_images(filepaths, True,  True,3)