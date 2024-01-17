import os
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import rawpy
import pandas as pd
import numpy as np


"""
Crop images into parts made of rows and columns (like a grid)
"""


# INIT
data_path = r"C:\Users\YO\OneDrive - UvA\ARW"
csv_path = r"C:\Users\YO\OneDrive - UvA\ARW\OADS_file_names.csv"
save_path = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\cropped_images"

# Original dimensions
og_height = 3672
og_width = 5496

# divide to 6 pieces
parts_h = 2
parts_w = 3


def main():

    # Read file names
    file_names = pd.read_csv(csv_path)

    # Iterate over file name table and crop
    for idx, row in file_names.iterrows():
        img_name = row["filename"]
        img = load_image(img_name)
        crop_image(img, img_name)


def load_image(img_name):
    # Load image
    img_path = os.path.join(data_path, img_name)

    img_raw = rawpy.imread(img_path)
    img = img_raw.postprocess()

    return img


def crop_image(img, img_name):
    # Size of each piece
    crop_h = int(og_height / parts_h)
    crop_w = int(og_width / parts_w)

    # coordinates
    h_beg = 0
    w_beg = 0
    h_end = crop_h
    w_end = crop_w

    count = 0
    for row_idx in range(parts_h):
        w_beg = 0
        w_end = crop_w
        for column_idx in range(parts_w):
            #do the crop
            cropped = img[h_beg:h_end, w_beg:w_end, :]

            #plt.imshow(cropped)
            #plt.show()

            # Save image
            c_img = Image.fromarray(cropped.astype(np.uint8))
            c_img_name = img_name[:-4] + "_crop" + str(count) +".png"#+ ".TIFF"
            c_img_path = os.path.join(save_path, c_img_name)
            c_img.save(c_img_path, "PNG", lossless=True)

            # increment
            w_beg = w_end
            w_end = w_end + crop_w
            count = count + 1
        # increment
        h_beg = h_end
        h_end = h_end + crop_h


main()
