import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os as os

list_file = os.listdir()

dir_name = "png_version"

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

for image_file in tqdm(list_file):
    if "image" in image_file:
        im = np.load(image_file)
        plt.imshow(im)
        plt.savefig(dir_name + "/" + image_file[:-4] + ".png")
        plt.close()


#plt.imshow(im)
