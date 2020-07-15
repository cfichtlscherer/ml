import numpy as np                                                  
import matplotlib.pyplot as plt                                     
from tqdm import tqdm                                               
import os as os                                                     
import PIL as PIL                                                     

from shutil import copy

dir_name = "hand_picked"                                            
                                                                    
list_file = os.listdir()                                            
list_file_good = os.listdir("png_version/")

if not os.path.exists(dir_name):                                    
    os.makedirs(dir_name)                                           
              
counter = 0

for ifile in tqdm(list_file_good):
    name_file = ifile[:-3] + "npy"
    value_file = "values" + ifile[5:-3] + "npy"
    copy(name_file, dir_name + "/image_" + str(counter) + ".npy")
    copy(value_file, dir_name + "/values_" + str(counter) + ".npy")
    counter += 1    

