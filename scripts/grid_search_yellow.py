#!/usr/bin/env python
# coding: utf-8

# Christopher Fichtlscherer, 30.06.2020, fichtlscherer@mailbox.org

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import keras
from keras import Sequential
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Convolution2D,MaxPooling2D,AveragePooling2D,Flatten,Dropout
from keras import optimizers

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os as os

# READ IN TRAFFIC SIGNS & Normalization 1

path = "hand_picked_yellow/"
file_list = os.listdir(path)
number_of_images = int(len(file_list) / 2)

image_list = []
value_list = []

np.random.seed(1)                            
                                             
all_image_array = np.arange(number_of_images)
np.random.shuffle(all_image_array)           
                                             

for i in all_image_array:
    image_list += [np.load(path + "image_" + str(i) + ".npy")]
    value_list += [np.load(path + "values_" + str(i) + ".npy")/300]

# SPLIT THE DATA & Normalization 2

split = 1000

image_list_training = image_list[:split]
value_list_training = value_list[:split]

image_training_array = (np.asarray(image_list_training)/255).astype("float32")
value_training_array = np.asarray(value_list_training)

# Create 8 times more data by flippling and rotating
# flip_image, rotate_image, flip_and_rotate from Balint

def flip_image(image,label):
    im = np.fliplr(image)
    l = [300-label[1] ,300-label[0],label[2],label[3]]
    return im, l

def rotate_image(image,label):
    im = np.rot90(image)
    l = [label[2] ,label[3],300-label[1],300-label[0]]
    return im,l

def flip_and_rotate(image,label,flip=True):
    #if flip=False, then only return a list of the 4 rotated images
    #if flip=True, then return 8 images (rotations+flips)
    output_images=[]
    output_labels=[]
    for k in range(4):
        output_images+=[image]
        output_labels+=[label]
        image,label=rotate_image(image,label)
    if flip:
        image,label=flip_image(image,label)
        for k in range(4):
            output_images+=[image]
            output_labels+=[label]
            image,label=rotate_image(image,label)
    return output_images, output_labels

fliped_and_rotated_images = []
fliped_and_rotated_labels = []

for i in tqdm(range(split)):
    new_im, new_lab = flip_and_rotate(image_training_array[i], value_training_array[i])
    fliped_and_rotated_images += new_im
    fliped_and_rotated_labels += new_lab


all_im = np.asarray(fliped_and_rotated_images)
all_lab = np.asarray(fliped_and_rotated_labels)
# DEFINE THE MODEL

def model_train(first_conv, second_conv):
    model = Sequential()
    model.add(MaxPooling2D(pool_size=(4,4),input_shape=(300,300,3)))
    model.add(Convolution2D(first_conv, (5,5),activation='relu'))
    model.add(Convolution2D(second_conv, (5,5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten()),
    model.add(Dense(16, activation='relu')),
    model.add(Dense(16, activation='relu')),
    model.add(Dense(4,activation='relu'))
    model.build()

    model.compile(loss= "mean_squared_error" , optimizer="adam")
    
    model.fit(image_training_array, value_training_array, batch_size=200, epochs=1000)

    model.save('my_model_' + str(first_conv) + '_' + str(second_conv) + '.h5')

for f in [5,6,7,8]:
    for s in [5,6,7,8]:
        model_train(first_conv, second_conv)
