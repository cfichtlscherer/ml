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

# READ IN TRAFFIC SIGNS

path = "only_one_red_circle/"
file_list = os.listdir(path)
number_of_images = int(len(file_list) / 2)

image_list = []
value_list = []

np.random.seed(1)                            
                                             
all_image_array = np.arange(number_of_images)
np.random.shuffle(all_image_array)           
                                             

for i in all_image_array:
    image_list += [np.load(path + "image_" + str(i) + ".npy")]
    value_list += [np.load(path + "values_" + str(i) + ".npy")]

# SPLIT THE DATA

split = 800

image_list_training = image_list[:split]
image_list_test = image_list[split:]

value_list_training = value_list[:split]
value_list_test = value_list[split:]

image_training_array = np.asarray(image_list_training)
image_test_array = np.asarray(image_list_test)

value_training_array = np.asarray(value_list_training)
value_test_array = np.asarray(value_list_test)

# DEFINE THE MODEL

model = Sequential()
model.add(MaxPooling2D(pool_size=(4,4),input_shape=(300,300,3)))
model.add(Convolution2D(6, (5,5),activation='relu'))
model.add(Convolution2D(4, (5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten()),
model.add(Dense(16, activation='relu')),
model.add(Dense(16, activation='relu')),
model.add(Dense(4,activation='relu'))
model.build()

model.compile(loss= "mean_squared_error" , optimizer="adam")

# TRAIN THE MODEL

model.fit(image_training_array, value_training_array, batch_size=200, epochs=1000)

model.save('my_model_5.h5')
