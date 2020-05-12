"""
May 12, 2020
Christopher Fichtlscherer (fichtlscherer@mailbox.org)
GNU General Public License

This script creates numpy arrays of circles of circles, which
can be used to train a network.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def create_circle_array(x_size, y_size, center_x, center_y, thickness, radius):
    """creates a binary, two dim numpy array in which a circle has the
        value 1 all other values are 0"""

    x_mesh, y_mesh = np.mgrid[:x_size, :y_size]

    circle = (x_mesh - center_x) ** 2 + (y_mesh - center_y) ** 2

    c_array = np.logical_and(circle < (radius**2 + thickness), circle > (radius**2 - thickness))

    return c_array


n = 10 # the number of arrays we want to create
x_size, y_size = 200, 200 # size of the array
center_x, center_y = 100, 100 # center of the circle
delta_x, delta_y = 20, 20 # center will be random in [center_x - delta_x: center_x + delta_x]

thickness = 100 # thickness of the circle
radius = 30 # radius of the circle
delta_r = 10 # radius will be random in [radius - delta_r: radius + delta_r]


if not os.path.exists("circle_data"):
    os.makedirs("circle_data")


for i in range(n):
    
    cx = center_x + delta_x * (2 * (np.random.random()-1))
    cy = center_y + delta_y * (2 * (np.random.random()-1))
    r = radius + delta_r * (2 * (np.random.random()-1))

    attributes = np.array([cx, cy, r])
    c_array = create_circle_array(x_size, y_size, cx, cy, thickness, r)    

    np.save("circle_data/" + str(i) + "_circle", c_array)
    np.save("circle_data/" + str(i) + "_attributes", attributes)



