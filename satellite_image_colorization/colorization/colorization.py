from .settings import img_default_size


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization,LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from PIL import Image


WIDTH=224
HEIGHT=224




def make_colorized_rolling_img(img):
    return img



def colorize_image(img):
    if img.size[0]<img_default_size[0]:
        img = img.resize((img_default_size[0], img.size[1]))

    if img.size[1]<img_default_size[1]:
        img = img.resize((img.size[0], img_default_size[1]))

    return make_colorized_rolling_img(img)

