from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU, Input, concatenate
from tensorflow.keras.models import Model
import cv2, os
import numpy as np

def defineFlow(in_):
    model_ = Conv2D(16, (3, 3),padding='same',strides=1)(in_)
    model_ = LeakyReLU()(model_)

    #model_ = Conv2D(64, (3, 3), activation='relu',strides=1)(model_)
    model_ = Conv2D(32, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2),padding='same')(model_)
    
    model_ = Conv2D(64, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    model_ = MaxPooling2D(pool_size=(2, 2),padding='same')(model_)
    
    model_ = Conv2D(128, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = Conv2D(256, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(128, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = UpSampling2D((2, 2))(model_)
    model_ = Conv2D(64, (3, 3), padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    #model_ = BatchNormalization()(model_)
    
    concat_ = concatenate([model_, in_]) 
    
    model_ = Conv2D(64, (3, 3), padding='same',strides=1)(concat_)
    model_ = LeakyReLU()(model_)
    model_ = BatchNormalization()(model_)
    
    model_ = Conv2D(32, (3, 3),padding='same',strides=1)(model_)
    model_ = LeakyReLU()(model_)
    #model_ = BatchNormalization()(model_)
    
    model_ = Conv2D(2, (3, 3), activation='tanh',padding='same',strides=1)(model_)

    return model_


def makeModel(input_shape_tuple, optimizer, loss):
    input_sample = Input(shape=input_shape_tuple)
    output = defineFlow(input_sample)

    model = Model(inputs=input_sample, outputs=output)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def extractInput(dataset_path, batch_size, dimensions, limit=None):
    img_files = os.listdir(dataset_path)
    dataset_size = len(img_files)

    width, height = dimensions

    img_pos = 0
    cycles_counter = 0
    stop_signal = False

    while not stop_signal:
        imgs_l = []
        imgs_ab = []
        for _ in range(batch_size):
            file_name = img_files[img_pos]
            img = cv2.imread(os.path.join(dataset_path, file_name))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
            img = img.astype(np.float32)
            img_lab_rs = cv2.resize(img_lab, (width, height)) # resize image to network input size
            img_l = img_lab_rs[:,:,0] # pull out L channel
            img_ab = img_lab_rs[:,:,1:] # Extracting the ab channel
            img_l = img_l - 50
            img_l = img_l/100
            img_ab = img_ab/110
            img_ab = img_ab - 1
            imgs_l.append(img_l.reshape((width, height, 1)))
            imgs_ab.append(img_ab.reshape((width, height, 2)))

            img_pos += 1
            if img_pos>=dataset_size:
                cycles_counter += 1
                img_pos = img_pos % dataset_size
        yield np.array(imgs_l), np.array(imgs_ab)

        if (limit is not None) and (img_pos + dataset_size*cycles_counter>=limit):
            stop_signal = True