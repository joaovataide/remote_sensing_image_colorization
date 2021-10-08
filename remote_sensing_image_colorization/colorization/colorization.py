from .settings import img_default_size
import numpy as np

step = 56
window = 224

def make_colorized_rolling_img(img):
    colorized_img = np.zeros(img.shape, dtype=np.float)
    num_imgs = np.zeros(img.shape, np.int)
    x, y = 0, 0
    while y - step + 1 <= img.shape[1] - window:
        while x - step + 1 <= img.shape[0] - window:
            x = min(x + window, img.shape[0]) - window
            y = min(y + window, img.shape[1]) - window
            l_img_window = img[x: x + window, y: y + window]

            colorized_window = colorize_image(l_img_window)
            colorized_img[x: x + window, y: y + window] = (
                colorized_img[x: x + window, y: y + window]*num_imgs[x: x + window, y: y + window]\
                    + colorized_window*1)\
                        /(num_imgs[x: x + window, y: y + window] + 1)
            num_imgs[x: x + window, y: y + window] = num_imgs[x: x + window, y: y + window] + 1

            x += step
        y += step

    return colorized_img



def colorize_image(img):
    if img.size[0]<img_default_size[0]:
        img = img.resize((img_default_size[0], img.size[1]))

    if img.size[1]<img_default_size[1]:
        img = img.resize((img.size[0], img_default_size[1]))

    return make_colorized_rolling_img(img)

