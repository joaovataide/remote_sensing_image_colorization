import os
# import cv2
from PIL import Image
import matplotlib.pyplot as plt


ImagePath='D:/Material de estudo/Engenharia Cartográfica/PFC/Prof/bgr_rend.tif'


# img = cv2.imread(ImagePath)
# cv2.imread(ImagePath, 0)
img = plt.imread(ImagePath)
img = img[:, :, :3]
print(img)
# img = Image.open(ImagePath)

y_max, x_max, _ = img.shape

y_max
x_max
# plt.imshow(img[0: 0+300, 500: 500 + 300])
x_max
x_step = 61
y_step = 61
window = 224

for i in range(0, x_max-window, x_step):
    for j in range(0, y_max-window, y_step):
        cut_img = Image.fromarray(img[j:j+window, i:i+window])
        cut_img.save('D:/pfc/satellite_image_colorization/data/wv2/training/{}_{}.jpg'.format(j, i))