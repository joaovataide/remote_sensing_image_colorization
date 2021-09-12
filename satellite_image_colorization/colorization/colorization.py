def make_colorized_rolling_img(img):
    return img

def make_colorized_image(img):
    return img


def colorize_image(img):
    if max(img.size)<300:
        img.resize()
        return make_colorized_image(img)
    else:
        make_colorized_rolling_img(img)

