import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf

def create_captcha(text, shear = 0, size = (100, 24)):
    # L means black and white pixels only
    im = Image.new("L", size, "black")
    draw = ImageDraw.Draw(im)

    font = ImageFont.truetype(r"bretan\Coval-Regular.ttf", 22)
    draw.text((3, -2), text, fill = 1, font = font)

    image = np.array(im)
    affine_tf = tf.AffineTransform(shear = shear)
    image = tf.warp(image, affine_tf)

    return image / image.max()
