from PIL import Image, ImageFilter, ImageOps, ImageEnhance

import cv2
import numpy as np
from skimage.util import random_noise


def showImage(path):
    orgImg = Image.open(path).convert("RGB")
    orgImg.show()
    return orgImg


def saveImage(path, fileName):
    orgImg = Image.open(path).convert("RGB")
    orgImg.save(fileName)
    return orgImg


# Returns blurred image
def blurImage(path):
    orgImg = Image.open(path).convert("RGB")
    blurredImg = orgImg.filter(ImageFilter.BLUR)
    return blurredImg


# Deblurs Image
def deblurImage(path):
    orgImg = Image.open(path).convert("RGB")
    deblurredImg = orgImg.filter(ImageFilter.SHARPEN)
    return deblurredImg


# Returns gray scaled image
def grayScaleImage(path):
    img = Image.open(path).convert("RGB").convert('L')
    return img


def cropImage(path, left, top, right, bottom):
    im = Image.open(path).convert("RGB")
    imCropped = im.crop((left, top, right, bottom))
    return imCropped


def flipImage(path):
    im = Image.open(path).convert("RGB")
    imFlip = ImageOps.flip(im)
    return imFlip


def mirrorImage(path):
    im = Image.open(path).convert("RGB")
    imMirrored = ImageOps.mirror(im)
    return imMirrored


# Takes theta as the rotation angle
def rotateImage(path, theta):
    im = Image.open(path).convert("RGB")
    imRotated = im.rotate(theta)
    return imRotated


def reverseColorOfImage(path):
    im = Image.open(path).convert("RGB")
    imInverted = ImageOps.invert(im)
    return imInverted


def changeColorBalance(path, rMult, gMult, bMult):
    im = Image.open(path).convert('RGB')
    # Split into 3 channels
    r, g, b = im.split()
    r = r.point(lambda i: i * rMult)
    g = g.point(lambda i: i * gMult)
    b = b.point(lambda i: i * bMult)

    # Recombine back to RGB image
    result = Image.merge('RGB', (r, g, b))
    return result


def adjustBrightness(path, scale):
    im = Image.open(path).convert("RGB")
    brightness = ImageEnhance.Brightness(im).enhance(scale)
    return brightness


def adjustContrast(path, scale):
    im = Image.open(path).convert("RGB")
    contrast = ImageEnhance.Contrast(im).enhance(scale)
    return contrast


def adjustSaturation(path, scale):
    im = Image.open(path).convert("RGB")
    saturationImg = ImageEnhance.Color(im).enhance(scale)
    return saturationImg


def addNoise(path):
    # Load the image
    img = cv2.imread(path)

    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img


def detectEdges(path):
    im = Image.open(path).convert("RGB")
    grayScaleImg = im.convert("L")
    detected = grayScaleImg.filter(ImageFilter.FIND_EDGES)
    return detected
