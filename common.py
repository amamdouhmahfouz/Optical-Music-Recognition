import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt



import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,binary_opening, skeletonize, thin
from skimage.measure import find_contours
from skimage.draw import rectangle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu, threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, binary_closing
from skimage.color import label2rgb
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.transform import rotate
from skimage import img_as_ubyte
import imutils
import pickle
import math
from skimage.transform import rescale, resize, downscale_local_mean

import cv2
import os

import sys
import time
import warnings


# Templates directories
quarter_dir = "templates/quarter"
beam_dir = "templates/beam"
time_44_dir = "templates/4-4-Time"
half_dir = "templates/half"

# Source: PrintedMusicSymbolDataset, Note:images in dataset are cropped to get the templates
templates = {
    "quarter": [cv2.imread(quarter_dir+"/"+file,0) for _, _, files in os.walk(quarter_dir) for file in files if ".png" in file.lower() or ".jpg" in file.lower() or ".jpeg" in file.lower() or ".bmp" in file.lower()],
    "beam": [cv2.imread(beam_dir+"/"+file,0) for _, _, files in os.walk(beam_dir) for file in files if ".png" in file.lower() or ".jpg" in file.lower() or ".jpeg" in file.lower() or ".bmp" in file.lower()],
    "4-4-Time": [cv2.imread(time_44_dir+"/"+file,0) for _, _, files in os.walk(time_44_dir) for file in files if ".png" in file.lower() or ".jpg" in file.lower() or ".jpeg" in file.lower() or ".bmp" in file.lower()],
    "half": [cv2.imread(half_dir+"/"+file,0) for _, _, files in os.walk(half_dir) for file in files if ".png" in file.lower() or ".jpg" in file.lower() or ".jpeg" in file.lower() or ".bmp" in file.lower()]
}

# all possible labels that we can detect
symbols_labels = {
    "Thirty-Two-Note": "/32",
    "Eighth-Note": "/8",
    "2-4-Time": '\meter<"2/4">', # TODO: check if <4/2> not <2/4>
    #"G-Clef": None, #not written as a score
    "Sixteenth-Note": "/16",
    "4-4-Time": '\meter<"4/4">',
    "Flat": "&",
    "Sharp": "#",
    "Whole-Note": "/1",
    "Half-Note": "/2",
    "Dot": ".",
    "Natural": "",
    "Quarter-Note": "/4",
    "Double-Sharp": "##",
    "Barline": ""

}


# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12,8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X,Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()

def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)

    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))

    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

def uint8_255(im):
    im_new = im.copy()
    if im.max() <= 1:
        im_new *= 255
        im_new = im_new.astype('uint8')
    else:
        im_new = im_new.astype('uint8')

    return im_new
