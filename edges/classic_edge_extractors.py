import skimage as sk
from skimage import filters
import numpy as np
from skimage.morphology import disk


def diblasi_edges(image):

    # segment bright areas to blobs
    variance = image.std() ** 2  #  evtl. direkt die std verwenden
    image_seg = np.ones((image.shape[0], image.shape[1]))
    threshold = variance / 4 * 2 * 1
    image_seg[abs(image - image.mean()) > threshold] = 0

    ### 5. Kanten finden
    image_edge = filters.laplace(image_seg, ksize=3)
    image_edge[image_edge != 0] = 1

    return image_edge


def sobel_edges(image):

    edge_sobel = filters.sobel(image)
    binary = edge_sobel > 0.04
    image_edges = sk.morphology.opening(binary, footprint=disk(2).astype(int))
    image_edges = sk.morphology.skeletonize(image_edges).astype(int)

    return image_edges


def preprocess_clasic(image):

    # RGB to gray ("Luminance channel" in Di Blasi)
    image_gray = sk.color.rgb2gray(image)

    # equalize histogram
    image_eq = sk.exposure.equalize_hist(image_gray)

    # soften image
    image_gauss = filters.gaussian(image_eq, sigma=16, truncate=5 / 16)

    return image_gauss
