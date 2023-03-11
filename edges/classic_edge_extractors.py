"""
Module with classic methods for edge extraction
"""
import skimage as sk
from skimage.morphology import disk
import numpy as np


def diblasi_edges(image: np.array) -> np.array:
    """
    Apply Di Blasi edge detection algorithm on the input image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        Binary image where edges are marked as 1.

    References
    ----------
    .. [1] Di Blasi, G., 1998. "A new edge detection method based on
        noise reduction". Pattern Recognition Letters, 19(5), pp.417-423.
    """

    preprocessed_image = preprocess_clasic(image)
    # segment bright areas to blobs
    variance = preprocessed_image.std() ** 2
    image_seg = np.ones((preprocessed_image.shape[0], preprocessed_image.shape[1]))
    threshold = variance / 4 * 2 * 1
    image_seg[abs(preprocessed_image - preprocessed_image.mean()) > threshold] = 0

    image_edge = sk.filters.laplace(image_seg, ksize=3)
    image_edge[image_edge != 0] = 1

    return image_edge


def sobel_edges(image: np.array) -> np.array:
    """
    Apply Sobel edge detection algorithm on the input image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        Binary image where edges are marked as 1.

    References
    ----------
    .. [1] Sobel, I., 1990. "An isotropic 3x3 image gradient operator".
        Machine Vision for Three-Dimensional Scenes, pp. 376-379.
    """
    preprocessed_image = preprocess_clasic(image)
    edge_sobel = sk.filters.sobel(preprocessed_image)
    binary = edge_sobel > 0.04
    image_edges = sk.morphology.opening(binary, footprint=disk(2).astype(int))
    image_edges = sk.morphology.skeletonize(image_edges).astype(int)

    return image_edges


def preprocess_clasic(image: np.array) -> np.array:
    """
    Apply classic preprocessing steps on the input image.

    markdown
    Copy code
    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    ndarray
        Preprocessed image.
    """
    # RGB to gray ("Luminance channel" in Di Blasi)
    image_gray = sk.color.rgb2gray(image)

    # equalize histogram
    image_eq = sk.exposure.equalize_hist(image_gray)

    # soften image
    preprocessed_image = sk.filters.gaussian(image_eq)

    return preprocessed_image
