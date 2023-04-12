"""
Module to extract the edges of an image using Holistically-Nested Edge Detection
"""
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import skimage as sk
from skimage import filters


# based on https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
class CropLayer(object):
    """Modified version of a Crop Layer in OpenCV"""

    def __init__(self, params, blobs):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch_size, num_channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]
        self.y_start = int((input_shape[2] - target_shape[2]) / 2)
        self.x_start = int((input_shape[3] - target_shape[3]) / 2)
        self.y_end = self.y_start + height
        self.x_end = self.x_start + width
        return [[batch_size, num_channels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y_start : self.y_end, self.x_start : self.x_end]]


def load_network():
    """Load the pretrained HED model (if files not available, download from source: https://github.com/s9xie/hed)"""
    #
    script_path = Path(__file__).parent.absolute()
    proto_txt_file = os.path.join(script_path, "deploy.prototxt")
    assert os.path.isfile(
        proto_txt_file
    ), f"{proto_txt_file} does not exist, download from source https://github.com/s9xie/hed"

    pretrained_model_file = os.path.join(script_path, "hed_pretrained_bsds.caffemodel")
    assert os.path.isfile(
        pretrained_model_file
    ), f"{pretrained_model_file} does not exist, download from source https://github.com/s9xie/hed"

    network = cv.dnn.readNetFromCaffe(proto_txt_file, pretrained_model_file)
    return network


def run_hed_network(image):
    """Runs the HED network for edge extraction on an image

    Parameters
    ----------
    image : np.array
        Image to use as input for the network

    Returns
    -------
    output: np.array
        Output image from the HED model
    """
    cv.dnn_registerLayer("Crop", CropLayer)

    # prepare image as input dataset (mean values from full image dataset)
    inp = cv.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(image.shape[1], image.shape[0]),  # w,h
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )

    network = load_network()
    network.setInput(inp)
    output = network.forward()
    cv.dnn_unregisterLayer("Crop")  # get rid of issues when run in a loop
    output = output[0, 0]

    return output


def preprocess_image(image):
    """Preprocess an image for edge extraction using HED

    Parameters
    ----------
    image : np.array
        Image to preprocess

    Returns
    -------
    preprocessed_image : np.array
        Preprocessed image
    """
    image = filters.gaussian(image, channel_axis=2)
    image = image / np.amax(image) * 255
    image = image.astype(np.uint8)
    preprocessed_image = cv.resize(image, (image.shape[1], image.shape[0]))

    return preprocessed_image


def extract_edges(image):
    """Extracts the edges of an image using HED. It also applies a threshold and skeletonization to extcract
    the stronger edges.

    Parameters
    ----------
    image : np.array
        Image used to extract te edges

    Returns
    -------
    edges : np.array
        Image with the stronger edges
    """
    input_image = preprocess_image(image)
    image_edges = run_hed_network(input_image)

    hed_seg = np.ones((image_edges.shape[0], image_edges.shape[1]))
    hed_seg[image_edges < 0.3] = 0

    edges = sk.morphology.skeletonize(hed_seg).astype(int)

    return edges
