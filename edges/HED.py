from pathlib import Path
import cv2 as cv
from skimage import filters
import numpy as np
import skimage as sk

# based on https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
class CropLayer(object):
    def __init__(self):
        self.x_start = 0
        self.x_end = 0
        self.y_start = 0
        self.y_end = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
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


def run_network(image):

    # Load the pretrained model (source: https://github.com/s9xie/hed)
    script_path = Path(__file__).parent.absolute()
    net = cv.dnn.readNetFromCaffe(
        str(script_path / "deploy.prototxt"), str(script_path / "hed_pretrained_bsds.caffemodel")
    )
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
    net.setInput(inp)
    out = net.forward()
    cv.dnn_unregisterLayer("Crop")  # get rid of issues when run in a loop
    out = out[0, 0]

    return out


def extract_edges(image):

    image = filters.gaussian(image, sigma=16, truncate=5 / 16, channel_axis=2)

    image = image / np.amax(image) * 255
    image = image.astype(np.uint8)

    edges = run_network(image)

    # gray to binary
    hed_seg = np.ones((edges.shape[0], edges.shape[1]))
    hed_seg[edges < 0.3] = 0

    # skeletonize to get inner lines
    edges = sk.morphology.skeletonize(hed_seg).astype(int)
    # image_edges = sk.morphology.closing(image_edges, footprint=np.ones((5, 5))).astype(int)

    return edges
