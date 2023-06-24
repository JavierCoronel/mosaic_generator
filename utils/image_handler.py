"""
image_handler.py
Module to open, process and save images.
Copyright (c) 2023 Javier Coronel
"""

import os
import logging
import numpy as np
from skimage import io, transform
from hydra.utils import get_original_cwd

logger = logging.getLogger("__main__." + __name__)


class ImageHandler:
    """Class to open, preprocess and save an image"""

    def __init__(self, config_parameters):

        self.config = config_parameters
        self.image_path = os.path.join(get_original_cwd(), config_parameters.image_path)
        self.resize_image = None or config_parameters.resize_image

    def read_image(self) -> np.array:
        """Opens an image specified in the ImageHandler

        Returns
        -------
        np.array
            Array with the image
        """

        assert os.path.isfile(self.image_path), f"Image file does not exist: {self.image_path}"
        logger.info("Loading image")
        image = io.imread(self.image_path)

        if self.resize_image:
            image = self.preprocess_image(image)

        return image

    def save_image(self, image: np.array, output_path: str):
        """Saves image array to a specified output path

        Parameters
        ----------
        image : np.array
            Image array
        output_path : str
            Path where to save the image
        """
        io.imsave(output_path, image)

    def preprocess_image(self, image_array: np.array, img_resize_factor: float = 0.5) -> np.array:
        """Preprocess an image by rescaling it by a resize factor

        Parameters
        ----------
        image_array : np.array
            Image array
        img_resize_factor : float, optional
            Rescaling factor of the image, by default 0.5

        Returns
        -------
        np.array
            Rescaled image
        """
        resize_factor = img_resize_factor or self.config.resize_factor
        img_width, img_height, _ = image_array.shape
        processed_dimensions = (int(img_width * resize_factor), int(img_height * resize_factor))

        processed_image = transform.resize(image_array, processed_dimensions, anti_aliasing=True)
        processed_image = (processed_image * 255).astype(int)

        return processed_image
