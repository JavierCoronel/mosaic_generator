import os
import numpy as np
import edges.classic_edge_extractors as cee
from edges.hed_edge_extractor import extract_edges
from skimage import io, transform


class ImagePreprocessor:
    """Class to open, preprocess, extract edges and save an image"""

    def __init__(self, config_parameters):

        self.config = config_parameters
        self.image_path = config_parameters.image_path
        self.edge_extractor_name = config_parameters.edges

    def read_image(self, preprocess_image=True) -> np.array:
        """Opens an image specified in the ImagePreprocessor

        Parameters
        ----------
        preprocess_image : bool, optional
            Whether to resize the read image, by default True

        Returns
        -------
        np.array
            Array with the image
        """
        assert os.path.isfile(self.image_path), f"Image file does not exist: {self.image_path}"
        image = io.imread(self.image_path)

        if preprocess_image:
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

    def extract_edges(self, image: np.array) -> np.array:
        """Extractes the edges of an image depending on the edge extracting method specified in
        the ImagePreprocessor

        Parameters
        ----------
        image : np.array
            Image array

        Returns
        -------
        np.array
            Extractes edges of the image
        """
        if self.edge_extractor_name == "HED":
            edges = extract_edges(image)

        elif self.edge_extractor_name == "sobel":
            edges = cee.sobel_edges(image)

        elif self.edge_extractor_name == "diblasi":
            edges = cee.diblasi_edges(image)
        else:
            print(f"Edge extraction option {self.edge_extractor_name} not recognized")

        return edges
