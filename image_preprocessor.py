from edges.HED import extract_edges
import edges.classic_edge_extractors as cee
from skimage import io, transform
import os


class ImagePreprocessor:
    def __init__(self, config_parameters):

        self.config = config_parameters
        self.image_path = config_parameters.image_path
        self.edge_extractor_name = config_parameters.edges

    def read_image(self, preprocess_image=True):
        assert os.path.isfile(self.image_path), f"Image file does not exist"
        image = io.imread(self.image_path)

        if preprocess_image:
            image = self.preprocess_image(image)

        return image

    def save_image(self, image, output_path):

        io.imsave(output_path, image)

    def preprocess_image(self, image_array, img_resize_factor=0.5):

        resize_factor = img_resize_factor or self.config.resize_factor
        img_width, img_height, _ = image_array.shape
        processed_dimensions = (int(img_width * resize_factor), int(img_height * resize_factor))

        processed_image = transform.resize(image_array, processed_dimensions, anti_aliasing=True)
        processed_image = (processed_image * 255).astype(int)

        return processed_image

    def extract_edges(
        self,
        image,
    ):

        if self.edge_extractor_name == "HED":
            edges = extract_edges(image)

        if self.edge_extractor_name != "HED":
            image = cee.preprocess_clasic(image)

            if self.edge_extractor_name == "sobel":
                edges = cee.sobel_edges(image)

            if self.edge_extractor_name == "diblasi":
                edges = cee.diblasi_edges(image)

        return edges

    def postprocess_image():
        return 1
