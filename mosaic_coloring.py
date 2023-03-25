"""
Module to extract the color from an image and apply it to the tiles of a mosaic
"""
from pathlib import Path
import os
from typing import List
import numpy as np
from skimage import io
from skimage import draw
from tqdm import tqdm
from sklearn.cluster import KMeans


class MosaicColoring:
    """Class for applying colors to a mosaic"""

    def __init__(self, config_parameters):
        self.coloring_method = config_parameters.get("coloring_method", None)
        self.num_colors = config_parameters.get("num_colors", None)
        self.colormap_path = config_parameters.get("colormap_path", None)

    def kmeans_colors(self, input_image: np.array, num_colors: int = 6):
        """Estimates the kmeans model for the principal colors of an image

        Parameters
        ----------
        input_image : np.array
            Image used to extract the colors
        num_colors : int, optional
            Number of colors to extract, by default 6

        Returns
        -------
            Kmeans estimator
        """

        n_colors = self.num_colors or num_colors
        sample_size = 200000  # reduce sample size to speed up KMeans
        original = input_image[:, :, :3]  # drop alpha channel if there is one
        arr = original.reshape((-1, 3))
        random_indices = np.random.choice(arr.shape[0], size=sample_size, replace=True)
        arr = arr[random_indices, :]
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)

        return kmeans

    def extract_colormap(self, img_path: str):
        """Extracts the principal colors of an image using kmeans and saves the colors to color collections directory

        Parameters
        ----------
        img_path : str
            Path to the image where to extract the colors
        """
        input_image = io.imread(img_path)
        kmeans = self.kmeans_colors(input_image)
        color_centers = kmeans.cluster_centers_.astype(int)
        script_path = Path(__file__).parent.absolute()
        out_path = Path.joinpath(script_path, "data", "color_collections")
        os.makedirs(out_path, exist_ok=True)
        np.save(out_path / os.path.basename(img_path), color_centers)

    def apply_kmeans_to_image(self, image: np.array) -> np.array:
        """Extracts and applies the principal colors of an image using kmeans

        Parameters
        ----------
        image : np.array
            Image used to extact and apply the principal colors

        Returns
        -------
        np.array
            Image with applied kmeans
        """
        width, height, depth = image.shape
        print("Estimating colormap with kmeans")
        kmeans = self.kmeans_colors(image)

        flat_image = image.reshape(width * height, depth)
        labels = kmeans.predict(flat_image)
        codebook = kmeans.cluster_centers_

        kmeans_image = codebook[labels].reshape(width, height, -1).astype(int)

        return kmeans_image

    def get_colors_from_original(self, polygons: List, image: np.array) -> List:
        """Applies colors to a list of polygons, the colors can come from the nearest neighboring pixel of the
        original image or from a modified image with reduced colors using kmeans

        Parameters
        ----------
        polygons : List
            List of polygons
        image : np.array
            Image used to extract the colors

        Returns
        -------
        List
            _description_
        """
        if self.coloring_method == "kmeans":
            image = self.apply_kmeans_to_image(image)

        colors = []
        for polygon in tqdm(polygons):

            x_cord, y_cord = polygon.exterior.xy
            x_list, y_list = draw.polygon(x_cord, y_cord)
            if len(x_list) > 1 and len(y_list) > 1:
                img_cut = image[min(y_list) : max(y_list) + 1, min(x_list) : max(x_list) + 1, :]
                # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
                average = img_cut.mean(axis=0).mean(axis=0)
                color = average / 255
            else:
                if x_cord[0] >= image.shape[1]:
                    x_cord[0] = image.shape[1] - 1
                if y_cord[0] >= image.shape[0]:
                    y_cord[0] = image.shape[0] - 1
                color = image[int(y_cord[0]), int(x_cord[0]), :] / 255

            colors += [color]

        return colors


def modify_colors(colors, variant, colors_collection=None):
    def nearest_color(subjects, query):
        # https://stackoverflow.com/questions/34366981/python-pil-finding-nearest-color-rounding-colors
        return min(subjects, key=lambda subject: sum((s - q) ** 2 for s, q in zip(subject, query)))

    # nearest_color( ((1, 1, 1, "white"), (1, 0, 0, "red"),), (64/255,0,0) ) # example
    new_colors = []
    print("Recoloring...")
    for color in tqdm(colors):
        if variant == "monochrome":
            c_new = nearest_color(((1, 1, 1), (0, 0, 0)), color)  # monochrom
        elif variant == "grayscale":
            c_new = str(
                0.2989 * color[0] + 0.5870 * color[1] + 0.1140 * color[2]
            )  # matplotlib excepts grayscale be strings
        elif variant == "polychrome":
            n_gray = 9
            some_gray = [(g / n_gray, g / n_gray, g / n_gray) for g in range(n_gray + 1)]
            c_new = nearest_color(some_gray, color)  # monochrom
        elif variant == "source":
            c_new = nearest_color(colors_collection / 255, color)
        else:
            raise ValueError("Parameter not understood.")
        new_colors += [c_new]
    return new_colors


def load_colors():
    script_path = Path(__file__).parent.absolute()
    collection_path = Path.joinpath(script_path, "color_collections")
    color_dict = {}
    for color_name in collection_path.glob("*.npy"):
        color_dict[color_name.stem] = np.load(color_name)
    return color_dict


if __name__ == "__main__":

    data_paths = [r"data\input\dalle_4.jpg"]
    color_extractor = MosaicColoring({"coloring_method": "kmeans"})
    for fname in data_paths:
        color_extractor.extract_colormap(fname)
