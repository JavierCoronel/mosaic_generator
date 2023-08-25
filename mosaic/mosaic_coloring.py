"""
mosaic_coloring.py
Module to extract the colors of an image and apply themto the tiles of a mosaic.
Copyright (c) 2023 Javier Coronel
"""
from pathlib import Path
import os
import logging
from typing import List
import numpy as np
from skimage import io
from skimage import draw
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

logger = logging.getLogger("__main__." + __name__)


class MosaicColoring:
    """Class for applying colors to a mosaic"""

    def __init__(self, config_parameters):
        self.config_params = config_parameters
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
        out_path = Path.joinpath(script_path, "..", "data", "color_collections")
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
        logger.info("Estimating colormap for image with kmeans")
        kmeans = self.kmeans_colors(image)

        flat_image = image.reshape(width * height, depth)
        labels = kmeans.predict(flat_image)
        codebook = kmeans.cluster_centers_

        kmeans_image = codebook[labels].reshape(width, height, -1).astype(int)

        return kmeans_image

    def get_colors_from_image(self, polygons: List, image: np.array, color_collection: np.array = None) -> List:
        """Applies colors to a list of polygons, the colors can come from the nearest neighboring pixel of the
        original image or from a modified image with reduced colors using kmeans

        Parameters
        ----------
        polygons : List
            List of polygons
        image : np.array
            Image used to extract the colors
        color_collection : np.array
            An array containing a collection of colors, where each color is represented as [R, G, B], by default None

        Returns
        -------
        List
            _description_
        """
        colors = []
        logger.info("Applying colors to mosaic")
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
            if color_collection is not None:
                color = self.get_closest_color(color, color_collection)

            colors += [color]

        return colors

    @staticmethod
    def get_closest_color(input_color: np.array, color_collection: np.array) -> np.array:
        """Find the closest color in a collection to a given input color using Euclidean distance.

        Parameters
        ----------
        input_color : np.array
            An array representing the input color [R, G, B]
        color_collection : np.array
            An array containing a collection of colors, where each color is represented as [R, G, B]

        Returns
        -------
        np.array
            The color from the collection that is closest to the input color.
        """
        distances = np.linalg.norm(color_collection - input_color, axis=1)
        closest_index = np.argmin(distances)
        closest_color = color_collection[closest_index]

        return closest_color

    def save_histogram_of_colors(self, unique_colors: np.array, normalized_count: List[float]):
        """Saves a figure with the color distribution.

        Parameters
        ----------
        unique_colors : np.array
            Array with the bins representation of the colors
        normalized_count : List[float]
            Normalized count of colors
        """
        color_names = [" ".join(map(str, ((rgb_color) * 255).astype(int))) for rgb_color in unique_colors]
        fig, _ = plt.subplots(dpi=96, figsize=(10, 6))

        # Create a bar plot
        plt.bar(color_names, normalized_count, color=list(unique_colors), edgecolor="black")
        plt.xticks(rotation=20, ha="right")
        plt.xlabel("RGB Colors")
        plt.ylabel("Percentage of total area")
        plt.title("Color Distribution")

        xlocs, _ = plt.xticks()
        for i, value in enumerate(normalized_count):
            plt.text(xlocs[i] - 0.3, value + 0.02, str(value))

        fig.tight_layout()
        file_name = Path(self.config_params.image_path).stem
        output_path = os.path.join(os.getcwd(), "intermediate_steps", f"{file_name}_colors.png")
        logger.info("Saving color info to %s", output_path)
        dest_dir = os.path.dirname(output_path)
        os.makedirs(dest_dir, exist_ok=True)
        fig.savefig(output_path, dpi=96, format="png")

    def report_color_histograms(self, mosaic_colors: List[np.array], mosaic: List):
        """Report color histograms of a mosaic. It calculates color histograms based on the mosaic's colors
        and tiles. It provides information about the percentage area covered by each color in the mosaic

        Parameters
        ----------
        mosaic_colors : List[np.array]
            List of color arrays present in the mosaic.
        mosaic : List
            List of Tile objects composing the mosaic.
        """
        unique_colors, indices = np.unique(np.array(mosaic_colors), axis=0, return_inverse=True)
        indices_by_bin = [np.where(indices == i)[0] for i in range(len(unique_colors))]

        total_mosaic_area = sum([tile.area for tile in mosaic])
        precentage_texts = []
        for bin_id, tile_idx in enumerate(indices_by_bin):
            color_tile_areas = [mosaic[idx].area for idx in tile_idx]
            color_tile_percentage = round(100 * (sum(color_tile_areas) / total_mosaic_area), 2)

            rgb_color = (unique_colors[bin_id]) * 255
            logger.info(f"Color {rgb_color} covers a total area of {color_tile_percentage}% in the mosaic")

            precentage_texts.append(color_tile_percentage)

        self.save_histogram_of_colors(unique_colors, precentage_texts)

    def apply_colors(self, mosaic: List, image: np.array) -> List[np.array]:
        """Returns a list of RGB colors, each color correspoding to a mosaic tile.

        Coloring Methods: configured at initialization
        - "original": Extract colors directly from the input image for each mosaic tile.
        - "kmeans": Apply k-means clustering to the image and assign cluster center colors to mosaic tiles.
        - "color_collection": Load a color collection from a file and assign colors to mosaic tiles.

        Parameters
        ----------
        mosaic : List
            List with the mosaic tiles
        image : np.array
            Image where to find the colors
        """

        if self.coloring_method == "original":
            mosaic_colors = self.get_colors_from_image(mosaic, image)
        elif self.coloring_method == "kmeans":
            image = self.apply_kmeans_to_image(image)
            mosaic_colors = self.get_colors_from_image(mosaic, image)
        elif self.coloring_method == "color_collection":
            collor_collection = (np.load(self.colormap_path)) / 255
            mosaic_colors = self.get_colors_from_image(mosaic, image, collor_collection)
            self.report_color_histograms(mosaic_colors, mosaic)

        return mosaic_colors


if __name__ == "__main__":

    data_paths = [r"..\data\input\marmor_colors.jpg"]
    color_extractor = MosaicColoring({"coloring_method": "kmeans", "num_colors": 13})
    for fname in data_paths:
        color_extractor.extract_colormap(fname)
