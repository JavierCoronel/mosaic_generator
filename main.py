import os
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, filters, transform
from easydict import EasyDict as edict
from edges.edge_extractor import EdgeExtractor
from image_preprocessor import ImagePreprocessor
from mosaic_guides import MosaicGuides
from mosaic_tiles import MosaicTiles
from mosaic_coloring import MosaicColoring


class MosaicGenerator:
    def __init__(self, config_parameters):

        self.config_params = config_parameters
        self.image_path = config_parameters.image_path
        self.edge_extractor = EdgeExtractor(config_parameters)
        self.mosaic_coloring = MosaicColoring(config_parameters)
        ##############
        self.image_preprocessor = ImagePreprocessor(config_parameters)
        self.mosaic_guides = MosaicGuides(config_parameters)
        self.mosaic_tiles = MosaicTiles(config_parameters)

    def fill_mosaic_gaps(self, mosaic, iter_num=4):

        for iteration in range(iter_num):
            gap_guides, gap_angles = self.mosaic_guides.get_gaps_from_polygons(mosaic)
            mosaic = self.mosaic_tiles.place_tiles_along_guides(gap_guides, gap_angles, polygons=mosaic)

        post_proc_mosaic = self.mosaic_tiles._postprocess_polygons(mosaic)

        return post_proc_mosaic

    def save_mosaic(self, mosaic_figure):

        file_name = os.path.basename(self.config_params.image_path)
        output_path = os.path.join(self.config_params.output_folder, file_name)
        dpi = 300
        mosaic_figure.savefig(output_path, dpi=dpi)

    def generate_mosaic(self):

        # Load and preprocess image
        image = self.image_preprocessor.read_image()
        image_edges = self.image_preprocessor.extract_edges(image)

        # Get initial guidelines and place tiles
        initial_guides, initial_angles = self.mosaic_guides.get_initial_guides(image_edges)
        raw_mosaic = self.mosaic_tiles.place_tiles_along_guides(initial_guides, initial_angles)

        # Iterate to replace gaps guidelines and tiles
        final_mosaic = self.fill_mosaic_gaps(raw_mosaic)

        colors = self.mosaic_coloring.get_colors_from_original(final_mosaic, image)

        fig = self.mosaic_tiles.plot_polygons(final_mosaic, colors)

        self.save_mosaic(fig)

        print("Done")


if __name__ == "__main__":

    config_parameters = {
        "image_path": "data\input\ds_guilloche_600.jpg",
        "output_folder": "data\output\double_strand",
        "edges": "diblasi",
        "tile_size": 8,
        "coloring_method": "kmeans",  # original/kmeans/dict
        "num_colors": 5,
    }
    config_parameters = edict(config_parameters)
    mosaic = MosaicGenerator(config_parameters)

    mosaic.generate_mosaic()
