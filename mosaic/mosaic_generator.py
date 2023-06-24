"""
mosaic_generator.py
Module that contains the MosaicGenerator class that reads an input image, extracts its edges, and then creates polygons
representing the tiles of a mosaic. The coloring of the tiles is taken from the original image, or it can be generated
using different methods. The generated mosaic is plotted and saved.
Copyright (c) 2023 Javier Coronel
"""

import os
from typing import List

import logging
from utils.image_handler import ImageHandler
from edges.edge_extractor import EdgeExtractor
from mosaic.mosaic_guides import MosaicGuides
from mosaic.mosaic_tiles import MosaicTiles
from mosaic.mosaic_coloring import MosaicColoring

logger = logging.getLogger("__main__." + __name__)


class MosaicGenerator:
    """A class to generate a mosaic"""

    def __init__(self, configuration_params):
        logging.getLogger("MosaicGenerator")
        self.config_params = configuration_params
        self.image_path = configuration_params.image_path
        self.mosaic_coloring = MosaicColoring(configuration_params)
        self.image_handler = ImageHandler(configuration_params)
        self.mosaic_guides = MosaicGuides(configuration_params)
        self.mosaic_tiles = MosaicTiles(configuration_params)
        self.edge_extractor = EdgeExtractor(configuration_params)

    def fill_mosaic_gaps(self, tiles: List[int], iter_num: int = 4) -> List[int]:
        """Loop to find gaps and fill them with more tiles if there are any after generating the mosaic.

        Parameters
        ----------
        tiles : List[int]
            List of polygons that represent the tiles of a mosaic
        iter_num : int, optional
            Number of iterations to fill the gaps in the mosaic, by default 4

        Returns
        -------
        List[int]
            New list of polygons with filled gaps
        """
        while iter_num != 0:
            gap_guides, gap_angles = self.mosaic_guides.get_gaps_from_polygons(tiles)
            tiles = self.mosaic_tiles.place_tiles_along_guides(gap_guides, gap_angles, polygons=tiles)
            iter_num -= 1

        post_proc_mosaic = self.mosaic_tiles.postprocess_polygons(tiles)

        return post_proc_mosaic

    def save_mosaic(self, mosaic_figure):
        """Saves the plot of a mosaic

        Parameters
        ----------
        mosaic_figure : matplotlib figure
            Matplotlib figure with the plot of a mosaic
        """
        file_name = os.path.basename(self.config_params.image_path)
        output_path = os.path.join(os.getcwd(), file_name)
        dpi = 300
        logger.info("Saving mosaic to %s", output_path)
        dest_dir = os.path.dirname(output_path)
        os.makedirs(dest_dir, exist_ok=True)
        mosaic_figure.savefig(output_path, dpi=dpi)

    def generate_mosaic(self):
        """Generates a mosaic based on pre-initialized parameters"""
        # Load and preprocess image
        image = self.image_handler.read_image()
        image_edges = self.edge_extractor.run(image)

        # Get initial guidelines and place tiles
        initial_guides, initial_angles = self.mosaic_guides.get_initial_guides(image_edges)
        raw_mosaic = self.mosaic_tiles.place_tiles_along_guides(initial_guides, initial_angles)

        logger.info("Refining mosaic...")
        # Iterate to replace gaps guidelines and tiles
        final_mosaic = self.fill_mosaic_gaps(raw_mosaic)

        colors = self.mosaic_coloring.get_colors_from_original(final_mosaic, image)

        fig = self.mosaic_tiles.plot_polygons(final_mosaic, colors)

        self.save_mosaic(fig)

        logger.info("Mosaic finished")
