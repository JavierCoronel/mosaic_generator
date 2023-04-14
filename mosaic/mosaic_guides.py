"""
mosaic_guides.py
Module that uses image edges to estimate guidelines for placing mosaic tiles.
Copyright (c) 2023 Javier Coronel
"""
import copy
import logging
from typing import List
import numpy as np
from scipy.ndimage import label, morphology
from skimage import draw
from skimage.morphology import closing, disk, skeletonize
import matplotlib.pyplot as plt

logger = logging.getLogger("__main__." + __name__)


class MosaicGuides:
    def __init__(self, config_parameters):
        self.half_tile = config_parameters.tile_size // 2
        self.chain_spacing = 0.5
        self.height = None
        self.width = None
        self.neighbors_coords = [(x, y) for x in range(1, -2, -1) for y in range(-1, 2) if x != 0 or y != 0]

    def get_initial_guides(self, image_edges: np.array) -> List:
        """Computes the a set of angles and guidelines, useful for tile placement. The guidelines are
        obtained based on the edges of an image and a given tile size

        Parameters:
        -----------
        image_edges: np.array
            The edge map of an image

        Returns:
        --------
        Tuple[List, np.array]
            A tuple containing the list of initial guidelines and the array of guideline angles

        """
        logger.info("Calculating mosaic guides")
        self.height = image_edges.shape[0]
        self.width = image_edges.shape[1]

        # for each pixel get distance to closest edge
        distance_to_edge = morphology.distance_transform_edt(
            image_edges == 0,
        )
        guidelines = (distance_to_edge.astype(int) + self.half_tile) % (2 * self.half_tile) == 0

        list_of_guidelines = self._get_list_of_guidelines(guidelines)
        angles = self._get_guideline_angles(distance_to_edge)

        return list_of_guidelines, angles

    def _get_guideline_angles(self, distances: np.array) -> np.array:
        """Calculates the angle of the gradient of distance_to_edge at each pixel location

        Parameters
        ----------
        distances : np.array
            A numpy array representing the distance to the closest guideline for each pixel

        Returns
        -------
        np.array
            A numpy array representing the angle of the gradient of distance_to_edge at each pixel
            location in the range [0, 180)
        """

        gradient = np.zeros((self.height, self.width))

        for x_coord in range(1, self.height - 1):
            for y_coord in range(1, self.width - 1):

                numerator = distances[x_coord, y_coord + 1] - distances[x_coord, y_coord - 1]
                denominator = distances[x_coord + 1, y_coord] - distances[x_coord - 1, y_coord]
                gradient[x_coord, y_coord] = np.arctan2(numerator, denominator)

        angles_0to180 = (gradient * 180 / np.pi + 180) % 180

        return angles_0to180

    def _get_list_of_guidelines(self, raw_guidelines: np.array) -> List:
        """Breaks an array of guidelines into a list of smaller sub_guidelines containing pixel coordinates

        Parameters
        ----------
        raw_guidelines : np.array
            Binary image array with the raw guidelines

        Returns
        -------
        List
            List of guidelines with the pixel coordinates
        """

        # break guidelines into chains and order the pixel for all chain

        raw_guidelines = skeletonize(raw_guidelines)  # nicer lines, better results
        raw_guidelines_labeled, guidelines_count = label(raw_guidelines, structure=[[1] * 3 for _ in range(3)])

        guidelines = []
        for guide_id in range(1, guidelines_count):
            binary_guide = copy.deepcopy(raw_guidelines_labeled)
            binary_guide[binary_guide != guide_id] = 0

            while True:
                points = np.argwhere(binary_guide != 0)
                if len(points) == 0:
                    break
                x_coord, y_coord = points[0]  # set starting point
                done = False
                sub_guide = []
                while not done:
                    sub_guide += [[x_coord, y_coord]]
                    binary_guide[x_coord, y_coord] = 0
                    done = True

                    for dx_coord, dy_coord in self.neighbors_coords:
                        x_coord_in_image = self.check_coords_in_range(x_coord + dx_coord, 0, self.height)
                        y_coord_in_image = self.check_coords_in_range(y_coord + dy_coord, 0, self.width)

                        if x_coord_in_image and y_coord_in_image:
                            neighbor_pixel_value = binary_guide[x_coord + dx_coord, y_coord + dy_coord]

                            if neighbor_pixel_value > 0:  # check for pixel here
                                x_coord, y_coord = x_coord + dx_coord, y_coord + dy_coord  # if yes, jump here
                                done = False  # tell the middle loop that the chain is not finished
                                break  # break inner loop

                if len(sub_guide) > self.half_tile // 2:
                    guidelines += [sub_guide]

        return guidelines

    def get_gaps_from_polygons(self, polygons):

        logger.info("Getting guidelines for gaps in mosaic")
        # get area which are already_coord occupied
        img_chains = np.zeros((self.height, self.width), dtype=np.uint8)

        for polygon in polygons:
            y_coord, x_coord = polygon.exterior.coords.xy
            row_coord, col_coord = draw.polygon(x_coord, y_coord, shape=img_chains.shape)
            img_chains[row_coord, col_coord] = 1

        img_chains_2 = closing(img_chains, disk(2))
        distance_to_tile = morphology.distance_transform_edt(img_chains_2 == 0).astype(int)

        chain_spacing = int(round(self.half_tile * self.chain_spacing))
        if chain_spacing <= 1:
            chain_spacing = 2
        mask = (distance_to_tile == 1) | ((distance_to_tile % chain_spacing == 0) & (distance_to_tile > 0))

        guidelines2 = np.zeros((self.height, self.width), dtype=np.uint8)
        guidelines2[mask] = 1

        chains = self._get_list_of_guidelines(skeletonize(guidelines2))
        angles = self._get_guideline_angles(distance_to_tile)

        return chains, angles

    @staticmethod
    def check_coords_in_range(coords, lower_bound, higher_bound):
        if lower_bound <= coords < higher_bound:
            return True
        return False

    @staticmethod
    def plot_chains(chains):

        _, axis = plt.subplots(dpi=90)
        axis.invert_yaxis()
        axis.autoscale()

        logger.info("Drawing chain")
        for chain in chains:
            y_coord, x_coord = np.array(chain).T
            axis.plot(y_coord, x_coord, lw=0.7)  # , c='w'

        plt.show()
