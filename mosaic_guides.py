import numpy as np
from skimage import draw
from scipy.ndimage import label, morphology
from skimage.morphology import closing, disk
from skimage.morphology import skeletonize
import copy
import time
import skimage as sk
import matplotlib.pyplot as plt


class MosaicGuides:
    def __init__(self, config_parameters):
        self.half_tile = config_parameters.tile_size // 2
        self.chain_spacing = 0.5

    def get_initial_guides(self, image_edges):

        # for each pixel get distance to closest edge
        distances = morphology.distance_transform_edt(
            image_edges == 0,
        )

        self.height = image_edges.shape[0]
        self.width = image_edges.shape[1]
        guidelines = np.zeros((self.height, self.width), dtype=np.uint8)
        mask = (distances.astype(int) + self.half_tile) % (2 * self.half_tile) == 0
        guidelines[mask] = 1

        chains = self._pixellines_to_ordered_points(guidelines)
        angles = self._get_gradient_angles(distances)

        return chains, angles

    def _get_gradient_angles(self, distances):

        gradient = np.zeros((self.height, self.width))
        for x in range(1, self.height - 1):
            for y in range(1, self.width - 1):
                numerator = distances[x, y + 1] - distances[x, y - 1]
                denominator = distances[x + 1, y] - distances[x - 1, y]
                gradient[x, y] = np.arctan2(numerator, denominator)
        angles_0to180 = (gradient * 180 / np.pi + 180) % 180

        return angles_0to180

    def _pixellines_to_ordered_points(self, matrix):
        # break guidelines into chains and order the pixel for all chain

        matrix = sk.morphology.skeletonize(matrix)  # nicer lines, better results
        matrix_labeled, chain_count = label(matrix, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # find chains
        chains = []
        for i_chain in range(1, chain_count):
            pixel = copy.deepcopy(matrix_labeled)
            pixel[pixel != i_chain] = 0

            # alternative using openCV results results in closed chains (might be better), but a few chains are missing
            # hierarchy,contours = cv2.findContours(pixel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # for h in hierarchy:
            #     h2 = h.reshape((-1,2))
            #     h3 = [list(xy)[::-1] for xy in h2]
            # if len(h3)>3:
            #     chains3 += [ h3 ]

            while True:
                points = np.argwhere(pixel != 0)
                if len(points) == 0:
                    break
                x, y = points[0]  # set starting point
                done = False
                subchain = []
                while not done:
                    subchain += [[x, y]]
                    pixel[x, y] = 0
                    done = True
                    for dx, dy in [
                        (+1, 0),
                        (-1, 0),
                        (+1, -1),
                        (-1, +1),
                        (
                            0,
                            -1,
                        ),
                        (0, +1),
                        (-1, -1),
                        (+1, +1),
                    ]:
                        if (
                            x + dx >= 0 and x + dx < pixel.shape[0] and y + dy >= 0 and y + dy < pixel.shape[1]
                        ):  # prüfen ob im Bild drin
                            if pixel[x + dx, y + dy] > 0:  # check for pixel here
                                x, y = x + dx, y + dy  # if yes, jump here
                                done = False  # tell the middle loop that the chain is not finished
                                break  # break inner loop
                if len(subchain) > self.half_tile // 2:
                    chains += [subchain]

        return chains

    def get_gaps_from_polygons(self, polygons):

        # get area which are already occupied
        img_chains = np.zeros((self.height, self.width), dtype=np.uint8)

        for p in polygons:
            y, x = p.exterior.coords.xy
            rr, cc = draw.polygon(x, y, shape=img_chains.shape)
            img_chains[rr, cc] = 1

        img_chains_1 = closing(img_chains, disk(1))
        img_chains_2 = closing(img_chains, disk(2))
        img_chains_3 = closing(img_chains, disk(3))
        distance_to_tile = morphology.distance_transform_edt(img_chains_2 == 0).astype(int)

        distance_to_tile_binary = distance_to_tile / distance_to_tile.max() > 0.2
        guides = skeletonize(distance_to_tile_binary)

        # chain_spacing = int(round(self.half_tile * self.chain_spacing))
        # if chain_spacing <= 1:  # would select EVERY pixel inside gap
        #     chain_spacing = 2
        # # first condition (d==1) => chains around all (even the smallest) gap borders
        # # (set e.g. d==2 for faster calculations)
        # # second condition (...) => more chains inside larger gaps
        # mask = (distance_to_tile == 1) | ((distance_to_tile % chain_spacing == 0) & (distance_to_tile > 0))

        # guidelines2 = np.zeros((self.height, self.width), dtype=np.uint8)
        # guidelines2[mask] = 1

        chains = self._pixellines_to_ordered_points(guides)
        angles = self._get_gradient_angles(distance_to_tile)

        return chains, angles

    @staticmethod
    def plot_chains(chains, colors=None):

        fig, ax = plt.subplots(dpi=90)
        ax.invert_yaxis()
        ax.autoscale()

        print("Drwaing chain")
        for chain in chains:
            yy, xx = np.array(chain).T
            ax.plot(xx, yy, lw=0.7)  # , c='w'

        plt.show()
