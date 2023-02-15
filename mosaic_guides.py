import copy
import numpy as np
from scipy.ndimage import label, morphology
from skimage import draw
from skimage.morphology import closing, disk, skeletonize
import matplotlib.pyplot as plt


class MosaicGuides:
    def __init__(self, config_parameters):
        self.half_tile = config_parameters.tile_size // 2
        self.chain_spacing = 0.5
        self.height = None
        self.width = None

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
        for x_cord in range(1, self.height - 1):
            for y_cord in range(1, self.width - 1):
                numerator = distances[x_cord, y_cord + 1] - distances[x_cord, y_cord - 1]
                denominator = distances[x_cord + 1, y_cord] - distances[x_cord - 1, y_cord]
                gradient[x_cord, y_cord] = np.arctan2(numerator, denominator)
        angles_0to180 = (gradient * 180 / np.pi + 180) % 180

        return angles_0to180

    def _pixellines_to_ordered_points(self, matrix):
        # break guidelines into chains and order the pixel for all chain

        matrix = skeletonize(matrix)  # nicer lines, better results
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
                x_cord, y_cord = points[0]  # set starting point
                done = False
                subchain = []
                while not done:
                    subchain += [[x_cord, y_cord]]
                    pixel[x_cord, y_cord] = 0
                    done = True
                    for dx_cord, dy_cord in [
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
                            x_cord + dx_cord >= 0
                            and x_cord + dx_cord < pixel.shape[0]
                            and y_cord + dy_cord >= 0
                            and y_cord + dy_cord < pixel.shape[1]
                        ):  # prüfen ob im Bild drin
                            if pixel[x_cord + dx_cord, y_cord + dy_cord] > 0:  # check for pixel here
                                x_cord, y_cord = x_cord + dx_cord, y_cord + dy_cord  # if yes, jump here
                                done = False  # tell the middle loop that the chain is not finished
                                break  # break inner loop
                if len(subchain) > self.half_tile // 2:
                    chains += [subchain]

        return chains

    def get_gaps_from_polygons(self, polygons):

        # get area which are already_cord occupied
        img_chains = np.zeros((self.height, self.width), dtype=np.uint8)

        for polygon in polygons:
            y_cord, x_cord = polygon.exterior.coords.xy
            row_cord, col_cord = draw.polygon(x_cord, y_cord, shape=img_chains.shape)
            img_chains[row_cord, col_cord] = 1

        img_chains_2 = closing(img_chains, disk(2))
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
    def plot_chains(chains):

        _, axis = plt.subplots(dpi=90)
        axis.invert_yaxis()
        axis.autoscale()

        print("Drwaing chain")
        for chain in chains:
            y_cord, x_cord = np.array(chain).T
            axis.plot(y_cord, x_cord, lw=0.7)  # , c='w'

        plt.show()
