import time
import random
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPoint
from shapely.validation import make_valid
from shapely import affinity
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches

RAND_SIZE = 0.3  # portion of tile size which is added or removed randomly during construction
MAX_ANGLE = 40  # 30...75 => max construction angle for tiles along roundings


class MosaicTiles:
    def __init__(self, config_parameters):
        self.half_tile = config_parameters.tile_size // 2
        self.a_0 = (2 * self.half_tile) ** 2
        self.tile_area = (config_parameters.tile_size) ** 2
        self.rand_extra = int(round(self.half_tile * RAND_SIZE))
        self.mosaic_height = 0
        self.mosaic_width = 0

    def place_tiles_along_guides(self, chains, angles, polygons=None):

        if polygons is None:
            polygons = []
        self.polygons = polygons
        t_0 = time.time()
        print("Placing tiles along chains...")
        for _, chain in enumerate(tqdm(chains)):

            # consider existing polygons next to the new lane (reason: speed)
            search_area = LineString(np.array(chain)[:, ::-1]).buffer(2.1 * self.half_tile)
            self.preselected_nearby_polygons = [poly for poly in self.polygons if poly.intersects(search_area)]

            self.estimate_polygons_from_chains(chain, angles)

        print(f"Placed {len(self.polygons)} tiles along guidelines", f"{time.time()-t_0:.1f}s")

        return self.polygons

    def estimate_polygons_from_chains(self, chain, angles):

        delta_i = int(self.half_tile * 2)  # width of standard tile (i.e. on straight lines)
        for self.i in range(len(chain)):
            y_cord, x_cord = chain[self.i]
            self.winkel = angles[y_cord, x_cord]

            if self.i == 0:  # at the beginning save the first side of the future polygon
                self.i_start = self.i
                rand_i = random.randint(-self.rand_extra, +self.rand_extra)  # a<=x<=b
                self.winkel_start = self.winkel
                self.line_start = LineString([(x_cord, y_cord - self.half_tile), (x_cord, y_cord + self.half_tile)])
                self.line_start = affinity.rotate(self.line_start, -self.winkel_start)

            # Draw polygon as soon as one of the three conditions is fullfilled:
            draw_polygon = False
            # 1. end of chain is reached
            if self.i == len(chain) - 1:
                draw_polygon = True
            else:
                y_next, x_next = chain[self.i + 1]
                winkel_next = angles[y_next, x_next]
                winkeldelta = winkel_next - self.winkel_start
                winkeldelta = min(180 - abs(winkeldelta), abs(winkeldelta))
                # 2. with the NEXT point a large angle would be reached => draw now
                if winkeldelta > MAX_ANGLE:
                    draw_polygon = True
                # 3. goal width is reached
                if self.i - self.i_start == delta_i + rand_i:
                    draw_polygon = True

            if draw_polygon:
                self._draw_polygon(x_cord, y_cord)

    def postprocess_polygons(self, polygons):

        # complete_polygons = self.cut_tiles_outside_frame(polygons)
        shrinked_polygons = self._irregular_shrink(polygons)
        repaired_polygons = self._repair_tiles(shrinked_polygons)
        reduced_polygons = self._reduce_edge_count(repaired_polygons)
        polygons = self._drop_small_tiles(reduced_polygons)

        return polygons

    def _irregular_shrink(self, polygons):
        polygons_shrinked = []
        for polygon in polygons:
            polygon = affinity.scale(polygon, xfact=random.uniform(0.85, 1), yfact=random.uniform(0.85, 1))
            polygon = polygon.buffer(-0.03 * self.half_tile)
            # polygon = affinity.rotate(polygon, random.uniform(-5,5))
            # polygon = affinity.skew(polygon, random.uniform(-5,5),random.uniform(-5,5))
            polygons_shrinked += [polygon]

        return polygons_shrinked

    def _repair_tiles(self, polygons):
        # remove or correct strange polygons
        polygons_new = []
        for polygon in polygons:
            if polygon.type == "MultiPolygon":
                for polygon_repaired in polygon.geoms:
                    polygons_new += [polygon_repaired]
            else:
                polygons_new += [polygon]

        polygons_new2 = []
        for polygon in polygons_new:
            if polygon.exterior.type == "LinearRing":
                polygons_new2 += [polygon]

        return polygons_new2

    def _reduce_edge_count(self, polygons, tol=20):
        polygons_new = []
        for polygon in polygons:
            polygon = polygon.simplify(tolerance=self.half_tile / tol)
            polygons_new += [polygon]
        return polygons_new

    def _drop_small_tiles(self, polygons, threshold=0.03):
        polygons_new = []
        counter = 0
        for polygon in polygons:
            if polygon.area > threshold * self.a_0:
                polygons_new += [polygon]
            else:
                counter += 1
        print(f"Dropped {counter} small tiles ")
        return polygons_new

    def cut_tiles_outside_frame(self, polygons):
        # remove parts of tiles which are outside of the actual image
        t_0 = time.time()
        outer = Polygon(
            [
                (-3 * self.half_tile, -3 * self.half_tile),
                (self.mosaic_width + 3 * self.half_tile, -3 * self.half_tile),
                (self.mosaic_width + 3 * self.half_tile, self.mosaic_height + 3 * self.half_tile),
                (-3 * self.half_tile, self.mosaic_height + 3 * self.half_tile),
            ],
            holes=[
                [
                    (1, 1),
                    (self.mosaic_height - 1, 1),
                    (self.mosaic_height - 1, self.mosaic_width - 1),
                    (1, self.mosaic_width - 1),
                ],
            ],
        )
        polygons_cut = []
        counter = 0
        for polygon in polygons:
            x_cord, y_cord = list(polygon.representative_point().coords)[0]
            if (
                y_cord < 4 * self.half_tile
                or y_cord > self.mosaic_height - 4 * self.half_tile
                or x_cord < 4 * self.half_tile
                or x_cord > self.mosaic_width - 4 * self.half_tile
            ):
                polygon = make_valid(polygon).difference(make_valid(outer))  # => if outside image borders
                counter += 1
            if polygon.area >= 0.05 * self.a_0 and polygon.geom_type == "Polygon":
                x_exterior, y_exterior = polygon.exterior.xy
                x_cords_in_range = [self.check_coords_in_range(coord, 0, self.mosaic_width) for coord in x_exterior]

                y_coords_in_range = [self.check_coords_in_range(coord, 0, self.mosaic_height) for coord in y_exterior]

                polygon_is_in_valid_area = np.all(y_coords_in_range, x_cords_in_range)

                if polygon_is_in_valid_area:
                    polygons_cut += [polygon]
        print(f"Up to {counter} tiles beyond image borders were cut", f"{time.time()-t_0:.1f}s")
        return polygons_cut

    @staticmethod
    def check_coords_in_range(coords, lower_bound, higher_bound):
        if lower_bound <= coords <= higher_bound:
            return True
        return False

    def _draw_polygon(self, x_cord, y_cord):

        line = LineString([(x_cord, y_cord - self.half_tile), (x_cord, y_cord + self.half_tile)])
        line = affinity.rotate(line, -self.winkel)

        # construct new tile
        polygon = MultiPoint([self.line_start.coords[0], self.line_start.coords[1], line.coords[0], line.coords[1]])
        polygon = polygon.convex_hull

        self.line_start = line
        self.winkel_start = self.winkel

        self.i_start = self.i
        # rand_i = random.randint(-self.rand_extra, +self.rand_extra)  # a<=x<=b

        # cut off areas that overlap with already existing tiles
        nearby_polygons = [poly for poly in self.preselected_nearby_polygons if polygon.disjoint(poly) is False]
        polygon = self._fit_in_polygon(polygon, nearby_polygons)

        # Sort out small tiles
        if polygon.area >= 0.08 * self.tile_area and polygon.geom_type == "Polygon" and polygon.is_valid:
            self.polygons += [polygon]
            self.preselected_nearby_polygons += [polygon]

    def _fit_in_polygon(self, polygon, nearby_polygons):
        # Remove parts from polygon which overlap with existing ones:
        for p_there in nearby_polygons:
            polygon = polygon.difference(p_there)
        # only keep largest part if polygon consists of multiple fragments:
        if polygon.geom_type == "MultiPolygon":
            i_largest = np.argmax([p_i.area for p_i in polygon.geoms])
            polygon = polygon.geoms[i_largest]
        # remove pathologic polygons with holes (rare event):
        if polygon.type not in ["MultiLineString", "LineString", "GeometryCollection"]:
            if polygon.interiors:  # check for attribute interiors if accessible
                polygon = Polygon(list(polygon.exterior.coords))

        return polygon

    @staticmethod
    def plot_polygons(polygons, colors=None, background=None):

        fig, axes = plt.subplots(dpi=90)
        axes.invert_yaxis()
        axes.autoscale()
        axes.set_facecolor("antiquewhite")
        # ax.set_facecolor((1.0, 0.47, 0.42))

        print("Drwaing mosaic")
        for j, polygon in enumerate(tqdm(polygons)):  # +

            if colors is not None:
                color = colors[j]
                edgecolor = "black"
            else:
                color = "silver"
                edgecolor = "black"

            corners = np.array(polygon.exterior.coords.xy).T
            tile = patches.Polygon(corners, edgecolor=edgecolor, lw=0.3, facecolor=color)  # facecolor=color)
            axes.add_patch(tile)

        if background is not None:
            axes.set_facecolor("antiquewhite")
        axes.margins(0)

        fig.canvas.draw()
        plt.show()

        return fig
