import numpy as np
import time
import random
from shapely.geometry import LineString, Polygon, MultiPoint
from shapely import affinity
from tqdm import tqdm
from mosaic_guides import MosaicGuides
import matplotlib.pyplot as plt
from matplotlib import patches
from shapely.validation import make_valid

RAND_SIZE = 0.3  # portion of tile size which is added or removed randomly during construction
MAX_ANGLE = 40  # 30...75 => max construction angle for tiles along roundings


class MosaicTiles:
    def __init__(self, config_parameters):
        self.half_tile = config_parameters.tile_size // 2
        self.A0 = (2 * self.half_tile) ** 2
        self.tile_area = (config_parameters.tile_size) ** 2
        self.rand_extra = int(round(self.half_tile * RAND_SIZE))

    def place_tiles_along_guides(self, chains, angles, polygons=[], gaps=False):

        self.polygons = polygons
        t0 = time.time()
        print("Placing tiles along chains...")
        for ik, chain in enumerate(tqdm(chains)):

            # consider existing polygons next to the new lane (reason: speed)
            search_area = LineString(np.array(chain)[:, ::-1]).buffer(2.1 * self.half_tile)
            self.preselected_nearby_polygons = [poly for poly in self.polygons if poly.intersects(search_area)]

            if gaps:
                polygons = self.estimate_polygons_from_gap_chains(chain)

            else:
                self.estimate_polygons_from_chains(chain, angles)

        return self.polygons

        print(f"Placed {len(self.polygons)} tiles along guidelines", f"{time.time()-t0:.1f}s")

    def estimate_polygons_from_gap_chains(self, chain):

        index_list = list(range(0, len(chain), self.half_tile * 2))
        last_i = len(chain) - 1
        min_delta = 3
        if index_list[-1] != last_i and last_i - index_list[-1] >= min_delta:
            index_list += [last_i]

        for self.i in index_list:
            y, x = chain[self.i]

            p = Polygon(
                [
                    [x - self.half_tile, y + self.half_tile],
                    [x + self.half_tile, y + self.half_tile],
                    [x + self.half_tile, y - self.half_tile],
                    [x - self.half_tile, y - self.half_tile],
                ]
            )
            # fit in polygon (concave ones are okay for now)
            p_buff = p.buffer(0.1)
            nearby_polygons = [poly for poly in self.preselected_nearby_polygons if p_buff.intersects(poly)]
            for p_vorhanden in nearby_polygons:
                try:
                    p = p.difference(p_vorhanden)  # => remove overlap
                except:
                    p = p.difference(p_vorhanden.buffer(0.1))  # => remove overlap
            # keep only largest fragment if more than one exists
            if p.geom_type == "MultiPolygon":
                i_largest = np.argmax([p_i.area for p_i in p.geoms])
                p = p.geoms[i_largest]
            if p.area >= 0.05 * self.A0 and p.geom_type == "Polygon":  # sort out very small tiles
                self.polygons += [p]
                self.preselected_nearby_polygons += [p]
                # counter += 1

    def estimate_polygons_from_chains(self, chain, angles):

        delta_i = int(self.half_tile * 2)  # width of standard tile (i.e. on straight lines)
        for self.i in range(len(chain)):
            y, x = chain[self.i]
            self.winkel = angles[y, x]

            if self.i == 0:  # at the beginning save the first side of the future polygon
                self.i_start = self.i
                rand_i = random.randint(-self.rand_extra, +self.rand_extra)  # a<=x<=b
                self.winkel_start = self.winkel
                self.line_start = LineString([(x, y - self.half_tile), (x, y + self.half_tile)])
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
                self._draw_polygon(x, y)

    def _postprocess_polygons(self, polygons):

        # complete_polygons = self.cut_tiles_outside_frame(polygons)
        shrinked_polygons = self._irregular_shrink(polygons)
        repaired_polygons = self._repair_tiles(shrinked_polygons)
        reduced_polygons = self._reduce_edge_count(repaired_polygons)
        polygons = self._drop_small_tiles(reduced_polygons)

        return polygons

    def _irregular_shrink(self, polygons):
        polygons_shrinked = []
        for p in polygons:
            p = affinity.scale(p, xfact=random.uniform(0.85, 1), yfact=random.uniform(0.85, 1))
            p = p.buffer(-0.03 * self.half_tile)
            # p = affinity.rotate(p, random.uniform(-5,5))
            # p = affinity.skew(p, random.uniform(-5,5),random.uniform(-5,5))
            polygons_shrinked += [p]

        return polygons_shrinked

    def _repair_tiles(self, polygons):
        # remove or correct strange polygons
        polygons_new = []
        for p in polygons:
            if p.type == "MultiPolygon":
                for pp in p.geoms:
                    polygons_new += [pp]
            else:
                polygons_new += [p]

        polygons_new2 = []
        for p in polygons_new:
            if p.exterior.type == "LinearRing":
                polygons_new2 += [p]

        return polygons_new2

    def _reduce_edge_count(self, polygons, tol=20):
        polygons_new = []
        for p in polygons:
            p = p.simplify(tolerance=self.half_tile / tol)
            polygons_new += [p]
        return polygons_new

    def _drop_small_tiles(self, polygons, threshold=0.03):
        polygons_new = []
        counter = 0
        for p in polygons:
            if p.area > threshold * self.A0:
                polygons_new += [p]
            else:
                counter += 1
        print(f"Dropped {counter} small tiles ")
        return polygons_new

    def cut_tiles_outside_frame(self, polygons):
        # remove parts of tiles which are outside of the actual image
        t0 = time.time()
        outer = Polygon(
            [
                (-3 * self.half_tile, -3 * self.half_tile),
                (self.mosaic_guides.width + 3 * self.half_tile, -3 * self.half_tile),
                (self.mosaic_guides.width + 3 * self.half_tile, self.mosaic_guides.height + 3 * self.half_tile),
                (-3 * self.half_tile, self.mosaic_guides.height + 3 * self.half_tile),
            ],
            holes=[
                [
                    (1, 1),
                    (self.mosaic_guides.height - 1, 1),
                    (self.mosaic_guides.height - 1, self.mosaic_guides.width - 1),
                    (1, self.mosaic_guides.width - 1),
                ],
            ],
        )
        polygons_cut = []
        counter = 0
        for j, p in enumerate(polygons):
            x, y = list(p.representative_point().coords)[0]
            if (
                y < 4 * self.half_tile
                or y > self.mosaic_guides.height - 4 * self.half_tile
                or x < 4 * self.half_tile
                or x > self.mosaic_guides.width - 4 * self.half_tile
            ):
                p = make_valid(p).difference(make_valid(outer))  # => if outside image borders
                counter += 1
            if p.area >= 0.05 * self.A0 and p.geom_type == "Polygon":
                x_exterior, y_exterior = p.exterior.xy
                x_cords_in_range = [
                    self.check_coords_in_range(coord, 0, self.mosaic_guides.width) for coord in x_exterior
                ]

                y_coords_in_range = [
                    self.check_coords_in_range(coord, 0, self.mosaic_guides.height) for coord in y_exterior
                ]

                polygon_is_in_valid_area = np.all(y_coords_in_range)

                if polygon_is_in_valid_area:
                    polygons_cut += [p]
        print(f"Up to {counter} tiles beyond image borders were cut", f"{time.time()-t0:.1f}s")
        return polygons_cut

    @staticmethod
    def check_coords_in_range(coords, lower_bound, higher_bound):
        if lower_bound <= coords <= higher_bound:
            return True
        return False

    def _draw_polygon(self, x, y):

        line = LineString([(x, y - self.half_tile), (x, y + self.half_tile)])
        line = affinity.rotate(line, -self.winkel)

        # construct new tile
        p = MultiPoint([self.line_start.coords[0], self.line_start.coords[1], line.coords[0], line.coords[1]])
        p = p.convex_hull

        self.line_start = line
        self.winkel_start = self.winkel

        if 0:  # self.i - self.i_start <= 2:
            self.i_start = self.i

        else:

            self.i_start = self.i
            rand_i = random.randint(-self.rand_extra, +self.rand_extra)  # a<=x<=b

            # cut off areas that overlap with already existing tiles
            nearby_polygons = [poly for poly in self.preselected_nearby_polygons if p.disjoint(poly) == False]
            p = self._fit_in_polygon(p, nearby_polygons)

            # Sort out small tiles
            if p.area >= 0.08 * self.tile_area and p.geom_type == "Polygon" and p.is_valid:
                self.polygons += [p]
                self.preselected_nearby_polygons += [p]

    def _fit_in_polygon(self, p, nearby_polygons):
        # Remove parts from polygon which overlap with existing ones:
        for p_there in nearby_polygons:
            p = p.difference(p_there)
        # only keep largest part if polygon consists of multiple fragments:
        if p.geom_type == "MultiPolygon":
            i_largest = np.argmax([p_i.area for p_i in p.geoms])
            p = p.geoms[i_largest]
        # remove pathologic polygons with holes (rare event):
        if p.type not in ["MultiLineString", "LineString", "GeometryCollection"]:
            if p.interiors:  # check for attribute interiors if accessible
                p = Polygon(list(p.exterior.coords))

        return p

    @staticmethod
    def plot_polygons(polygons, colors=None, background=None):

        fig, ax = plt.subplots(dpi=90)
        ax.invert_yaxis()
        ax.autoscale()
        ax.set_facecolor("antiquewhite")
        # ax.set_facecolor((1.0, 0.47, 0.42))

        print("Drwaing mosaic")
        for j, p in enumerate(tqdm(polygons)):  # +

            if colors is not None:
                color = colors[j]
                edgecolor = "black"
            else:
                color = "silver"
                edgecolor = "black"

            x, y = p.exterior.xy
            # plt.plot(x,y)
            corners = np.array(p.exterior.coords.xy).T
            stein = patches.Polygon(corners, edgecolor=edgecolor, lw=0.3, facecolor=color)  # facecolor=color)
            ax.add_patch(stein)

        ax.set_facecolor("antiquewhite")
        ax.margins(0)

        fig.canvas.draw()
        plt.show()

        return fig
