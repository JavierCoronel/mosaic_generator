from pathlib import Path
import numpy as np
from skimage import io
from skimage import draw
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import os


class MosaicColoring:
    def __init__(self, config_parameters):
        self.coloring_method = config_parameters.get("coloring_method", None)
        self.num_colors = config_parameters.get("num_colors", None)
        self.colormap_path = config_parameters.get("colormap_path", None)

    def apply_color(polygons):

        colored_polygons = 1
        return colored_polygons

    def kmeans_colors(self, input_image, num_colors=6):

        n_colors = self.num_colors
        sample_size = 200000  # reduce sample size to speed up KMeans
        original = input_image[:, :, :3]  # drop alpha channel if there is one
        arr = original.reshape((-1, 3))
        random_indices = np.random.choice(arr.shape[0], size=sample_size, replace=True)
        arr = arr[random_indices, :]
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)

        return kmeans

    def extract_colormap(self, img_path):

        input_fname = io.imread(img_path)
        kmeans = self.kmeans_colors(input_fname)
        color_centers = kmeans.cluster_centers_.astype(int)
        script_path = Path(__file__).parent.absolute()
        out_path = Path.joinpath(script_path, "data", "color_collections")
        os.makedirs(out_path, exist_ok=True)
        np.save(out_path / os.path.basename(img_path), centers)

    def apply_kmeans_to_image(self, image):

        width, height, depth = image.shape
        print("Estimating colormap with kmeans")
        kmeans = self.kmeans_colors(image)

        flat_image = image.reshape(width * height, depth)
        labels = kmeans.predict(flat_image)
        codebook = kmeans.cluster_centers_

        kmeans_image = codebook[labels].reshape(width, height, -1).astype(int)

        return kmeans_image

    def get_colors_from_original(self, polygons, image):

        if self.coloring_method == "kmeans":
            image = self.apply_kmeans_to_image(image)

        colors = []
        print("Obtaining colors for polygons")
        for j, p in enumerate(tqdm(polygons)):

            xx, yy = p.exterior.xy
            x_list, y_list = draw.polygon(xx, yy)
            if len(x_list) > 1 and len(y_list) > 1:
                img_cut = image[min(y_list) : max(y_list) + 1, min(x_list) : max(x_list) + 1, :]
                # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
                average = img_cut.mean(axis=0).mean(axis=0)
                color = average / 255
            else:
                color = image[int(yy[0]), int(xx[0]), :] / 255

            colors += [color]

        return colors


def modify_colors(colors, variant, colors_collection=[]):
    def nearest_color(subjects, query):
        # https://stackoverflow.com/questions/34366981/python-pil-finding-nearest-color-rounding-colors
        return min(subjects, key=lambda subject: sum((s - q) ** 2 for s, q in zip(subject, query)))

    # nearest_color( ((1, 1, 1, "white"), (1, 0, 0, "red"),), (64/255,0,0) ) # example
    new_colors = []
    print("Recoloring...")
    for c in tqdm(colors):
        if variant == "monochrome":
            c_new = nearest_color(((1, 1, 1), (0, 0, 0)), c)  # monochrom
        elif variant == "grayscale":
            c_new = str(0.2989 * c[0] + 0.5870 * c[1] + 0.1140 * c[2])  # matplotlib excepts grayscale be strings
        elif variant == "polychrome":
            n = 9
            some_gray = [(g / n, g / n, g / n) for g in range(n + 1)]
            c_new = nearest_color(some_gray, c)  # monochrom
        elif variant == "source":
            c_new = nearest_color(colors_collection / 255, c)
        else:
            raise ValueError("Parameter not understood.")
        new_colors += [c_new]
    return new_colors


def load_colors():
    script_path = Path(__file__).parent.absolute()
    collection_path = Path.joinpath(script_path, "color_collections")
    color_dict = {}
    for fname in collection_path.glob("*.npy"):
        color_dict[fname.stem] = np.load(fname)
    return color_dict


if __name__ == "__main__":

    data_paths = ["data\input\dalle_4.jpg"]
    color_extractor = MosaicColoring({"coloring_method": "kmeans"})
    for fname in data_paths:
        color_extractor.extract_colormap(fname)
