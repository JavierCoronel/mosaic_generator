"""
edge_extractor.py
Module to extract the edges of an image with different methods.
Copyright (c) 2023 Javier Coronel
"""
from pathlib import Path
import os
import logging
import skimage as sk
from skimage.morphology import disk
import numpy as np
from edges import hed
import cv2

logger = logging.getLogger("__main__." + __name__)


class EdgeExtractor:
    """Class for edge extraction using different methods"""

    def __init__(self, config_parameters):
        self.config_params = config_parameters
        self.edge_extraction_method = config_parameters.edge_extraction_method
        self.interactive_edge_modification = config_parameters.interactive_edge_modification

    def diblasi_edges(self, image: np.array) -> np.array:
        """
        Apply Di Blasi edge detection algorithm on the input image.

        Parameters
        ----------
        image : ndarray
            Input image.

        Returns
        -------
        ndarray
            Binary image where edges are marked as 1.

        References
        ----------
        .. [1] Di Blasi, G., 1998. "A new edge detection method based on
            noise reduction". Pattern Recognition Letters, 19(5), pp.417-423.
        """

        preprocessed_image = self.preprocess_clasic(image)
        # segment bright areas to blobs
        variance = preprocessed_image.std() ** 2
        image_seg = np.ones((preprocessed_image.shape[0], preprocessed_image.shape[1]))
        threshold = variance / 4 * 2 * 1
        image_seg[abs(preprocessed_image - preprocessed_image.mean()) > threshold] = 0

        image_edge = sk.filters.laplace(image_seg, ksize=3)
        image_edge[image_edge != 0] = 1

        return image_edge

    def sobel_edges(self, image: np.array) -> np.array:
        """
        Apply Sobel edge detection algorithm on the input image.

        Parameters
        ----------
        image : ndarray
            Input image.

        Returns
        -------
        ndarray
            Binary image where edges are marked as 1.

        References
        ----------
        .. [1] Sobel, I., 1990. "An isotropic 3x3 image gradient operator".
            Machine Vision for Three-Dimensional Scenes, pp. 376-379.
        """
        preprocessed_image = self.preprocess_clasic(image)
        edge_sobel = sk.filters.sobel(preprocessed_image)
        binary = edge_sobel > 0.04
        image_edges = sk.morphology.opening(binary, footprint=disk(2).astype(int))
        image_edges = sk.morphology.skeletonize(image_edges).astype(int)

        return image_edges

    def preprocess_clasic(self, image: np.array) -> np.array:
        """
        Apply classic preprocessing steps on the input image.

        markdown
        Copy code
        Parameters
        ----------
        image : ndarray
            Input image.

        Returns
        -------
        ndarray
            Preprocessed image.
        """
        image_gray = sk.color.rgb2gray(image)
        image_eq = sk.exposure.equalize_hist(image_gray)
        preprocessed_image = sk.filters.gaussian(image_eq)

        return preprocessed_image

    @staticmethod
    def interactive_edge_correction(original_image: np.array, edges_image: np.array, use_drawing=False) -> np.array:
        """Interactive edge correction in an image using drawing or erasing.

        Keybindings
        ----------
            'c': Toggle between drawing and erasing mode.
            '+': Increase the size of the eraser.
            '-': Decrease the size of the eraser.
            'q': Finish the interactive correction process.

        Parameters
        ----------
        original_image : np.array
            RGB image to overlay for interactive visualization.
        edges_image : np.array
            Binary image with edges to be corrected.
        use_drawing : bool, optional
            Whether to enable drawing mode for correcting edges, by default False

        Returns
        -------
        np.array
            Edited edges image after user interaction, normalized to the range [0, 1]
        """
        original_image = cv2.cvtColor(original_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        edited_edges = (edges_image * 255).astype(np.uint8).copy()
        drawing = False
        eraser_size = 5
        is_drawing_edges = use_drawing

        def draw_edges(event, x_cord, y_cord, flags, param):
            nonlocal edited_edges, drawing, eraser_size

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                param["prev_pt"] = (x_cord, y_cord)
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                if "prev_pt" in param:
                    if is_drawing_edges:
                        cv2.line(edited_edges, param["prev_pt"], (x_cord, y_cord), 255, 1)
                        param["prev_pt"] = (x_cord, y_cord)
                    else:
                        cv2.circle(edited_edges, (x_cord, y_cord), eraser_size, 0, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                param.pop("prev_pt", None)

        cv2.namedWindow("Edge Correction", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Edge Correction", draw_edges, {"prev_pt": None})

        while True:
            binary_rgb = np.zeros_like(original_image)
            binary_rgb[:, :, 0] = edited_edges
            overlay_image = cv2.addWeighted(original_image, 0.35, binary_rgb, 0.65, 0)
            cv2.imshow("Edge Correction", overlay_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                is_drawing_edges = not is_drawing_edges
            elif key == ord("+"):  # Press 'f' to increase line size
                eraser_size += 1
            elif key == ord("-"):  # Press 'f' to increase line size
                eraser_size -= 1
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
        return edited_edges / 255

    def run(self, image: np.array) -> np.array:
        """Extractes the edges of an image depending on the edge extracting method specified in
        the configuration file

        Parameters
        ----------
        image : np.array
            Image array

        Returns
        -------
        np.array
            Extractes edges of the image
        """
        logger.info("Extracting edges of image")
        if self.edge_extraction_method == "HED":
            edges = hed.extract_edges(image)
        elif self.edge_extraction_method == "sobel":
            edges = self.sobel_edges(image)
        elif self.edge_extraction_method == "diblasi":
            edges = self.diblasi_edges(image)
        else:
            logger.error("Edge extraction option %s not recognized", self.edge_extractor_name)

        if self.interactive_edge_modification:
            edges = self.interactive_edge_correction(image, edges)

        if self.config_params.save_intermediate_steps:
            os.makedirs(os.path.join(os.getcwd(), "intermediate_steps"), exist_ok=True)
            file_name = Path(self.config_params.image_path).stem
            output_path = os.path.join(os.getcwd(), "intermediate_steps", file_name + "_edges.png")
            logger.info("Saving image edges to %s", output_path)
            sk.io.imsave(output_path, (edges * 255).astype(np.uint8))

        return edges
