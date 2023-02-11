from edges.HED import extract_edges
import edges.classic_edge_extractors as cee


class EdgeExtractor:
    def __init__(self, config_parameters):
        self.edge_extractor_name = config_parameters.edges

    def extract_edges():
        return 1

    def run(self, image):
        if self.edge_extractor_name == "HED":
            edges = extract_edges(image)

        if self.edge_extractor_name != "HED":
            image = cee.preprocess_clasic(image)

            if self.edge_extractor_name == "sobel":
                edges = cee.sobel_edges(image)

            if self.edge_extractor_name == "diblasi":
                edges = cee.diblasi_edges(image)

        return edges

    def post_process_image():
        return 1
