"""
main.py
This is the main script for converting any image into an ancient mosaic style.
It uses the MosaicGenerator class to read an image and create a mosaic. The generated mosaic is plotted and saved.
It uses hydra for handling the parameters to create the mosaic.
Copyright (c) 2023 Javier Coronel
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from mosaic.mosaic_generator import MosaicGenerator

logger = logging.getLogger(__name__)


@hydra.main(config_path="data/configs", config_name="default.yaml")
def generate_mosaic(cfg: DictConfig) -> None:
    """Main function to generate a mosaic using .yaml parameters

    Parameters
    ----------
    cfg : DictConfig
        Parameters listed in a yaml config file
    """
    logger.info("Using the following configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info("Starting MosaicGenerator")
    mosaic = MosaicGenerator(cfg)
    mosaic.generate_mosaic()


if __name__ == "__main__":
    generate_mosaic()  # pylint: disable=no-value-for-parameter
