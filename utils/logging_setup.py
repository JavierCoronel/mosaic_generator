"""
logging_setup.py
Module to setup the logging for the main module.
Copyright (c) 2023 Javier Coronel
"""
import os
import logging


def intialize_logging(output_folder: str) -> logging.Logger:
    """Initializes the logging for a main script. It displays the logging in console and saves it to a file.

    Parameters
    ----------
    output_folder : str
        Path where to save the logging

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger("__main__")
    logger.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler which logs debug messages
    log_file_path = os.path.join(output_folder, "mosaic_log.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s : %(name)s : %(levelname)s : %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
