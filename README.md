# Ancient Mosaic Image Converter

This project uses Python to convert any image into an ancient mosaic style.

## Usage

To use this project, simply run the `mosaic_generator.py` script and provide the path to the input image as an argument. The script will then process the image and generate an output mosaic image.

## Requirements

- Python 3.x
- Skimage
- Matplotlib
- Numpy
- Easydict

## Implementation Details

The project uses the following steps to convert an image into a mosaic:

1. Read the input image
2. Process the image to reduce its size and improve its quality
3. Extract the edges of the processed image using the Skimage library
4. Calculate polygons that will serve as the tiles of the mosaic
5. Generate colors for each tile based on the original image
6. Plot the mosaic using Matplotlib

## Contributions

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
