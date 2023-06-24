# Mosaic Image Generator

This repository contains Python code to convert any image into an artistic mosaic representation. It is based on previous work from [yobeatz](https://github.com/yobeatz/mosaic).

|     **Original Image**     |     **Generated Mosaic**     |
| :------------------------: | :--------------------------: |
|  <img src="data/donkey_original.jpg" width="350" height="350" style="border: 1px solid black; max-width:100%; height:auto;"/>  |  <img src="data/donkey_mosaic.jpg" width="350" height="350" style="border: 1px solid black; max-width:100%; height:auto;"/>  |


## Implementation Details

The project uses the following steps to convert an image into a mosaic:
1. Read and process and image.
2. Extract the edges of the image using AI or classical edge extraction methods.
4. Derive guidelines to place polygons that will serve as the tiles of the mosaic.
5. Apply color for each tile based on the original image.
6. Plot and save the mosaic.


## Usage

To generate a mosaic have a look at the following steps:
1. **Create your configuration parameters**. Use the file `data/configs/default.yaml` as a template for the parameters:
    - Provide the path to the input image and a desired output folder.
    - Define the method of edge extraction (HED, diblasi, sobel).
    - Define the desired tile size for the mosaic.
    - Define the coloring method (Original colors, k-means clustered colors) and the total number of colors.
    - Whether to resize the image or not (if the image is too big).
    
    Save the yaml file with a new file name.
2. **Run the code to generate the mosaic**: Run the main script as follows:
    ```
    python main.py --config-name=config_name
    ```
3. **Inspect your mosaic**: In the output folder you specified, you will find a subfolder named with the date of code execution. This allows for running the code multiple times with different parameters for a same image. Inside each folder you will find the following files:
    - The mosaic saved as an image.
    - The YAML file with the parameters you used for this mosaic.
    - A log file capturing all the information printed by the code during its execution.


## Requirements

Check the [`requirements`](requirements.txt) file.

## Contributions

Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
