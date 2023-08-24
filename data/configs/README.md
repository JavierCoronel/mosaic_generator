# Configuration parameters

## Usage
We use YAML configuration files managed by [Hydra](https://hydra.cc/docs/intro/) to define various parameters to generating mosaics. 

To get started, consider using the following templates and adjust them to tailor the mosaic generation process according to your requirements. Save your modified `.yaml` file with a different name, and use it to execute the code as explained in the [usage](../../README.md#usage) section. 

You can run two types of hydra configs:
* [`default.yaml`](./default.yaml) has a list of parameters to run a single execution for mosaic generation.
* [`default_parallel.yaml`](./default_parallel.yaml) has additional hydra configurations to run multiple mosaic generations in parallel. Here you can list multiple values for a single parameter under the `sweeper/params` part of the `.yaml` file.

## Parameters
Here you can find a list of the posible parameters with their explaination and possible values:
#### `image_path` 
- Type: String
- Description: Absolute or relative path to the input image.
- Example: `/path/to/image.jpg`

#### `output_folder`
- Type: String
- Description: Absolute or relative path to the output folder where generated mosaics will be saved.
- Example: `/path/to/folder`

#### `edge_extraction_method`
- Type: String
- Description: Method to extract the edges.
- Options: `HED` (deep learning method), `diblasi`, `sobel`
- Example: `HED`

#### `tile_size`
- Type: Integer
- Description: Size in pixels of the tiles used to create the mosaic.
- Example: `10`

#### `coloring_method`
- Type: String
- Description: Coloring method for the mosaic. If specified as original, the original image colors will be used. If kmeans is specified, a clustering of the main colors will be obtained.
- Options: `original` (more colors), `kmeans`
- Example: `kmeans`

#### `num_colors`
- Type: Integer
- Description: Number of colors to extract. Applicable only when `coloring_method` is set to `kmeans`.
- Example: `5`

#### `resize_image`
- Type: Boolean
- Description: Whether to resize the input image to half its size.
- Example: `False`

#### `mosaic_height`
- Type: Integer
- Description: Desired mosaic height in centimeters of the saved mosaic.
- Example: `10`

#### `mosaic_width`
- Type: Integer
- Description: Desired mosaic width in centimeters of the saved mosaic.
- Example: `10`

#### `interactive_edge_modification`
- Type: Boolean
- Description: If `True`, an interactive window will open after extracting the edges in order to modify them.
Use the following keybindings to modify the edges:
*'c': Toggle between drawing and erasing mode.
*'+': Increase the size of the eraser.
*'-': Decrease the size of the eraser.
*'q': Finish the interactive correction process.
- Example: `False`

#### `save_intermediate_steps`
- Type: Boolean
- Description: If `True`, intermediate steps like images with the extracted edges and mosaic guides will be saved.
- Example: `False`

#### `edges_path`
- Type: String
- Description: Path to an image containing extracted edges. Useful for loading previously modified and saved edges. By default, `null` meaning no input file with edeges.
- Example: `null`

##
Got an idea of a parameter that might be relevant to use? [Open an issue](https://github.com/JavierCoronel/mosaic_generator/issues/new/choose) describing your idea!