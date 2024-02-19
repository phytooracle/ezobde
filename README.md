# EZOBDE | EaZy Object Detection
This tool was developed to easily train an object detection model (Faster R-CNN) from ImageBox labels without the need to pre-download images or label JSON.

# Edit YAML file
The YAML file requires the following fields, which can be retrieved from Labelbox:
1. api_key - An API key from Labelbox
2. project_id - The project ID from Labelbox
3. classes - The class name used in Labelbox, e.g. plant, panicle

# Run code
## Python
To run the code, use:

```
./detecto_labelbox_object_detection.py -y config.yaml
```
## Singularity
To build the container, run:

```
singularity build ezobde.simg docker://phytooracle/ezobde:latest
```

Then, run the container using:

```
singularity run ezobde.simg -y <your YAML file here>
```
