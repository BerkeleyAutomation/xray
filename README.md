[![python-versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# X-Ray: Mechanical Search for an Occluded Object by Minimizing Support of Learned Occupancy Distributions
[[PDF]](https://arxiv.org/pdf/2004.09039.pdf) [[Video]](https://youtu.be/n9leWuXJ6Ko) [[Presentation]](https://youtu.be/NtJdx9H3TD8) [[Website]](http://sites.google.com/berkeley.edu/x-ray)

## Provided Dataset and Pre-trained Model
You can download the provided dataset and a pre-trained model using the scripts `scripts/download_dataset.sh` and `scripts/download_weights.sh`, respectively.

## Local Install
You can install the library with all dependencies using `pip install .` from the root directory of the project.

### Training
You may train a new model on the existing dataset (or on a different dataset):
```shell
python tools/train.py
```
Training configuration is set using the `cfg/train.yaml` file. Command line arguments `--resume` and `--amp` allow for resuming training of an existing model or using automatic mixed precision during training, respectively.

### Benchmarking
You may benchmark an existing model on a dataset:
```
python tools/benchmark.py
```

### Other Scripts
There are several other scripts in the `tools` folder that provide various functionalities for training, testing, or analyzing models or datasets.
- `tools/loss_plots.py` generates plots of metrics tracked during training (e.g., loss, mean IOU, accuracy, and balanced accuracy) and validation.
- `tools/mean_bgr.py` finds the mean BGR values in a training dataset (useful for normalization).
- `tools/test_real.py` provides an example script that reads an image from the camera and plots the image with the predicted occupancy distribution from the model.

## Docker
You can build a docker image for training/benchmarking using the provided dockerfile (e.g., 
`docker build -t xray .` from the root directory). The following commands assume a built docker image named `xray`.
### Example docker training command
```shell
docker run --gpus <device> --rm -v /path/to/cfg:/cfg -v /path/to/dataset:/dataset -v /path/to/models/models:/models xray python3 xray/tools/train.py --config /cfg/train_docker.yaml --amp
```
This command trains a model using AMP and the configuration specified in `train_docker.yaml`. You may also need to adjust the `--shm-size` docker parameter.

### Example docker benchmarking command
```shell
docker run --gpus <device> --rm -v /path/to/cfg:/cfg -v /path/to/dataset:/dataset -v /path/to/models/models:/models -v /path/to/benchmark/output:/benchmark xray python3 xray/tools/benchmark.py --config /cfg/benchmark_docker.yaml 
```
This command benchmarks a model using the configuration specified in `benchmark_docker.yaml`. You may also need to adjust the `--shm-size` docker parameter.