# pose_inference

[![DOI](https://zenodo.org/badge/991307682.svg)](https://zenodo.org/badge/latestdoi/991307682)

`pose_inference` is a C++ library and set of command-line tools for batched keypoint inference on synchronized image and video inputs. In this workspace it is usually used after a detection stage has already produced person crops or bounding boxes.

The code assumes a fixed batch size at configure time. In practice that usually means one input stream per camera.

<p align="center">
  <img src="content/4cams.gif" alt="Pose overlay example" width="720">
</p>

## Dependencies

This project depends on:

- OpenCV with CUDA support
- TensorRT 10
- Eigen3
- `cpp_utils`
- `tensorrt-cpp-api`

The simplest way to satisfy those dependencies is to use the container from the root repository:

- [Docker-OpenCV-TensorRT-Dev](https://github.com/HenrikTrom/Docker-OpenCV-TensorRT-Dev)

## Build

Set any required environment variables first. The important ones for this repo are:

```bash
export OPENCV_VERSION=4.13.0
export BATCH_SIZE=5
```

`BATCH_SIZE` is compiled into the binaries. The ONNX model and configuration need to match it.

Build and install:

```bash
sudo ./build_install.sh
```

## Configuration

- Default config file: `cfg/pose_all_config.json`
- Optional config path can be passed to most executables as the third argument
- The model should be exported with a fixed batch size that matches `BATCH_SIZE`
- The current implementation is structured around RTMPose-style models

## Executables

`inference_benchmark`

```bash
./build/inference_benchmark <input_dir> <output_dir> [config_path]
```

Reads `BATCH_SIZE` `.jpg` files from `input_dir`, runs keypoint inference, and writes images with keypoints drawn to `output_dir`.

`video_inference_export`

```bash
./build/video_inference_export <input_dir> <output_dir> [config_path]
```

Reads `BATCH_SIZE` synchronized `.mp4` files from `input_dir` and writes one JSON keypoint log per input video to `output_dir`.

`keypoint_overlay`

```bash
./build/keypoint_overlay <input_dir> <output_dir> [config_path]
```

Reads `.mp4` files and matching keypoint JSON files from `input_dir`, then writes overlay videos to `output_dir`. It also writes a tiled `4cams.mp4` summary video. The optional `config_path` argument is accepted for interface consistency but ignored by this executable.

## Tested Configurations

- Intel Xeon W-2145, RTX 2080 Super, Ubuntu 20.04, CUDA 11.8, TensorRT 8.6.1.6, OpenCV 4.10.0, `BATCH_SIZE=5`
  Preprocess about 1 ms, inference about 4 ms, postprocess about 1 ms over 1000 samples
- Intel Core i7-12700H, RTX 3050 Ti, Ubuntu 24.04, CUDA 12.9, TensorRT 10.14.1.48, OpenCV 4.13.0, `BATCH_SIZE=5`
  Preprocess above 3 ms, inference about 11 ms, postprocess above 1 ms over 1000 samples
- AMD Ryzen 9 7900X3D, RTX 4070 Super, Ubuntu 20.04, CUDA 12.4, TensorRT 10.9.0.34, OpenCV 4.10.0, `BATCH_SIZE=5`
  Preprocess below 1 ms, inference about 2 ms, postprocess below 1 ms over 1000 samples

## Citation

If you use this repository in academic work, use the GitHub "Cite this repository" entry.
