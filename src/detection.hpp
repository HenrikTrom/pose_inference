#pragma once
#include "config_parser.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <opencv2/dnn.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <algorithm> 

namespace pose_inference {

void preprocess(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &input_batch, std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
    const  nvinfer1::Dims3 &inputDim
);

void _postprocess(
    std::vector<std::vector<std::vector<float>>> &featureVectors,
    std::vector<std::pair<uint16_t, uint16_t>> &shapes,
    std::vector<std::pair<uint16_t, uint16_t>> &offsets, 
    std::vector<std::vector<std::vector<float>>> &keypoints_out
);

}