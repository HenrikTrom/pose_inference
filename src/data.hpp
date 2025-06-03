#pragma once

#include<vector>
#include<time.h>
#include <utility>
#include <opencv2/core/cuda.hpp>
#include "config.h"

namespace pose_inference {

struct input_pose {
    std::array<cv::Rect, BATCH_SIZE> bboxes;
    std::array<cv::cuda::GpuMat, BATCH_SIZE> images;
    
};

struct input_postprocess {
    std::array<cv::Rect, BATCH_SIZE> bboxes;
    std::vector<std::vector<std::vector<float>>> features;
};

} // namespace pose_inference