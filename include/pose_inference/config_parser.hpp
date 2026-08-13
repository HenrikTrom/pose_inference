#pragma once
#include "cpp_utils/jsontools.h"
#include "tensorrt-cpp-api/engine.h"
#include "pose_inference/config.h"

namespace pose_inference {

constexpr int DOC_BUFFER = 65536;


struct config_pose{

    std::string onnxModelPath = "";
    std::string trtModelPath = "";
    std::string engineFileDir = "";
    uint32_t modelInputWidth = 0;

    uint16_t input_depth = 0;
    std::string precision = "";

    std::vector<nvinfer1::Dims3> inputDims;
    std::vector<nvinfer1::Dims> outputDims;
    float confidence_threshold;
    uint16_t calibrationBatchSize = 0;
    uint16_t kf_threshold = 0;
    uint16_t ma_threshold = 0;
    float kf_q = 0;
    float kf_r = 0;
};

Precision stringToPrecision(const std::string &precisionStr);

bool load_config(const std::string &cfg_path, config_pose &cfg);

bool load_cfg_engine(
    const std::string cfg_path, config_pose &cfg, 
    std::unique_ptr<Engine<float>> &engine
);

} // namespace pose_inference