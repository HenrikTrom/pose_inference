#include "detection.hpp"
#include <chrono>

namespace pose_inference {

void preprocess(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &input_batch, std::vector<std::vector<cv::cuda::GpuMat>> &inputs, 
    const nvinfer1::Dims3 &inputDim
){
    inputs.clear();

    std::vector<cv::cuda::GpuMat> input;
    for (size_t j = 0; j < BATCH_SIZE; ++j) { // For each element we want to add to the batch...
        // You can choose to resize by scaling, adding padding, or a combination
        // of the two in order to maintain the aspect ratio You can use the
        // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
        // maintain the aspect ratio (adds padding where necessary to achieve
        // this).
        cv::cuda::GpuMat resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(
            input_batch.at(j), inputDim.d[1], inputDim.d[2]
        );
        // You could also perform a resize operation without maintaining aspect
        // ratio with the use of padding by using the following instead:
        //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
        //            inputDim.d[1])); // TRT dims are (height, width) whereas
        //            OpenCV is (width, height)
        input.emplace_back(std::move(resized));
    }
    inputs.emplace_back(std::move(input));
}

}