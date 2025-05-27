#include "utils.hpp"

namespace pose_inference{

void load_image_data(
    input_pose &PreProcessIn, std::array<cv::Mat, BATCH_SIZE> &cpuImgs, 
    std::array<std::string, BATCH_SIZE> fnames, std::string resources
) {
    for (uint8_t i =0; i<BATCH_SIZE; i++){
        std::string inputImage = resources+fnames.at(i)+".jpg";
        cpuImgs.at(i) = cv::imread(inputImage);
        if (cpuImgs.at(i).empty()){
            const std::string msg = "Unable to read image at path: " + inputImage;
            spdlog::error(msg);
            throw std::runtime_error(msg);
        }
        PreProcessIn.images.at(i).upload(cpuImgs.at(i));
        cv::cuda::cvtColor(PreProcessIn.images.at(i), PreProcessIn.images.at(i), cv::COLOR_BGR2RGB);
        PreProcessIn.bboxes.at(i) = cv::Rect(0, 0, cpuImgs.at(i).cols, cpuImgs.at(i).rows);
    }
}

} // namespace pose_inference