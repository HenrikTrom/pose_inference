#include "../utils.hpp"

using namespace pose_inference;

// model parameters:
constexpr uint16_t nkpts = 133; 
constexpr uint16_t feat_w = 384;
constexpr uint16_t feat_h = 512;

int main(int argc, char *argv[]) {
    // Load inference data
    input_pose PreProcessIn;
    std::array<cv::Mat, BATCH_SIZE> cpuImgs;
    std::string resources = std::string(CONFIG_DIR)+"/../inputs/";

    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".jpg"
    );

    load_image_data(PreProcessIn, cpuImgs, fnames, resources);

    // Load config, engine and module
    config_pose cfg_pose;
    std::unique_ptr<Engine<float>> engine;
    load_cfg_engine(
        std::string(CONFIG_DIR)+"/pose_all_config.json", 
        cfg_pose,
        engine
    );

    PoseModule<nkpts, feat_w, feat_h> pose_module(cfg_pose, std::move(engine));
    
    while (!pose_module.IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::thread feeder{feeder_thread_benchmark<nkpts, feat_w, feat_h>, &pose_module, PreProcessIn};
    spdlog::info("Feeder thread started");
    spdlog::info("Inferencing samples...");

    // Run inference
    std::size_t count{0};
    std::array<std::array<std::array<float, 2>, nkpts>, BATCH_SIZE> keypoints;
    while (count != MAX_ITER){
        if(pose_module.Get(keypoints)){
            count++;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    if (feeder.joinable()){
        feeder.join();
    }
    spdlog::info("Joined feeder thread");
    pose_module.Terminate();
    
    // Draw keypoints on images
    for (int i = 0; i<BATCH_SIZE; i++){
        for (auto &kp : keypoints.at(i)) {
            cv::drawMarker(
                cpuImgs.at(i), 
                cv::Point((int) kp[0], (int) kp[1]), 
                cv::Scalar(0, 255, 0), 
                cv::MARKER_CROSS, 10, 1
            );
        }
        std::string filename = std::string(CONFIG_DIR)+"/../outputs/"+fnames.at(i)+"_out.jpg";
        spdlog::info("Saving {}", filename);
        cv::imwrite(filename, cpuImgs.at(i));
    }

    return 0;
}
