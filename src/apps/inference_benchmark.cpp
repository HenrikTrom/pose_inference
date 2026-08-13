#include "pose_inference/utils.hpp"
#include <tensorrt-cpp-api/util/Util.h>

using namespace pose_inference;

// model parameters:
constexpr uint16_t nkpts = 133;
constexpr uint16_t feat_w = 576;
constexpr uint16_t feat_h = 768;

int main(int argc, char *argv[]) {
    if (argc != 3 && argc != 4) {
        spdlog::error("Usage: {} <input_dir> <output_dir> [config_path]", argv[0]);
        return 1;
    }
    const std::filesystem::path input_dir(argv[1]);
    const std::filesystem::path output_dir(argv[2]);
    const std::filesystem::path cfg_path = (argc == 4) ? std::filesystem::path(argv[3])
        : std::filesystem::path(std::string(CONFIG_DIR) + "/pose_all_config.json");
    if (!std::filesystem::is_directory(input_dir)) {
        spdlog::error("Input directory does not exist: {}", input_dir.string());
        return 1;
    }
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error) {
        spdlog::error("Could not create output directory {}: {}", output_dir.string(), error.message());
        return 1;
    }
    const std::string resources = input_dir.string() + "/";
    const std::string output_resources = output_dir.string() + "/";
    if (!Util::ensureCudaDeviceAvailable()) {
        return 1;
    }

    // Load inference data
    input_pose PreProcessIn;
    std::array<cv::Mat, BATCH_SIZE> cpuImgs;
    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".jpg"
    );

    load_image_data(PreProcessIn, cpuImgs, fnames, resources);

    // Load config, engine and module
    config_pose cfg_pose;
    std::unique_ptr<Engine<float>> engine;
    load_cfg_engine(
        cfg_path.string(),
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
        std::string filename = output_resources+fnames.at(i)+"_out.jpg";
        spdlog::info("Saving {}", filename);
        cv::imwrite(filename, cpuImgs.at(i));
    }

    return 0;
}
