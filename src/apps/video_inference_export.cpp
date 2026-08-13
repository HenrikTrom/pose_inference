#include "pose_inference/utils.hpp"
#include <tensorrt-cpp-api/util/Util.h>

using namespace pose_inference;
namespace fs = std::filesystem;

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

    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".mp4"
    );
    cpp_utils::SyncVideoIterator<BATCH_SIZE> video_iter(resources, fnames);
    PoseLogger<nkpts> logger(output_resources, fnames);

    config_pose cfg;
    std::unique_ptr<Engine<float>> engine;
    load_cfg_engine(
        cfg_path.string(),
        cfg,
        engine
    );

    PoseModule<nkpts, feat_w, feat_h> pose_module(cfg, std::move(engine));

    while (!pose_module.IsReady())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::thread feeder{feeder_thread_video<nkpts, feat_w, feat_h>, &pose_module, &video_iter};
    spdlog::info("Feeder thread started");
    spdlog::info("Inferencing samples...");

    std::size_t counter{0};
    std::size_t max_elements = video_iter.get_framecount();
    while (counter != (max_elements)){
        std::array<std::array<std::array<float, 2>, nkpts>, BATCH_SIZE> keypoints;
        if(pose_module.Get(keypoints)){
            logger.log(keypoints);
            counter++;

        };
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    if (feeder.joinable()){
        feeder.join();
    }
    spdlog::info("Joined feeder thread");
    pose_module.Terminate();
    logger.write();

    return 0;
}
