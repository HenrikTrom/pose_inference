#include "../utils.hpp"

using namespace pose_inference;
namespace fs = std::filesystem;

// model parameters:
constexpr uint16_t nkpts = 133; 
constexpr uint16_t feat_w = 384;
constexpr uint16_t feat_h = 512;

int main(int argc, char *argv[]) {

    if (argc != 2)
    {
        std::string error_msg = "Expected missing argument for video dir";
        spdlog::error(error_msg);
        return 1;
    }
    std::string folder = std::string(argv[1]);

    std::string resources = std::string(CONFIG_DIR)+"/../"+folder+"/";

    std::array<std::string, BATCH_SIZE> fnames = cpp_utils::get_filenames<BATCH_SIZE>(
        resources, ".mp4"
    );
    cpp_utils::SyncVideoIterator<BATCH_SIZE> video_iter(resources, fnames);
    PoseLogger<nkpts> logger(resources, fnames);

    config_pose cfg;
    std::unique_ptr<Engine<float>> engine;
    load_cfg_engine(
        std::string(CONFIG_DIR)+"/pose_all_config.json", 
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
