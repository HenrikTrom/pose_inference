#include "cpp_utils/opencvtools.h"
#include "cpp_utils/jsontools.h"
#include "pose_inference/config.h"
#include "pose_inference/utils.hpp"
#include <tensorrt-cpp-api/util/Util.h>
#include <sstream>

using namespace pose_inference;

int main(int argc, char *argv[]) {

    if (argc != 3 && argc != 4) {
        spdlog::error("Usage: {} <input_dir> <output_dir> [config_path]", argv[0]);
        return 1;
    }
    const std::filesystem::path input_dir(argv[1]);
    const std::filesystem::path output_dir(argv[2]);
    if (argc == 4) {
        spdlog::info("Config path argument ignored by keypoint_overlay: {}", argv[3]);
    }
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
    SyncPoseIterator pose_iter(resources, fnames);
    int WIDTH, HEIGHT;
    video_iter.get_wh(WIDTH, HEIGHT);
    
    std::array<cv::VideoWriter, BATCH_SIZE+1> writers;
    std::vector<std::string> save_names;
    for (uint16_t i = 0; i < BATCH_SIZE; i++){
        std::string sname = output_resources+fnames.at(i)+"_overlay.mp4";
        save_names.push_back(sname);
        writers.at(i) = cv::VideoWriter(
            sname,
            cv::VideoWriter::fourcc('a','v','c','1'),
            30.0, // FPS
            cv::Size(WIDTH, HEIGHT)
        );
    }
    std::string sname = output_resources+"4cams.mp4";
    save_names.push_back(sname);
    writers.at(BATCH_SIZE) = cv::VideoWriter(
        sname,
        cv::VideoWriter::fourcc('a','v','c','1'),
        30.0, // FPS
        cv::Size(WIDTH, HEIGHT)
    );

    std::array<std::vector<float>, BATCH_SIZE> batch_kptsx, batch_kptsy;
    const std::size_t n_frames = video_iter.get_framecount();
    for (std::size_t m = 0; m < n_frames; m++){
        std::array<cv::Mat, BATCH_SIZE> images;
        video_iter.get_next(images);
        pose_iter.get(batch_kptsx, batch_kptsy);
        cv::Mat result_img(HEIGHT, WIDTH, CV_8UC3);
        for (uint16_t i = 0; i < BATCH_SIZE; i++){
            for (std::size_t k = 0; k<batch_kptsx.at(i).size(); k++){
                if (batch_kptsx.at(i).at(k) == 0 || batch_kptsy.at(i).at(k) == 0){
                    continue;
                }
                cv::drawMarker(
                    images.at(i), 
                    cv::Point(
                        (int) batch_kptsx.at(i).at(k),
                        (int) batch_kptsy.at(i).at(k)
                    ),
                    cv::Scalar(0, 255, 0),
                    cv::MARKER_CROSS,
                    10,
                    2
                );
            }
            writers.at(i).write(images.at(i));

            if (i != 0){
                int col, row;
                cv::Mat tile_img = images.at(i).clone();
                cv::resize(tile_img, tile_img, cv::Size(WIDTH/2, HEIGHT/2));
                switch (i)
                {
                    case 1:
                        col = 0;
                        row = 0;
                        break;
                    case 2:
                        col = 0;
                        row = 1;
                        break;
                    case 3:
                        col = 1;
                        row = 0;
                        break;
                    case 4:
                        col = 1;
                        row = 1;
                        break;
                }
                cv::Rect roi(col * tile_img.cols, row * tile_img.rows, tile_img.cols, tile_img.rows);
                tile_img.clone().copyTo(result_img(roi));
            }
        }
        writers.at(BATCH_SIZE).write(result_img);
    }

    for (uint16_t i = 0; i <= BATCH_SIZE; i++){
        writers.at(i).release();
        spdlog::info("Saved {}", save_names.at(i));
    }
    
    return 0;
}
