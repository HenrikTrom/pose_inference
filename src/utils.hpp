#pragma once
#include "stages.hpp"
#include "cpp_utils/clitools.h"
#include "cpp_utils/opencvtools.h"

namespace pose_inference{

constexpr size_t MAX_ITER = 1000; // benchmark samples
constexpr size_t MAX_INFERENCE_SLEEP_MS = 10; // Fastest benchmark inference time in milliseconds

void load_image_data(
    input_pose &PreProcessIn, std::array<cv::Mat, BATCH_SIZE> &cpuImgs, 
    std::array<std::string, BATCH_SIZE> fnames, std::string resources
);

template<uint16_t NKPS, uint16_t FEAT_W, uint16_t FEAT_H>
void feeder_thread_benchmark(PoseModule<NKPS, FEAT_W, FEAT_H> *stage, input_pose PreProcessIn){
    cpp_utils::ProgressBar progressBar(MAX_ITER);
    progressBar.update(0);
    for (std::size_t i = 1; i <= MAX_ITER; i++)
    {
        if (stage->GetInFIFOSize() < cpp_utils::MAXINFIFOSIZE)
        {
            input_pose tmp = PreProcessIn;
            stage->InPost(tmp);
            progressBar.update(i);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(MAX_INFERENCE_SLEEP_MS));
    }
    progressBar.finish();
};

template<uint16_t NKPS, uint16_t FEAT_W, uint16_t FEAT_H>
void feeder_thread_video(
    PoseModule<NKPS, FEAT_W, FEAT_H> *stage, 
    cpp_utils::SyncVideoIterator<BATCH_SIZE> *video_iter
){
    const std::size_t n_frames = video_iter->get_framecount();
    cpp_utils::ProgressBar progressBar(n_frames);
    progressBar.update(0);
    for (std::size_t i = 1; i <= n_frames; i++)
    {
        if (stage->GetInFIFOSize() < cpp_utils::MAXINFIFOSIZE)
        {
            input_pose PreProcessIn;
            std::array<cv::Mat, BATCH_SIZE> tmp;
            video_iter->get_next(tmp);
            for(uint16_t cidx = 0; cidx<BATCH_SIZE; cidx++) {
                PreProcessIn.images.at(cidx).upload(tmp.at(cidx));
                cv::cuda::cvtColor(PreProcessIn.images.at(cidx), PreProcessIn.images.at(cidx), cv::COLOR_BGR2RGB);
                PreProcessIn.bboxes.at(cidx) = cv::Rect(0, 0, tmp.at(cidx).cols, tmp.at(cidx).rows);
            }
            stage->InPost(PreProcessIn);
            progressBar.update(i);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(MAX_INFERENCE_SLEEP_MS));
    }
    progressBar.finish();
};

template<uint16_t NKPS>
class PoseLogger
{
public:
PoseLogger(const std::string &resources, std::array<std::string, BATCH_SIZE> fnames){
    this->resources = resources;
    this->fnames = fnames;
};
~PoseLogger(){};
void log(std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE> keypoints){
    this->log_q.push(keypoints);
};
bool write(){
    std::array<rapidjson::Document, BATCH_SIZE> docs;
    for (std::size_t i = 0; i < BATCH_SIZE; i++) {
        docs.at(i).SetObject();
    }
    int count{0};
    while (!this->log_q.empty()){
        count++;
        std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE> keypoints = this->log_q.front();
        this->log_q.pop();
        for (std::size_t i = 0; i < BATCH_SIZE; i++) {
            rapidjson::Value img_obj(rapidjson::kObjectType);

            rapidjson::Document::AllocatorType& allocator = docs.at(i).GetAllocator();
            rapidjson::Value kptsx_arr(rapidjson::kArrayType);
            rapidjson::Value kptsy_arr(rapidjson::kArrayType);
            for (std::size_t j = 0; j < NKPS; j++) {
                kptsx_arr.PushBack(keypoints.at(i).at(j).at(0), allocator);            
                kptsy_arr.PushBack(keypoints.at(i).at(j).at(1), allocator);            
            }
            img_obj.AddMember("kptsx", kptsx_arr, allocator);
            img_obj.AddMember("kptsy", kptsy_arr, allocator);
            rapidjson::Value key;
            key.SetString(std::to_string(count).c_str(), allocator);
            docs.at(i).AddMember(key, img_obj, allocator);
        }
    }
    // write docs to file
    for (std::size_t i = 0; i < BATCH_SIZE; i++){
        std::string filename = this->resources+this->fnames.at(i)+".json";
        spdlog::info("Saving {}", filename);
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::string error_msg = "Unable to open file: "+filename;
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
            return false;
        }
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        writer.SetIndent(' ', 4);
        docs.at(i).Accept(writer);
        ofs << buffer.GetString();
        ofs.close();
    }
    return true;
}

private:
    std::string resources;
    std::array<std::string, BATCH_SIZE> fnames;
    std::queue<std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE>> log_q;

};


class SyncPoseIterator{
public:
SyncPoseIterator(const std::string &resources, const std::array<std::string, BATCH_SIZE> &fnames){
    for (std::size_t i = 0; i<BATCH_SIZE; i++){
        rapidjson::Document doc;
        cpp_utils::read_json_document(resources+fnames.at(i)+".json", 65536, doc);
        for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) // for s in samples
        {
            std::vector<float> kptsx, kptsy;
            const std::string key = it->name.GetString();  // e.g. "19"
            const auto& obj = it->value;

            for (const auto& v : obj["kptsx"].GetArray()){
                kptsx.push_back(v.GetFloat());
            }
            for (const auto& v : obj["kptsy"].GetArray()){
                kptsy.push_back(v.GetFloat());
            }

            this->x.at(i).push_back(kptsx);
            this->y.at(i).push_back(kptsy);
        }
    }
}
~SyncPoseIterator(){};

void get(
    std::array<std::vector<float>, BATCH_SIZE> &batch_kptsx,
    std::array<std::vector<float>, BATCH_SIZE> &batch_kptsy
){
    for (std::size_t i = 0; i<BATCH_SIZE; i++){
        batch_kptsx.at(i) = this->x.at(i).at(this->frame_idx);
        batch_kptsy.at(i) = this->y.at(i).at(this->frame_idx);
    }
    this->frame_idx++;
}

private:
// batch, samples, keypoints
std::array<std::vector<std::vector<float>>, BATCH_SIZE> x;
std::array<std::vector<std::vector<float>>, BATCH_SIZE> y;
std::size_t frame_idx{0};

void reset(){
    spdlog::info("Resetting SyncPoseIterator");
    this->frame_idx = 0;
}

};

} // namespace pose_inference