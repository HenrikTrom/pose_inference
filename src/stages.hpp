#pragma once
#include "cpp_utils/StageBase.h"
#include "detection.hpp"
#include "data.hpp"
#include "config.h"
#include <future>

namespace pose_inference {

// NN
class NNStage : public cpp_utils::StageBase<
    // input_nn, input_postprocess
    std::vector<std::vector<cv::cuda::GpuMat>>, std::vector<std::vector<std::vector<float>>>
>
{
private:
    const config_pose &cfg;
    std::unique_ptr<Engine<float>> engine;
    bool ProcessFunction(
        std::vector<std::vector<cv::cuda::GpuMat>> &images, 
        std::vector<std::vector<std::vector<float>>> &features
    );

public:
    NNStage(config_pose &cfg, std::unique_ptr<Engine<float>> engine);
    ~NNStage();
    void Terminate(void);
};

// PREPROCESS
class PreProcessStage : public cpp_utils::StageBase<
    // input_preprocess, input_nn
    std::array<cv::cuda::GpuMat, BATCH_SIZE>, std::vector<std::vector<cv::cuda::GpuMat>>
>
{
private:
    const config_pose &cfg;
    bool ProcessFunction(
        std::array<cv::cuda::GpuMat, BATCH_SIZE> &inputs, 
        std::vector<std::vector<cv::cuda::GpuMat>> &outputs
    );

public:
    PreProcessStage(const config_pose &cfg);
    ~PreProcessStage();
    void Terminate(void);
    std::vector<cv::Mat> cpuImgsTest;
};
// POSTPROCESS-TEMPLATE
template<uint16_t NKPS, uint16_t FEAT_W, uint16_t FEAT_H>
class PostProcessStage : public cpp_utils::StageBase<
    // input_postprocess, output_postprocess
    input_postprocess, std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE>
>
{
private:
    const config_pose &cfg;
    bool ProcessFunction(
        input_postprocess &inputs, 
        std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE> &keypoints
    ){
        #ifdef USE_DEBUG_TIME_LOGGING
            this->t1 = std::chrono::steady_clock::now();
        #endif
        this->postprocess(
            inputs.features, inputs.bboxes, keypoints
        );
    
        #if defined(USE_DEBUG_TIME_LOGGING)
        this->t2 = std::chrono::steady_clock::now();
            this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
            this->n_iterations++;
            this->total_dt += this->duration;
        #endif
        return true;
    };
    void postprocess(
        std::vector<std::vector<std::vector<float>>> &featureVectors,
        std::array<cv::Rect, BATCH_SIZE> &bboxes,
        std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE> &keypoints_out
    ){
        for (size_t j = 0; j < BATCH_SIZE; j++){
            cv::Rect &bbox = bboxes.at(j);
    
            float factor_w, factor_h;
            
            if (bbox.width > bbox.height){ // get longest side of bbox and compute scaling factors
                factor_w = 1.f;
                factor_h = this->cfg.inputDims[0].d[1] / (static_cast<float>(bbox.height) / static_cast<float>(bbox.width) * this->cfg.inputDims[0].d[2]);
            }
            else{
                factor_w = this->cfg.inputDims[0].d[2] / (static_cast<float>(bbox.width) / static_cast<float>(bbox.height) * this->cfg.inputDims[0].d[1]);
                factor_h = 1.f;
            }        
            
            Eigen::Map<const Eigen::Matrix<float, NKPS, FEAT_W, Eigen::RowMajor>> matx(featureVectors.at(j).at(0).data());
            Eigen::Map<const Eigen::Matrix<float, NKPS, FEAT_H, Eigen::RowMajor>> maty(featureVectors.at(j).at(1).data());
    
            Eigen::Index coeff_y, coeff_x;
            for(u_int16_t i =0; i<NKPS; i++){
                maty.row(i).maxCoeff(&coeff_y);
                matx.row(i).maxCoeff(&coeff_x);
    
                if (
                    maty(i, coeff_y) < this->cfg.confidence_threshold || 
                    matx(i, coeff_x) < this->cfg.confidence_threshold
                ){
                    keypoints_out.at(j).at(i)[0] = 0;//-1;
                    keypoints_out.at(j).at(i)[1] = 0;//-1;
                }
                else
                {
                    keypoints_out.at(j).at(i)[0] = (static_cast<float>(coeff_x)/static_cast<float>(FEAT_W))*factor_w*bbox.width;
                    keypoints_out.at(j).at(i)[1] = (static_cast<float>(coeff_y)/static_cast<float>(FEAT_H))*factor_h*bbox.height;
                }
            }
        }
    
    };


public:
    PostProcessStage(config_pose &cfg) : cfg(cfg){
        this->ThreadHandle.reset(new std::thread(&PostProcessStage::ThreadFunction, this));
    };
    ~PostProcessStage(){};
    void Terminate(void){
        this->ShouldClose = true;
        this->ThreadHandle->join();
        spdlog::info(
            "Average Pose PostProcess Time: {} milliseconds over {} samples", 
            this->total_dt.count()/this->n_iterations, 
            this->n_iterations
        );
    };
};

// FULL POSE 
template<uint16_t NKPS, uint16_t FEAT_W, uint16_t FEAT_H>
class PoseModule
{
private:
    // stages
    std::unique_ptr<PreProcessStage> preprocess_stage;
    std::unique_ptr<NNStage> nn_stage;
    std::unique_ptr<PostProcessStage<NKPS, FEAT_W, FEAT_H>> postprocess_stage;
    // Threads
    void ThreadPreprocessNN(){
        while (!this->ShouldClose) {
            std::vector<std::vector<cv::cuda::GpuMat>> NNIn;
            if (this->preprocess_stage->Get(NNIn)){
                this->nn_stage->Post(NNIn);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };
    void ThreadNNPostProcess(){
        while (!this->ShouldClose) {
            input_postprocess PostIn;
            if (this->nn_stage->Get(PostIn.features)){
                std::lock_guard<std::mutex> lck(this->mtx);
                {
                    PostIn.bboxes = this->queue_bboxes.front();
                }
                this->postprocess_stage->Post(PostIn);
                this->queue_bboxes.pop();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    std::queue<std::array<cv::Rect, BATCH_SIZE>> queue_bboxes;
    std::mutex mtx;

    std::unique_ptr<std::thread> ThreadHandlePreprocessNN;
    std::unique_ptr<std::thread> ThreadHandleNNProstprocess;

    bool ShouldClose = false;
    bool IsReady_flag = false;
public:
    PoseModule(config_pose &cfg, std::unique_ptr<Engine<float>> engine){
        this->nn_stage.reset(new NNStage(cfg, std::move(engine)));
        while (!nn_stage->IsReady())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        this->preprocess_stage.reset(new PreProcessStage(cfg));
        while (!preprocess_stage->IsReady())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        this->postprocess_stage.reset(new PostProcessStage<133, 384, 512>(cfg));
        while (!postprocess_stage->IsReady())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        this->ThreadHandlePreprocessNN.reset(new std::thread(&PoseModule::ThreadPreprocessNN, this)); 
        this->ThreadHandleNNProstprocess.reset(new std::thread(&PoseModule::ThreadNNPostProcess, this)); 
        this->IsReady_flag=true;
    };
    ~PoseModule(){};
    bool Get(std::array<std::array<std::array<float, 2>, NKPS>, BATCH_SIZE> &DataOut){
        if (this->postprocess_stage->GetOutFIFOSize()!=0)
        {
            return this->postprocess_stage->Get(DataOut);;
        }
        return false;
    };
    uint16_t GetInFIFOSize(void)
    {
        return this->preprocess_stage->GetInFIFOSize();
    }
    uint16_t GetOutFIFOSize(void)
    {
        return this->postprocess_stage->GetOutFIFOSize();
    }
    void InPost(input_pose &PreProcessIn){
        this->preprocess_stage->Post(PreProcessIn.images);
        std::lock_guard<std::mutex> lck(this->mtx);
        {
            this->queue_bboxes.push(PreProcessIn.bboxes);
        }
    }
    bool IsReady(void)
    {
        return this->IsReady_flag;
    }
    void Terminate(void){
        this->ShouldClose=true;
        this->ThreadHandlePreprocessNN->join();
        this->ThreadHandleNNProstprocess->join();

        this->preprocess_stage->Terminate();
        this->nn_stage->Terminate();
        this->postprocess_stage->Terminate();
    };
};

} // namespace pose_inference