#include "stages.hpp"

namespace pose_inference {

// DETECTION STAGE
NNStage::NNStage(config_pose &cfg, std::unique_ptr<Engine<float>> engine) : 
    cfg(cfg), engine(std::move(engine)){

    this->ThreadHandle.reset(new std::thread(&NNStage::ThreadFunction, this));

}

NNStage::~NNStage(){}

bool NNStage::ProcessFunction(
    std::vector<std::vector<cv::cuda::GpuMat>> &images, 
    std::vector<std::vector<std::vector<float>>> &features
)
{
    #ifdef USE_DEBUG_TIME_LOGGING
        this->t1 = std::chrono::steady_clock::now();
    #endif
    // outputs.timestamp = inputs.timestamp;
    // outputs.bboxes = inputs.bboxes;
    this->engine->runInference(images, features);
    #ifdef USE_DEBUG_TIME_LOGGING
        this->t2 = std::chrono::steady_clock::now();
        this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
        this->n_iterations++;
        this->total_dt += this->duration;
    #endif

    for (auto& frame : images)
    {
        for (auto& img : frame)
        {
            img.release();
        }
    }
    return true;
    
}   

void NNStage::Terminate(void)
{
    this->ShouldClose = true;
    this->ThreadHandle->join();
    spdlog::info(
        "Average Pose NN Inference Time: {} milliseconds over {} samples", 
        this->total_dt.count()/this->n_iterations, 
        this->n_iterations
    );
}


// PREPROCESS STAGE
PreProcessStage::PreProcessStage(const config_pose &cfg) : cfg(cfg){
    this->ThreadHandle.reset(new std::thread(&PreProcessStage::ThreadFunction, this));
}

PreProcessStage::~PreProcessStage(){}

bool PreProcessStage::ProcessFunction(
    std::array<cv::cuda::GpuMat, BATCH_SIZE> &inputs, 
    std::vector<std::vector<cv::cuda::GpuMat>> &outputs
)
{
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t1 = std::chrono::steady_clock::now();
    #endif
    preprocess(inputs, outputs, this->cfg.inputDims[0]);
    #if defined(USE_DEBUG_TIME_LOGGING)
        this->t2 = std::chrono::steady_clock::now();
        this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->t2 - this->t1);
        this->n_iterations++;
        this->total_dt += this->duration;
    #endif
    return true;
}


void PreProcessStage::Terminate(void)
{
    this->ShouldClose = true;
    this->ThreadHandle->join();
    #ifdef USE_DEBUG_TIME_LOGGING
    spdlog::info(
        "Average Detection PreProcess Time: {} milliseconds over {} samples", 
        this->total_dt.count()/this->n_iterations, 
        this->n_iterations
    );
    #endif
}

} // namespace pose_inference