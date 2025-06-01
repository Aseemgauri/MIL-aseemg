#ifndef __SL_MODEL_H__
#define __SL_MODEL_H__

#include <string>
#include <vector>
#include <map>
#include "InferenceWrapper.h"

class SL_Model : public InferenceWrapper {
public:
    SL_Model(const std::string &model_path);
    ~SL_Model();
    
    // Main processing function
    bool processFrame(float* inputFFT, float* outputFFT);
    
    // Buffer management
    void resetBuffers();
    void updateStateBuffers();
    
    // Getters for model properties
    int getNumClasses() const { return num_classes_; }
    int getFFTSize() const { return fft_size_; }
    int getInputChannels() const { return input_channels_; }
    
    // Override infer to handle state buffer updates
    void infer() override;

private:
    // Model properties
    int num_classes_;
    int fft_size_;
    int input_channels_;
    
    // Input/output shapes
    std::vector<int64_t> mixture_shape_;
    std::vector<int64_t> embedding_shape_;
    std::vector<int64_t> output_shape_;
    
    // Buffer names for state management
    std::vector<std::string> state_buffer_names_;
    
    // Current input data
    float* current_mixture_;
    float* current_embedding_;
    
    // Initialize model-specific parameters
    void initializeModelParams();
    void initializeStateBuffers();
    void identifyStateBuffers();
};

#endif // __SL_MODEL_H__ 