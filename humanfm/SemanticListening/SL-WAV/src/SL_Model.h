#ifndef __SL_MODEL_H__
#define __SL_MODEL_H__

#include <string>
#include <map>
#include "InferenceWrapper.h"

class SL_Model : public InferenceWrapper {
public:
    SL_Model(const std::string &model_path);
    virtual ~SL_Model();
    
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
    
    // Input/output shapes (fixed arrays instead of vectors)
    int64_t mixture_shape_[4];      // [batch, channels, time, freq]
    int64_t embedding_shape_[2];    // [batch, num_classes]
    int64_t output_shape_[4];       // [batch, classes, time, freq]
    
    // State buffer management (exact allocation based on discovered model structure)
    static const int MAX_MEMORY_POOL_SIZE = 8192;  // Large enough for reasonable models
    char nameMemoryPool[MAX_MEMORY_POOL_SIZE];
    char** state_buffer_names_;     // Array of pointers into nameMemoryPool
    int num_state_buffers_;
    int max_name_length_;
    
    // Current input data
    float* current_mixture_;
    float* current_embedding_;
    
    // Initialize model-specific parameters
    void initializeModelParams();
    void initializeStateBuffers();
    void identifyStateBuffers();
    void allocateNameStorage();
};

#endif // __SL_MODEL_H__ 