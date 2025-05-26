#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <map>

class OnnxProcessor {
public:
    OnnxProcessor(const std::string& modelPath);
    ~OnnxProcessor();

    // Process a single frame of FFT data
    bool processFrame(float* inputFFT, float* outputFFT);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};
    
    // Input/output tensor names
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // Buffer tensors and their underlying data
    std::map<std::string, Ort::Value> buffer_tensors_;
    std::map<std::string, float*> buffer_data_;
    
    // Tensor dimensions
    std::vector<int64_t> input_dims_{1, 4, 1, 129};  // mixture_tf
    std::vector<int64_t> embedding_dims_{1, 5};      // embedding
    std::vector<int64_t> conv_buf_dims_{1, 4, 2, 129}; // conv_buf
    std::vector<int64_t> deconv_buf_dims_{1, 32, 2, 129}; // deconv_buf
    std::vector<int64_t> block_buf_dims_{1, 129, 32}; // block_bufs
    const int64_t output_dims_[4] = {1, 5, 1, 258}; // output
    
    bool initializeModel(const std::string& modelPath);
    void initializeBuffers();
    void updateBuffers(const std::vector<Ort::Value>& output_tensors);
}; 