#ifndef ONNX_TENSOR_MANAGER_H
#define ONNX_TENSOR_MANAGER_H

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include "audio_config.h"

// Pre-allocated ONNX tensor manager to avoid dynamic allocation during inference
class ONNXTensorManager {
private:
    // Pre-allocated input/output buffers
    alignas(MEMORY_ALIGNMENT) float embeddingInput_[5];
    alignas(MEMORY_ALIGNMENT) float convBufInput_[4 * 2 * 129];
    alignas(MEMORY_ALIGNMENT) float blockBufsInput_[6][2][129 * 32];
    alignas(MEMORY_ALIGNMENT) float deconvBufInput_[32 * 2 * 129];
    alignas(MEMORY_ALIGNMENT) float onnxInputBuffer_[4 * 129];
    
    // Pre-allocated tensor shapes (stack allocated)
    std::array<int64_t, 4> mixtureShape_{{1, 4, 1, 129}};
    std::array<int64_t, 2> embeddingShape_{{1, 5}};
    std::array<int64_t, 4> convBufShape_{{1, 4, 2, 129}};
    std::array<int64_t, 3> blockBufShape_{{1, 129, 32}};
    std::array<int64_t, 4> deconvBufShape_{{1, 32, 2, 129}};
    
    // Pre-allocated input/output name arrays (stack allocated)
    std::array<const char*, 16> inputNames_{{
        "mixture_tf", "embedding", "conv_buf",
        "block_bufs::buf0::h0", "block_bufs::buf0::c0",
        "block_bufs::buf1::h0", "block_bufs::buf1::c0",
        "block_bufs::buf2::h0", "block_bufs::buf2::c0",
        "block_bufs::buf3::h0", "block_bufs::buf3::c0",
        "block_bufs::buf4::h0", "block_bufs::buf4::c0",
        "block_bufs::buf5::h0", "block_bufs::buf5::c0",
        "deconv_buf"
    }};
    
    std::array<const char*, 15> outputNames_{{
        "output", "out::conv_buf",
        "out::block_bufs::buf0::h0", "out::block_bufs::buf0::c0",
        "out::block_bufs::buf1::h0", "out::block_bufs::buf1::c0",
        "out::block_bufs::buf2::h0", "out::block_bufs::buf2::c0",
        "out::block_bufs::buf3::h0", "out::block_bufs::buf3::c0",
        "out::block_bufs::buf4::h0", "out::block_bufs::buf4::c0",
        "out::block_bufs::buf5::h0", "out::block_bufs::buf5::c0",
        "out::deconv_buf"
    }};
    
    // Pre-allocated input tensors array (stack allocated)
    std::array<Ort::Value, 16> inputTensors_;
    bool tensorsInitialized_;
    
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;

public:
    ONNXTensorManager() : tensorsInitialized_(false) {
        // Initialize all buffers to zero
        std::memset(embeddingInput_, 0, sizeof(embeddingInput_));
        std::memset(convBufInput_, 0, sizeof(convBufInput_));
        std::memset(blockBufsInput_, 0, sizeof(blockBufsInput_));
        std::memset(deconvBufInput_, 0, sizeof(deconvBufInput_));
        std::memset(onnxInputBuffer_, 0, sizeof(onnxInputBuffer_));
        
        // Set embedding to ones (as per original code)
        for (int i = 0; i < 5; i++) {
            embeddingInput_[i] = 1.0f;
        }
    }
    
    // Initialize tensors (call once during setup)
    bool initialize() {
        try {
            // Create memory info
            memoryInfo_ = std::make_unique<Ort::MemoryInfo>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            
            // Pre-create all input tensors using placement new to avoid allocation
            int tensorCount = 0;
            
            // mixture_tf: 1x4x1x129
            new (&inputTensors_[tensorCount++]) Ort::Value(
                Ort::Value::CreateTensor<float>(*memoryInfo_, 
                    onnxInputBuffer_, 4 * 129, mixtureShape_.data(), mixtureShape_.size()));
            
            // embedding: 1x5
            new (&inputTensors_[tensorCount++]) Ort::Value(
                Ort::Value::CreateTensor<float>(*memoryInfo_,
                    embeddingInput_, 5, embeddingShape_.data(), embeddingShape_.size()));
            
            // conv_buf: 1x4x2x129
            new (&inputTensors_[tensorCount++]) Ort::Value(
                Ort::Value::CreateTensor<float>(*memoryInfo_,
                    convBufInput_, 4 * 2 * 129, convBufShape_.data(), convBufShape_.size()));
            
            // block_bufs (6 blocks, each with h0 and c0): 1x129x32
            for (int i = 0; i < 6; i++) {
                new (&inputTensors_[tensorCount++]) Ort::Value(
                    Ort::Value::CreateTensor<float>(*memoryInfo_,
                        blockBufsInput_[i][0], 129 * 32, blockBufShape_.data(), blockBufShape_.size()));
                new (&inputTensors_[tensorCount++]) Ort::Value(
                    Ort::Value::CreateTensor<float>(*memoryInfo_,
                        blockBufsInput_[i][1], 129 * 32, blockBufShape_.data(), blockBufShape_.size()));
            }
            
            // deconv_buf: 1x32x2x129
            new (&inputTensors_[tensorCount++]) Ort::Value(
                Ort::Value::CreateTensor<float>(*memoryInfo_,
                    deconvBufInput_, 32 * 2 * 129, deconvBufShape_.data(), deconvBufShape_.size()));
            
            tensorsInitialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    // Prepare input data for inference (no allocation, just data copying)
    void prepareInputData(const struct FFTData& fftData) {
        // Prepare FFT data for ONNX (format: Real{channel1} Real{channel2} Imaginary{Channel1} Imaginary{Channel2})
        for (size_t i = 0; i < FFT_SIZE; i++) {
            onnxInputBuffer_[i * 4 + 0] = static_cast<float>(fftData.real[i]);  // Ch1 real
            onnxInputBuffer_[i * 4 + 1] = static_cast<float>(fftData.real[i]);  // Ch2 real (duplicate)
            onnxInputBuffer_[i * 4 + 2] = static_cast<float>(fftData.imag[i]);  // Ch1 imag
            onnxInputBuffer_[i * 4 + 3] = static_cast<float>(fftData.imag[i]);  // Ch2 imag (duplicate)
        }
    }
    
    // Get pre-allocated input tensors
    Ort::Value* getInputTensors() {
        return tensorsInitialized_ ? inputTensors_.data() : nullptr;
    }
    
    // Get input names
    const char* const* getInputNames() const {
        return inputNames_.data();
    }
    
    // Get output names
    const char* const* getOutputNames() const {
        return outputNames_.data();
    }
    
    // Get tensor counts
    static constexpr size_t getInputCount() { return 16; }
    static constexpr size_t getOutputCount() { return 15; }
    
    // Update state buffers after inference (no allocation, just data copying)
    void updateStateBuffers(std::vector<Ort::Value>& outputTensors) {
        if (outputTensors.size() < 15) return;
        
        try {
            // out::conv_buf -> conv_buf
            float* convBufOut = outputTensors[1].GetTensorMutableData<float>();
            std::memcpy(convBufInput_, convBufOut, 4 * 2 * 129 * sizeof(float));
            
            // out::block_bufs -> block_bufs
            for (int i = 0; i < 6; i++) {
                float* h0Out = outputTensors[2 + i * 2].GetTensorMutableData<float>();
                float* c0Out = outputTensors[2 + i * 2 + 1].GetTensorMutableData<float>();
                
                std::memcpy(blockBufsInput_[i][0], h0Out, 129 * 32 * sizeof(float));
                std::memcpy(blockBufsInput_[i][1], c0Out, 129 * 32 * sizeof(float));
            }
            
            // out::deconv_buf -> deconv_buf
            float* deconvBufOut = outputTensors[14].GetTensorMutableData<float>();
            std::memcpy(deconvBufInput_, deconvBufOut, 32 * 2 * 129 * sizeof(float));
            
        } catch (const std::exception& e) {
            // Handle error silently to avoid disrupting audio processing
        }
    }
    
    // Reset state buffers (call during initialization)
    void resetStateBuffers() {
        std::memset(convBufInput_, 0, sizeof(convBufInput_));
        std::memset(blockBufsInput_, 0, sizeof(blockBufsInput_));
        std::memset(deconvBufInput_, 0, sizeof(deconvBufInput_));
    }
    
    ~ONNXTensorManager() {
        // Explicitly destroy tensors if they were initialized
        if (tensorsInitialized_) {
            for (auto& tensor : inputTensors_) {
                tensor.~Value();
            }
        }
    }
};

#endif // ONNX_TENSOR_MANAGER_H 