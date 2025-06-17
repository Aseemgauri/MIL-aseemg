#ifndef AUDIO_DATA_STRUCTURES_H
#define AUDIO_DATA_STRUCTURES_H

#include "audio_config.h"
#include <cstring>
#include <fftw3.h>

// POD audio frame structure - no expensive constructor
struct AudioFrame {
    float data[CHUNK_SIZE];
    
    // Initialize to zero (call explicitly when needed)
    void clear() {
        std::memset(data, 0, sizeof(data));
    }
    
    // Copy from input data
    void copyFrom(const float* inputData, size_t numSamples) {
        const size_t samplesToCopy = (numSamples < CHUNK_SIZE) ? numSamples : CHUNK_SIZE;
        std::memcpy(data, inputData, samplesToCopy * sizeof(float));
        
        // Zero remaining if needed
        if (samplesToCopy < CHUNK_SIZE) {
            std::memset(data + samplesToCopy, 0, (CHUNK_SIZE - samplesToCopy) * sizeof(float));
        }
    }
};

// POD FFT data structure - no expensive constructor
struct FFTData {
    double real[FFT_OUT_SIZE];
    double imag[FFT_OUT_SIZE];
    
    // Initialize to zero (call explicitly when needed)
    void clear() {
        std::memset(real, 0, sizeof(real));
        std::memset(imag, 0, sizeof(imag));
    }
    
    // Copy from FFTW output
    void copyFromFFTW(const fftw_complex* fftOut) {
        for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
            real[i] = fftOut[i][0];  // Real part
            imag[i] = fftOut[i][1];  // Imaginary part
        }
    }
};

// POD ONNX output structure - no expensive constructor
struct ONNXOutput {
    float classData[5][258];  // 5 classes, each with 258 elements
    
    // Initialize to zero (call explicitly when needed)
    void clear() {
        std::memset(classData, 0, sizeof(classData));
    }
    
    // Copy from ONNX output data
    void copyFromONNX(const float* outputData) {
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            std::memcpy(classData[classIdx], outputData + classIdx * 258, 258 * sizeof(float));
        }
    }
    
    // Create fake output by duplicating FFT data across all classes
    void createFakeOutput(const FFTData& fftData) {
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            // Copy real parts (first 129 elements)
            for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
                classData[classIdx][i] = static_cast<float>(fftData.real[i]);
            }
            // Copy imaginary parts (next 129 elements)
            for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
                classData[classIdx][i + 129] = static_cast<float>(fftData.imag[i]);
            }
        }
    }
};

// Pre-allocated pool for reusing data structures
template<typename T, size_t PoolSize>
class StaticPool {
private:
    alignas(MEMORY_ALIGNMENT) T pool_[PoolSize];
    bool used_[PoolSize];
    
public:
    StaticPool() {
        // Initialize all as unused
        std::memset(used_, false, sizeof(used_));
        
        // Clear all pool objects
        for (size_t i = 0; i < PoolSize; ++i) {
            if constexpr (std::is_same_v<T, AudioFrame>) {
                pool_[i].clear();
            } else if constexpr (std::is_same_v<T, FFTData>) {
                pool_[i].clear();
            } else if constexpr (std::is_same_v<T, ONNXOutput>) {
                pool_[i].clear();
            }
        }
    }
    
    // Get an unused object from the pool
    T* acquire() {
        for (size_t i = 0; i < PoolSize; ++i) {
            if (!used_[i]) {
                used_[i] = true;
                return &pool_[i];
            }
        }
        return nullptr;  // Pool exhausted
    }
    
    // Return an object to the pool
    void release(T* obj) {
        if (!obj) return;
        
        // Find the object in the pool
        for (size_t i = 0; i < PoolSize; ++i) {
            if (&pool_[i] == obj) {
                used_[i] = false;
                
                // Clear the object for reuse
                if constexpr (std::is_same_v<T, AudioFrame>) {
                    obj->clear();
                } else if constexpr (std::is_same_v<T, FFTData>) {
                    obj->clear();
                } else if constexpr (std::is_same_v<T, ONNXOutput>) {
                    obj->clear();
                }
                break;
            }
        }
    }
    
    // Get pool statistics
    size_t getUsedCount() const {
        size_t count = 0;
        for (size_t i = 0; i < PoolSize; ++i) {
            if (used_[i]) count++;
        }
        return count;
    }
    
    static constexpr size_t getPoolSize() {
        return PoolSize;
    }
};

#endif // AUDIO_DATA_STRUCTURES_H 