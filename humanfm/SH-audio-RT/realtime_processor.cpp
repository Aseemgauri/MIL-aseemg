#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <fftw3.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "audio_manager.h"
#include "audio_config.h"
#include "static_buffers.h"
#include "unified_ringbuffer.h"
#include "audio_data_structures.h"
#include "onnx_tensor_manager.h"

// Passthrough mode: when enabled, bypasses ONNX inference and directly passes input to output
// Set to 1 to enable passthrough, 0 to use normal ONNX processing
#define ENABLE_PASSTHROUGH 0

// Inference mode: when ENABLE_PASSTHROUGH is 0 and ENABLE_INFERENCE is 0, 
// we do all FFT/IFFT processing but bypass ONNX inference (clone FFT output 5 times)
// Set to 1 to enable ONNX inference, 0 to bypass inference but keep FFT processing
#define ENABLE_INFERENCE 1

// Ring buffer capacities (must be powers of 2)
constexpr size_t AUDIO_FRAME_BUFFER_SIZE = 64;   // 64 audio frames
constexpr size_t FFT_DATA_BUFFER_SIZE = 32;      // 32 FFT frames  
constexpr size_t ONNX_OUTPUT_BUFFER_SIZE = 32;   // 32 ONNX outputs

class RealtimeProcessor {
public:
    RealtimeProcessor() : 
        stopProcessing_(false),
        ringBuffer_() {
        
        // Initialize windows if not already initialized
        static bool windowsInitialized = false;
        if (!windowsInitialized) {
            initializeWindows();
            windowsInitialized = true;
        }
        initializeFFTW();
        
        // Initialize class-specific overlap-add state
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            classLookbackBufIdx_[classIdx] = 0;
            std::memset(classCurrentBuffer_[classIdx], 0, BUFFER_SIZE * sizeof(double));
            std::memset(classIfftResult_[classIdx], 0, WINDOW_SIZE * sizeof(double));
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                std::memset(classIstftContextBuffers_[classIdx][bufIdx], 0, 
                           ISTFT_OUTPUT_SIZE * sizeof(double));
            }
        }
        
                 // Initialize weighted sum coefficients
         for (int i = 0; i < 5; i++) {
             classWeights_[i] = 0.2f; // Equal weights for now
         }
    }

    ~RealtimeProcessor() {
        stop();
        cleanupFFTW();
    }

    bool initializeONNX(const std::string& modelPath) {
        try {
            // Initialize ONNX Runtime environment
            onnxEnv_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RealtimeProcessor");
            
            // Create session options
            sessionOptions_ = std::make_unique<Ort::SessionOptions>();
            sessionOptions_->SetIntraOpNumThreads(1);
            sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // Load the model
            onnxSession_ = std::make_unique<Ort::Session>(*onnxEnv_, modelPath.c_str(), *sessionOptions_);
            
            // Initialize tensor manager
            if (!tensorManager_.initialize()) {
                throw std::runtime_error("Failed to initialize tensor manager");
            }
            
            std::cout << "[RealtimeProcessor] ONNX model loaded successfully from: " << modelPath << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RealtimeProcessor] Error initializing ONNX: " << e.what() << std::endl;
            return false;
        }
    }
    
    void start() {
        stopProcessing_ = false;
        
        // Start processing threads based on configuration
#if !ENABLE_PASSTHROUGH
        // Start FFT processing thread
        fftThread_ = std::thread(&RealtimeProcessor::fftProcessingLoop, this);
        
        // Start synthesis thread
        synthesisThread_ = std::thread(&RealtimeProcessor::synthesisLoop, this);
        
#if ENABLE_INFERENCE
        // Start ONNX inference thread
        onnxThread_ = std::thread(&RealtimeProcessor::onnxInferenceLoop, this);
#endif
#endif
        
        std::cout << "[RealtimeProcessor] Processing threads started" << std::endl;
    }
    
    void stop() {
        stopProcessing_ = true;
        
        // Join all threads
        if (fftThread_.joinable()) {
            fftThread_.join();
        }
        if (onnxThread_.joinable()) {
            onnxThread_.join();
        }
        if (synthesisThread_.joinable()) {
            synthesisThread_.join();
        }
        
        std::cout << "[RealtimeProcessor] All processing threads stopped" << std::endl;
    }
    
    // Called by audio manager in real-time audio callback
    void pushInputAudio(const float* stereoData, size_t numSamples) {
        // Convert stereo to mono quickly in audio callback
        AudioFrame frame;
        frame.clear();  // Explicit clear instead of constructor
        
        const size_t samplesToProcess = std::min(numSamples / 2, (size_t)CHUNK_SIZE);
        for (size_t i = 0; i < samplesToProcess; i++) {
            frame.data[i] = (stereoData[i * 2] + stereoData[i * 2 + 1]) * 0.5f;
        }
        
#if ENABLE_PASSTHROUGH
        // In passthrough mode, directly push to output buffer
        audioOutputBuffer_.tryPush(frame);
#else
        // Push to input buffer for processing
        audioInputBuffer_.tryPush(frame);
#endif
    }
    
    // Called by audio manager in real-time audio callback
    bool popOutputAudio(float* outputData, size_t numSamples) {
        AudioFrame frame;
        if (audioOutputBuffer_.tryPop(frame)) {
            std::copy(frame.data, frame.data + std::min(numSamples, (size_t)CHUNK_SIZE), outputData);
            return true;
        }
        
        // If no output available, fill with silence
        std::fill(outputData, outputData + numSamples, 0.0f);
        return false;
    }

private:
    // Lock-free ring buffers for inter-thread communication
    LockFreeRingBuffer<AudioFrame, AUDIO_FRAME_BUFFER_SIZE> audioInputBuffer_;
    LockFreeRingBuffer<FFTData, FFT_DATA_BUFFER_SIZE> fftBuffer_;
    LockFreeRingBuffer<ONNXOutput, ONNX_OUTPUT_BUFFER_SIZE> onnxOutputBuffer_;
    LockFreeRingBuffer<AudioFrame, AUDIO_FRAME_BUFFER_SIZE> audioOutputBuffer_;
    
    // Threading
    std::atomic<bool> stopProcessing_;
    std::thread fftThread_;
    std::thread onnxThread_;
    std::thread synthesisThread_;
    
    // Ring buffer for sliding window
    SlidingWindowBuffer ringBuffer_;
    
    // FFTW variables
    double* fftIn_;
    fftw_complex* fftOut_;
    fftw_plan fftPlan_;
    fftw_complex* ifftIn_;
    double* ifftOut_;
    fftw_plan ifftPlan_;
    
    // ONNX-related variables
    std::unique_ptr<Ort::Env> onnxEnv_;
    std::unique_ptr<Ort::Session> onnxSession_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    
    // Pre-allocated ONNX tensor manager
    ONNXTensorManager tensorManager_;
     
     // Class-specific overlap-add state buffers
     int8_t classLookbackBufIdx_[5];
     double classIstftContextBuffers_[5][ISTFT_LOOKBACK_BUFFERS][ISTFT_OUTPUT_SIZE];
     double classCurrentBuffer_[5][BUFFER_SIZE];
     double classIfftResult_[5][WINDOW_SIZE];
     
     // Weighted sum coefficients (pre-allocated)
     float classWeights_[5];
    
    void initializeFFTW() {
        // Initialize FFT
        fftIn_ = fftw_alloc_real(WINDOW_SIZE);
        fftOut_ = fftw_alloc_complex(FFT_SIZE);
        fftPlan_ = fftw_plan_dft_r2c_1d(WINDOW_SIZE, fftIn_, fftOut_, FFTW_ESTIMATE);
        
        // Initialize IFFT
        ifftIn_ = fftw_alloc_complex(FFT_SIZE);
        ifftOut_ = fftw_alloc_real(WINDOW_SIZE);
        ifftPlan_ = fftw_plan_dft_c2r_1d(WINDOW_SIZE, ifftIn_, ifftOut_, FFTW_ESTIMATE);
    }

    void cleanupFFTW() {
        fftw_destroy_plan(fftPlan_);
        fftw_destroy_plan(ifftPlan_);
        fftw_free(fftIn_);
        fftw_free(fftOut_);
        fftw_free(ifftIn_);
        fftw_free(ifftOut_);
    }
    
    // FFT processing thread
    void fftProcessingLoop() {
        std::cout << "[FFT_THREAD] Started" << std::endl;
        
        while (!stopProcessing_) {
            AudioFrame inputFrame;
            
            // Wait for input audio data
            if (audioInputBuffer_.tryPop(inputFrame)) {
                // Add frame to ring buffer
                ringBuffer_.writeChunk(reinterpret_cast<char*>(inputFrame.data), CHUNK_SIZE * sizeof(float));
                
                // Get windowed samples for FFT
                getWindowSamples();
                
                // Apply analysis window and copy to FFT input
                for (size_t i = 0; i < WINDOW_SIZE; i++) {
                    float sample = *reinterpret_cast<const float*>(&g_windowBuffer[i * sizeof(float)]);
                    fftIn_[i] = static_cast<double>(sample) * g_analysisWindow[i];
                }
                
                // Perform FFT
                fftw_execute(fftPlan_);
                
                // Store FFT result
                FFTData fftData;
                fftData.copyFromFFTW(fftOut_);
                
                std::cout << "[FFT_THREAD] Processed frame, buffer size: " << fftBuffer_.size() << std::endl;
                
#if ENABLE_INFERENCE
                // Send to ONNX inference buffer
                fftBuffer_.tryPush(fftData);
#else
                // Bypass inference: create fake ONNX output
                ONNXOutput onnxOutput;
                onnxOutput.createFakeOutput(fftData);
                onnxOutputBuffer_.tryPush(onnxOutput);
                std::cout << "[FFT_THREAD] Bypassed inference, created fake output" << std::endl;
#endif
                         }
             // No sleep - continue immediately to check for new data
        }
        
        std::cout << "[FFT_THREAD] Stopped" << std::endl;
    }
    
    // ONNX inference thread
    void onnxInferenceLoop() {
        std::cout << "[ONNX_THREAD] Started" << std::endl;
        
        while (!stopProcessing_) {
            FFTData fftData;
            
            // Wait for FFT data
            if (fftBuffer_.tryPop(fftData)) {
                std::cout << "[ONNX_THREAD] Processing inference..." << std::endl;
                
                // Run ONNX inference
                ONNXOutput onnxOutput;
                if (runONNXInference(fftData, onnxOutput)) {
                    onnxOutputBuffer_.tryPush(onnxOutput);
                    std::cout << "[ONNX_THREAD] Inference completed, output buffer size: " << onnxOutputBuffer_.size() << std::endl;
                } else {
                    std::cerr << "[ONNX_THREAD] Inference failed" << std::endl;
                }
                         }
             // No sleep - continue immediately to check for new data
        }
        
        std::cout << "[ONNX_THREAD] Stopped" << std::endl;
    }
    
    // Synthesis thread (IFFT + overlap-add)
    void synthesisLoop() {
        std::cout << "[SYNTHESIS_THREAD] Started" << std::endl;
        
        while (!stopProcessing_) {
            ONNXOutput onnxOutput;
            
            // Wait for ONNX output
            if (onnxOutputBuffer_.tryPop(onnxOutput)) {
                std::cout << "[SYNTHESIS_THREAD] Processing synthesis..." << std::endl;
                
                // Process each class
                float classOutputChunks[5][CHUNK_SIZE];
                for (int classIdx = 0; classIdx < 5; classIdx++) {
                    // Perform IFFT for this class
                    performIFFTForClass(classIdx, onnxOutput.classData[classIdx]);
                    
                    // Perform overlap-add synthesis for this class
                    performOverlapAddForClass(classIdx, classOutputChunks[classIdx]);
                }
                
                // Compute weighted sum of all classes
                AudioFrame outputFrame;
                outputFrame.clear();
                
                for (int classIdx = 0; classIdx < 5; classIdx++) {
                    for (int i = 0; i < CHUNK_SIZE; i++) {
                        outputFrame.data[i] += classOutputChunks[classIdx][i] * classWeights_[classIdx];
                    }
                }
                
                // Push to output buffer
                audioOutputBuffer_.tryPush(outputFrame);
                std::cout << "[SYNTHESIS_THREAD] Synthesis completed, output buffer size: " << audioOutputBuffer_.size() << std::endl;
                         }
             // No sleep - continue immediately to check for new data
        }
        
        std::cout << "[SYNTHESIS_THREAD] Stopped" << std::endl;
    }
    
    bool runONNXInference(const FFTData& fftData, ONNXOutput& output) {
        try {
            // Prepare input data using tensor manager (no allocation)
            tensorManager_.prepareInputData(fftData);
            
            // Get pre-allocated tensors
            Ort::Value* inputTensors = tensorManager_.getInputTensors();
            if (!inputTensors) {
                return false;
            }
            
            // Run inference using pre-allocated tensors
            auto outputTensors = onnxSession_->Run(Ort::RunOptions{nullptr}, 
                tensorManager_.getInputNames(), inputTensors, tensorManager_.getInputCount(),
                tensorManager_.getOutputNames(), tensorManager_.getOutputCount());
            
            // Extract output data using the new method
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            output.copyFromONNX(outputData);
            
            // Update state buffers for next iteration (no allocation)
            tensorManager_.updateStateBuffers(outputTensors);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[ONNX] Error in inference: " << e.what() << std::endl;
            return false;
        }
    }
    

    
         void performIFFTForClass(int classIdx, const float* classOutput) {
        // Split real and imaginary parts (first 129 real, next 129 imaginary)
        for (size_t i = 0; i < FFT_SIZE; i++) {
            ifftIn_[i][0] = static_cast<double>(classOutput[i]);           // Real part
            ifftIn_[i][1] = static_cast<double>(classOutput[i + 129]);     // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan_);
        
        // Store IFFT result in class-specific buffer (normalize by window size)
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            classIfftResult_[classIdx][i] = ifftOut_[i] / WINDOW_SIZE;
        }
    }
    
    void performOverlapAddForClass(int classIdx, float* outputChunk) {
        // Store the current IFFT result in the class-specific current buffer
        std::copy(classIfftResult_[classIdx], classIfftResult_[classIdx] + BUFFER_SIZE, classCurrentBuffer_[classIdx]);

        // Copy the last ISTFT_OUTPUT_SIZE frames to the class-specific context buffer
        double* ctxPtr = classIstftContextBuffers_[classIdx][classLookbackBufIdx_[classIdx]];
        std::copy(classIfftResult_[classIdx] + (WINDOW_SIZE - ISTFT_OUTPUT_SIZE),
                 classIfftResult_[classIdx] + WINDOW_SIZE,
                 ctxPtr);

        // Do overlap-add for the current chunk using class-specific buffers
        for (int j = 0; j < CHUNK_SIZE; j++) {
            // Accumulate result from all relevant lookback buffers for this class
            double result = 0.0;
            for (int lookbackOffset = 0; lookbackOffset < ISTFT_LOOKBACK_BUFFERS; lookbackOffset++) {
                // Get the correct buffer index (going backwards in time)
                int bufIdx = (classLookbackBufIdx_[classIdx] - lookbackOffset + ISTFT_LOOKBACK_BUFFERS) % ISTFT_LOOKBACK_BUFFERS;
                double* historyPtr = classIstftContextBuffers_[classIdx][bufIdx];
                
                // Calculate the index in the history buffer
                int ctxIdx = j + lookbackOffset * CHUNK_SIZE;
                if (ctxIdx < ISTFT_OUTPUT_SIZE) {
                    // Apply synthesis window and accumulate
                    result += historyPtr[ctxIdx] * g_synthesisWindow[ctxIdx];
                }
            }
            
            // Store the result in output chunk
            outputChunk[j] = static_cast<float>(result);
        }

        // Advance the class-specific lookback buffer index
        classLookbackBufIdx_[classIdx] = (classLookbackBufIdx_[classIdx] + 1) % ISTFT_LOOKBACK_BUFFERS;
    }
    
    void getWindowSamples() {
        ringBuffer_.getWindowData(g_windowBuffer, WINDOW_SIZE * sizeof(float));
    }
};

// Main function for real-time processing
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model_file.onnx>" << std::endl;
        return 1;
    }

    // Initialize audio manager
    AudioManager audioManager;
    if (!audioManager.initialize()) {
        std::cerr << "Failed to initialize audio manager" << std::endl;
        return 1;
    }

    // Initialize realtime processor
    RealtimeProcessor processor;
    
#if !ENABLE_PASSTHROUGH && ENABLE_INFERENCE
    if (!processor.initializeONNX(argv[1])) {
        std::cerr << "Failed to initialize ONNX model" << std::endl;
        return 1;
    }
#else
    #if ENABLE_PASSTHROUGH
        std::cout << "[PASSTHROUGH] Skipping ONNX initialization - passthrough mode enabled" << std::endl;
    #else
        std::cout << "[FFT_ONLY] Skipping ONNX initialization - inference disabled, FFT processing only" << std::endl;
    #endif
#endif

    // Start processing threads
    processor.start();

    // Start audio processing
    if (!audioManager.start()) {
        std::cerr << "Failed to start audio manager" << std::endl;
        return 1;
    }

    std::cout << "Real-time audio processing started. Press Ctrl+C to stop." << std::endl;

    // Main processing loop - now just handles audio I/O
    try {
        while (true) {
            // Check if we have input data
            if (audioManager.getAvailableInputSamples() >= CHUNK_SIZE * 2) {
                float stereoBuffer[CHUNK_SIZE * 2];
                if (audioManager.getInputData(stereoBuffer, CHUNK_SIZE * 2)) {
                    // Push to processor (non-blocking)
                    processor.pushInputAudio(stereoBuffer, CHUNK_SIZE * 2);
                }
            }
            
            // Check if we need output data
            if (audioManager.getAvailableOutputSpace() >= CHUNK_SIZE) {
                float outputBuffer[CHUNK_SIZE];
                if (processor.popOutputAudio(outputBuffer, CHUNK_SIZE)) {
                    // Write to audio output
                    audioManager.writeOutputData(outputBuffer, CHUNK_SIZE);
                }
                         }
             
             // No sleep - audio I/O should be as responsive as possible
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in processing loop: " << e.what() << std::endl;
    }

    // Cleanup
    processor.stop();
    audioManager.stop();
    audioManager.cleanup();

    std::cout << "Real-time audio processing stopped." << std::endl;
    return 0;
} 