#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <fftw3.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "audio_manager.h"
#include "audio_config.h"
#include "static_buffers.h"
#include "ringbuffer.h"

// Passthrough mode: when enabled, bypasses ONNX inference and directly passes input to output
// Set to 1 to enable passthrough, 0 to use normal ONNX processing
#define ENABLE_PASSTHROUGH 0

// Inference mode: when ENABLE_PASSTHROUGH is 0 and ENABLE_INFERENCE is 0, 
// we do all FFT/IFFT processing but bypass ONNX inference (clone FFT output 5 times)
// Set to 1 to enable ONNX inference, 0 to bypass inference but keep FFT processing
#define ENABLE_INFERENCE 0

// Thread-safe queue for audio data
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;

public:
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        condition_.notify_one();
    }

    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = queue_.front();
        queue_.pop();
        return true;
    }

    void waitAndPop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        item = queue_.front();
        queue_.pop();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// Audio frame structure
struct AudioFrame {
    float data[CHUNK_SIZE];
    
    AudioFrame() {
        std::fill(data, data + CHUNK_SIZE, 0.0f);
    }
    
    AudioFrame(const float* inputData) {
        std::copy(inputData, inputData + CHUNK_SIZE, data);
    }
};

// FFT data structure
struct FFTData {
    double real[FFT_SIZE];
    double imag[FFT_SIZE];
    
    FFTData() {
        std::fill(real, real + FFT_SIZE, 0.0);
        std::fill(imag, imag + FFT_SIZE, 0.0);
    }
};

// ONNX output structure
struct ONNXOutput {
    float classData[5][258];  // 5 classes, each with 258 elements
    
    ONNXOutput() {
        for (int i = 0; i < 5; i++) {
            std::fill(classData[i], classData[i] + 258, 0.0f);
        }
    }
};

class RealtimeProcessor {
public:
    RealtimeProcessor() : 
        audioInputQueue_(),
        fftQueue_(),
        onnxOutputQueue_(),
        audioOutputQueue_(),
        stopProcessing_(false),
        ringBuffer_(CHUNK_SIZE * sizeof(float)) {
        
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
            std::fill(classCurrentBuffer_[classIdx], classCurrentBuffer_[classIdx] + BUFFER_SIZE, 0.0);
            std::fill(classIfftResult_[classIdx], classIfftResult_[classIdx] + WINDOW_SIZE, 0.0);
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                std::fill(classIstftContextBuffers_[classIdx][bufIdx], 
                         classIstftContextBuffers_[classIdx][bufIdx] + ISTFT_OUTPUT_SIZE, 0.0);
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
            
            // Create memory info
            memoryInfo_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            
                         // Initialize input buffers
             std::fill(embeddingInput_, embeddingInput_ + 5, 1.0f);  // 1x5 filled with 1s
             std::fill(convBufInput_, convBufInput_ + (4 * 2 * 129), 0.0f);  // 1x4x2x129
             std::fill(deconvBufInput_, deconvBufInput_ + (32 * 2 * 129), 0.0f);  // 1x32x2x129
             
             // Initialize block buffers
             for (int i = 0; i < 6; i++) {
                 std::fill(blockBufsInput_[i][0], blockBufsInput_[i][0] + (129 * 32), 0.0f);  // h0: 1x129x32
                 std::fill(blockBufsInput_[i][1], blockBufsInput_[i][1] + (129 * 32), 0.0f);  // c0: 1x129x32
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
        for (size_t i = 0; i < std::min(numSamples / 2, (size_t)CHUNK_SIZE); i++) {
            frame.data[i] = (stereoData[i * 2] + stereoData[i * 2 + 1]) * 0.5f;
        }
        
#if ENABLE_PASSTHROUGH
        // In passthrough mode, directly push to output queue
        audioOutputQueue_.push(frame);
#else
        // Push to input queue for processing
        audioInputQueue_.push(frame);
#endif
    }
    
    // Called by audio manager in real-time audio callback
    bool popOutputAudio(float* outputData, size_t numSamples) {
        AudioFrame frame;
        if (audioOutputQueue_.tryPop(frame)) {
            std::copy(frame.data, frame.data + std::min(numSamples, (size_t)CHUNK_SIZE), outputData);
            return true;
        }
        
        // If no output available, fill with silence
        std::fill(outputData, outputData + numSamples, 0.0f);
        return false;
    }

private:
    // Thread-safe queues
    ThreadSafeQueue<AudioFrame> audioInputQueue_;
    ThreadSafeQueue<FFTData> fftQueue_;
    ThreadSafeQueue<ONNXOutput> onnxOutputQueue_;
    ThreadSafeQueue<AudioFrame> audioOutputQueue_;
    
    // Threading
    std::atomic<bool> stopProcessing_;
    std::thread fftThread_;
    std::thread onnxThread_;
    std::thread synthesisThread_;
    
    // Ring buffer for sliding window
    RingBuffer ringBuffer_;
    
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
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
    
         // ONNX input/output buffers (pre-allocated, no dynamic allocation)
     float embeddingInput_[5];
     float convBufInput_[4 * 2 * 129];  // 1x4x2x129
     float blockBufsInput_[6][2][129 * 32];  // 6 blocks, 2 buffers each (h0, c0): 1x129x32
     float deconvBufInput_[32 * 2 * 129];  // 1x32x2x129
     
     // Pre-allocated buffers for ONNX inference (no allocation in real-time)
     float onnxInputBuffer_[4 * 129];  // 4 channels * 129 frequency bins
     
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
            if (audioInputQueue_.tryPop(inputFrame)) {
                // Add frame to ring buffer
                ringBuffer_.write(reinterpret_cast<char*>(inputFrame.data));
                
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
                for (size_t i = 0; i < FFT_SIZE; i++) {
                    fftData.real[i] = fftOut_[i][0];  // Real part
                    fftData.imag[i] = fftOut_[i][1];  // Imaginary part
                }
                
                std::cout << "[FFT_THREAD] Processed frame, queue size: " << fftQueue_.size() << std::endl;
                
#if ENABLE_INFERENCE
                // Send to ONNX inference queue
                fftQueue_.push(fftData);
#else
                // Bypass inference: create fake ONNX output
                ONNXOutput onnxOutput;
                for (int classIdx = 0; classIdx < 5; classIdx++) {
                    // Copy real parts (first 129 elements)
                    for (size_t i = 0; i < FFT_SIZE; i++) {
                        onnxOutput.classData[classIdx][i] = static_cast<float>(fftData.real[i]);
                    }
                    // Copy imaginary parts (next 129 elements)
                    for (size_t i = 0; i < FFT_SIZE; i++) {
                        onnxOutput.classData[classIdx][i + 129] = static_cast<float>(fftData.imag[i]);
                    }
                }
                onnxOutputQueue_.push(onnxOutput);
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
            if (fftQueue_.tryPop(fftData)) {
                std::cout << "[ONNX_THREAD] Processing inference..." << std::endl;
                
                // Run ONNX inference
                ONNXOutput onnxOutput;
                if (runONNXInference(fftData, onnxOutput)) {
                    onnxOutputQueue_.push(onnxOutput);
                    std::cout << "[ONNX_THREAD] Inference completed, output queue size: " << onnxOutputQueue_.size() << std::endl;
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
            if (onnxOutputQueue_.tryPop(onnxOutput)) {
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
                std::fill(outputFrame.data, outputFrame.data + CHUNK_SIZE, 0.0f);
                
                for (int classIdx = 0; classIdx < 5; classIdx++) {
                    for (int i = 0; i < CHUNK_SIZE; i++) {
                        outputFrame.data[i] += classOutputChunks[classIdx][i] * classWeights_[classIdx];
                    }
                }
                
                // Push to output queue
                audioOutputQueue_.push(outputFrame);
                std::cout << "[SYNTHESIS_THREAD] Synthesis completed, output queue size: " << audioOutputQueue_.size() << std::endl;
                         }
             // No sleep - continue immediately to check for new data
        }
        
        std::cout << "[SYNTHESIS_THREAD] Stopped" << std::endl;
    }
    
    bool runONNXInference(const FFTData& fftData, ONNXOutput& output) {
                 try {
             // Prepare FFT data for ONNX (format: real(ch1) imag(ch1) real(ch2) imag(ch1))
             // Use pre-allocated buffer - no dynamic allocation
             for (size_t i = 0; i < FFT_SIZE; i++) {
                 // Channel 1 real, Channel 1 imag, Channel 2 real, Channel 2 imag
                 onnxInputBuffer_[i * 4 + 0] = static_cast<float>(fftData.real[i]);  // Ch1 real
                 onnxInputBuffer_[i * 4 + 1] = static_cast<float>(fftData.imag[i]);  // Ch1 imag
                 onnxInputBuffer_[i * 4 + 2] = static_cast<float>(fftData.real[i]);  // Ch2 real (duplicate)
                 onnxInputBuffer_[i * 4 + 3] = static_cast<float>(fftData.imag[i]);  // Ch2 imag (duplicate)
             }
            
                         // Prepare input tensors (fixed arrays - no dynamic allocation)
             const char* inputNames[16] = {
                 "mixture_tf", "embedding", "conv_buf",
                 "block_bufs::buf0::h0", "block_bufs::buf0::c0",
                 "block_bufs::buf1::h0", "block_bufs::buf1::c0",
                 "block_bufs::buf2::h0", "block_bufs::buf2::c0",
                 "block_bufs::buf3::h0", "block_bufs::buf3::c0",
                 "block_bufs::buf4::h0", "block_bufs::buf4::c0",
                 "block_bufs::buf5::h0", "block_bufs::buf5::c0",
                 "deconv_buf"
             };
             
             const char* outputNames[15] = {
                 "output", "out::conv_buf",
                 "out::block_bufs::buf0::h0", "out::block_bufs::buf0::c0",
                 "out::block_bufs::buf1::h0", "out::block_bufs::buf1::c0",
                 "out::block_bufs::buf2::h0", "out::block_bufs::buf2::c0",
                 "out::block_bufs::buf3::h0", "out::block_bufs::buf3::c0",
                 "out::block_bufs::buf4::h0", "out::block_bufs::buf4::c0",
                 "out::block_bufs::buf5::h0", "out::block_bufs::buf5::c0",
                 "out::deconv_buf"
             };
            
                         // Create input tensors (fixed arrays - no dynamic allocation)
             Ort::Value inputTensors[16];
             int tensorCount = 0;
             
             // mixture_tf: 1x4x1x129
             int64_t mixtureShape[4] = {1, 4, 1, 129};
             inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_, 
                 onnxInputBuffer_, 4 * 129, mixtureShape, 4);
             
             // embedding: 1x5
             int64_t embeddingShape[2] = {1, 5};
             inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_,
                 embeddingInput_, 5, embeddingShape, 2);
             
             // conv_buf: 1x4x2x129
             int64_t convBufShape[4] = {1, 4, 2, 129};
             inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_,
                 convBufInput_, 4 * 2 * 129, convBufShape, 4);
             
             // block_bufs (6 blocks, each with h0 and c0): 1x129x32
             int64_t blockBufShape[3] = {1, 129, 32};
             for (int i = 0; i < 6; i++) {
                 inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_,
                     blockBufsInput_[i][0], 129 * 32, blockBufShape, 3);
                 inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_,
                     blockBufsInput_[i][1], 129 * 32, blockBufShape, 3);
             }
             
             // deconv_buf: 1x32x2x129
             int64_t deconvBufShape[4] = {1, 32, 2, 129};
             inputTensors[tensorCount++] = Ort::Value::CreateTensor<float>(*memoryInfo_,
                 deconvBufInput_, 32 * 2 * 129, deconvBufShape, 4);
            
                         // Run inference
             auto outputTensors = onnxSession_->Run(Ort::RunOptions{nullptr}, inputNames, 
                 inputTensors, 16, outputNames, 15);
            
            // Extract main output (1x5x1x258)
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            
                         // Copy output to result structure
             for (int classIdx = 0; classIdx < 5; classIdx++) {
                 std::copy(outputData + classIdx * 258, outputData + (classIdx + 1) * 258, 
                          output.classData[classIdx]);
             }
            
                         // Update state buffers for next iteration
             updateStateBuffers(outputTensors.data());
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[ONNX] Error in inference: " << e.what() << std::endl;
            return false;
        }
    }
    
         void updateStateBuffers(Ort::Value* outputTensors) {
         // Copy output buffers back to input buffers for next iteration
         
         // out::conv_buf -> conv_buf
         float* convBufOut = outputTensors[1].GetTensorMutableData<float>();
         std::copy(convBufOut, convBufOut + (4 * 2 * 129), convBufInput_);
         
         // out::block_bufs -> block_bufs
         for (int i = 0; i < 6; i++) {
             float* h0Out = outputTensors[2 + i * 2].GetTensorMutableData<float>();
             float* c0Out = outputTensors[2 + i * 2 + 1].GetTensorMutableData<float>();
             
             std::copy(h0Out, h0Out + (129 * 32), blockBufsInput_[i][0]);
             std::copy(c0Out, c0Out + (129 * 32), blockBufsInput_[i][1]);
         }
         
         // out::deconv_buf -> deconv_buf
         float* deconvBufOut = outputTensors[14].GetTensorMutableData<float>();
         std::copy(deconvBufOut, deconvBufOut + (32 * 2 * 129), deconvBufInput_);
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
        const std::vector<char>& buffer = ringBuffer_.getBuffer();
        size_t writePos = ringBuffer_.getWritePosition();
        
        for (size_t i = 0; i < WINDOW_SIZE * sizeof(float); i++) {
            size_t pos = (writePos - WINDOW_SIZE * sizeof(float) + i + buffer.size()) % buffer.size();
            g_windowBuffer[i] = buffer[pos];
        }
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