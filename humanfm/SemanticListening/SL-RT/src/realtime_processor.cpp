#include <iostream>
#include <cstring>
#include <algorithm>
#include <fftw3.h>
#include <thread>
#include <chrono>
#include "audio_manager.h"
#include "audio_config.h"
#include "static_buffers.h"
#include "SL_Model.h"
#include "sl_config.h"  // Include the static configuration (auto-generated)

#if !SL_FILL_EMBEDDING
#include "humanfm_client.hpp"
#endif

class RealtimeProcessor {
public:
    RealtimeProcessor() 
#if !SL_FILL_EMBEDDING
        : humanfmClient_()
#endif
    {
        
        // Initialize windows if not already initialized
        static bool windowsInitialized = false;
        if (!windowsInitialized) {
            initializeWindows();
            windowsInitialized = true;
        }
        initializeFFTW();
        
        // Initialize class-specific overlap-add state
        for (int classIdx = 0; classIdx < SYSTEM_NUM_CLASSES; classIdx++) {
            classLookbackBufIdx_[classIdx] = 0;
            std::memset(classIfftResult_[classIdx], 0, NFFT * sizeof(double));
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                std::memset(classIstftContextBuffers_[classIdx][bufIdx], 0, 
                           ISTFT_OUTPUT_SIZE * sizeof(double));
            }
        }
        
        // Initialize weighted sum coefficients
        for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
#if SL_ENABLE_INFERENCE
            classWeights_[i] = 1.0f; // Full weights when inference is enabled
#else
            classWeights_[i] = 1.0f / SYSTEM_NUM_CLASSES; // Average weights when inference is disabled
#endif
        }
        
        // Initialize embedding to all ones (default semantic embedding)
        for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
            embeddingBuffer_[i] = 1.0f;  // Default fallback value
        }
        
#if SL_FILL_EMBEDDING
        // Test mode: Set all embedding and weights to 1.0 and skip background thread
        for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
            embeddingBuffer_[i] = 1.0f;
            classWeights_[i] = 1.0f;
        }
        std::cout << "[SL_FILL_EMBEDDING] Test mode active - all embeddings and weights set to 1.0" << std::endl;
#else
        // Start the background thread to update from HumanFM server
        updateThread_ = std::thread(&RealtimeProcessor::updateFromServer, this);
#endif
        
        // Pre-allocate all processing buffers
        std::memset(fftInputBuffer_, 0, sizeof(fftInputBuffer_));
        std::memset(slModelOutputBuffer_, 0, sizeof(slModelOutputBuffer_));
#if SL_ENABLE_PASSTHROUGH
        std::memset(currentAudioFrame_, 0, sizeof(currentAudioFrame_));
#endif
        std::memset(outputAudioFrame_, 0, sizeof(outputAudioFrame_));
        std::memset(leftChannelFrame_, 0, sizeof(leftChannelFrame_));
        std::memset(rightChannelFrame_, 0, sizeof(rightChannelFrame_));
        std::memset(leftFftOut_, 0, sizeof(leftFftOut_));
        std::memset(rightFftOut_, 0, sizeof(rightFftOut_));
        std::memset(leftInputRollingBuffer_, 0, sizeof(leftInputRollingBuffer_));
        std::memset(rightInputRollingBuffer_, 0, sizeof(rightInputRollingBuffer_));
    }

    ~RealtimeProcessor() {
#if !SL_FILL_EMBEDDING
        stopUpdateThread_ = true;
        if (updateThread_.joinable()) {
            updateThread_.join();
        }
#endif
        cleanupFFTW();
    }

    bool initializeSLModel(const std::string& modelPath) {
        try {
            // Initialize SL Model
            slModel_ = std::make_unique<SL_Model>(modelPath);
            
            std::cout << "[RealtimeProcessor] SL Model loaded successfully from: " << modelPath << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[RealtimeProcessor] Error initializing SL Model: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Method to set embedding values for semantic processing
    void setEmbedding(const float* embedding) {
        if (embedding) {
            std::memcpy(embeddingBuffer_, embedding, SYSTEM_NUM_CLASSES * sizeof(float));
        }
    }
    
    // Method to set embedding for a specific class
    void setClassEmbedding(int classIndex, float value) {
        if (classIndex >= 0 && classIndex < SYSTEM_NUM_CLASSES) {
            embeddingBuffer_[classIndex] = value;
        }
    }
    
    // Single-threaded audio processing - called by audio manager in real-time callback
    void processAudio(const float* stereoInput, float* monoOutput, size_t numSamples) {
        // Input is already deinterleaved: left samples first, then right samples
        const size_t samplesToProcess = std::min(numSamples / 2, (size_t)CHUNK_SIZE);
        
        // Copy left channel samples (first half of input)
        for (size_t i = 0; i < samplesToProcess; i++) {
            leftChannelFrame_[i] = stereoInput[i];  // Left channel samples are first
        }
        
        // Copy right channel samples (second half of input)
        for (size_t i = 0; i < samplesToProcess; i++) {
            rightChannelFrame_[i] = stereoInput[i + samplesToProcess];  // Right channel samples follow
        }
        
        // Update rolling buffers for both channels
        std::memmove(leftInputRollingBuffer_, leftInputRollingBuffer_ + CHUNK_SIZE, sizeof(float) * (NFFT - CHUNK_SIZE));
        std::memcpy(leftInputRollingBuffer_ + (NFFT - CHUNK_SIZE), leftChannelFrame_, sizeof(float) * CHUNK_SIZE);
        std::memmove(rightInputRollingBuffer_, rightInputRollingBuffer_ + CHUNK_SIZE, sizeof(float) * (NFFT - CHUNK_SIZE));
        std::memcpy(rightInputRollingBuffer_ + (NFFT - CHUNK_SIZE), rightChannelFrame_, sizeof(float) * CHUNK_SIZE);
        
        
        
#if SL_ENABLE_PASSTHROUGH
        // For passthrough mode, mix stereo to mono for output compatibility
        for (size_t i = 0; i < samplesToProcess; i++) {
            currentAudioFrame_[i] = (leftChannelFrame_[i] + rightChannelFrame_[i]) * 0.5f;
        }
        // In passthrough mode, directly copy input to output
        std::copy(currentAudioFrame_, currentAudioFrame_ + samplesToProcess, monoOutput);
#else
        // Process through FFT/SL_Model/IFFT pipeline
        processAudioFrame();
        
        // Copy processed output
        std::copy(outputAudioFrame_, outputAudioFrame_ + samplesToProcess, monoOutput);
#endif
    }

private:
    // FFTW variables
    double* fftIn_;
    fftw_complex* fftOut_;
    fftw_plan fftPlan_;
    fftw_complex* ifftIn_;
    double* ifftOut_;
    fftw_plan ifftPlan_;
    
    // SL_Model
    std::unique_ptr<SL_Model> slModel_;
     
    // Class-specific overlap-add state buffers
    int8_t classLookbackBufIdx_[SYSTEM_NUM_CLASSES];
    double classIstftContextBuffers_[SYSTEM_NUM_CLASSES][ISTFT_LOOKBACK_BUFFERS][ISTFT_OUTPUT_SIZE];
    double classIfftResult_[SYSTEM_NUM_CLASSES][NFFT];
     
    // Weighted sum coefficients (pre-allocated)
    float classWeights_[SYSTEM_NUM_CLASSES];
    
    // Embedding buffer for SL_Model input
    float embeddingBuffer_[SYSTEM_NUM_CLASSES];
    
    // Pre-allocated processing buffers (no dynamic allocation during processing)
    float fftInputBuffer_[4][FFT_OUT_SIZE];  // [0]=left real, [1]=right real, [2]=left imag, [3]=right imag
    float slModelOutputBuffer_[SYSTEM_NUM_CLASSES * FFT_OUT_SIZE * 2];  // SYSTEM_NUM_CLASSES classes * 258 frequency bins for SL_Model output
    float currentAudioFrame_[CHUNK_SIZE];  // Current input audio frame
    float outputAudioFrame_[CHUNK_SIZE];   // Current output audio frame
    
    // Stereo channel buffers
    float leftChannelFrame_[CHUNK_SIZE];   // Left channel buffer
    float rightChannelFrame_[CHUNK_SIZE];  // Right channel buffer
    
    // Stereo FFT buffers
    fftw_complex leftFftOut_[FFT_OUT_SIZE];
    fftw_complex rightFftOut_[FFT_OUT_SIZE];
    
    float leftInputRollingBuffer_[NFFT];
    float rightInputRollingBuffer_[NFFT];
    
    // HumanFM client (only when not in test mode)
#if !SL_FILL_EMBEDDING
    HumanFMClient humanfmClient_;
    std::thread updateThread_;
    bool stopUpdateThread_ = false;
#endif
    
    void initializeFFTW() {
        // Initialize FFT
        fftIn_ = fftw_alloc_real(NFFT);
        fftOut_ = fftw_alloc_complex(FFT_OUT_SIZE);
        fftPlan_ = fftw_plan_dft_r2c_1d(NFFT, fftIn_, fftOut_, FFTW_ESTIMATE);
        
        // Initialize IFFT
        ifftIn_ = fftw_alloc_complex(FFT_OUT_SIZE);
        ifftOut_ = fftw_alloc_real(NFFT);
        ifftPlan_ = fftw_plan_dft_c2r_1d(NFFT, ifftIn_, ifftOut_, FFTW_ESTIMATE);
    }

    void cleanupFFTW() {
        fftw_destroy_plan(fftPlan_);
        fftw_destroy_plan(ifftPlan_);
        fftw_free(fftIn_);
        fftw_free(fftOut_);
        fftw_free(ifftIn_);
        fftw_free(ifftOut_);
    }
    
    // Single-threaded processing pipeline
    void processAudioFrame() {
        // Process left channel FFT
        // Apply analysis window to left channel using rolling buffer
        for (size_t i = 0; i < NFFT; i++) {
            fftIn_[i] = static_cast<double>(leftInputRollingBuffer_[i]) * g_analysisWindow[i];
        }
        // Remove zero-padding loop for left channel
        // Perform FFT on left channel
        fftw_execute(fftPlan_);
        
        // Store left channel FFT results
        for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
            leftFftOut_[i][0] = fftOut_[i][0];  // Real part
            leftFftOut_[i][1] = fftOut_[i][1];  // Imaginary part
        }
        
        // Process right channel FFT
        // Apply analysis window to right channel using rolling buffer
        for (size_t i = 0; i < NFFT; i++) {
            fftIn_[i] = static_cast<double>(rightInputRollingBuffer_[i]) * g_analysisWindow[i];
        }
        // Remove zero-padding loop for right channel
        // Perform FFT on right channel
        fftw_execute(fftPlan_);
        
        // Store right channel FFT results
        for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
            rightFftOut_[i][0] = fftOut_[i][0];  // Real part
            rightFftOut_[i][1] = fftOut_[i][1];  // Imaginary part
        }
        
        // Prepare FFT data for SL_Model or bypass
#if SL_ENABLE_INFERENCE
        // Run SL_Model inference
        if (runSLModelInference()) {
            // Process each class through IFFT and overlap-add
            processAllClasses();
        } else {
            // Fallback: fill output with silence
            std::fill(outputAudioFrame_, outputAudioFrame_ + CHUNK_SIZE, 0.0f);
        }
#else
        // Bypass inference: create fake output by duplicating FFT data
        createFakeOutput();
        processAllClasses();
#endif
    }
    
    bool runSLModelInference() {
        try {
            // Prepare FFT data for SL_Model with true stereo data
            // Format: fftInputBuffer_[0]=left real, [1]=right real, [2]=left imag, [3]=right imag
            for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
                fftInputBuffer_[0][i] = static_cast<float>(leftFftOut_[i][0]); // left real
                fftInputBuffer_[1][i] = static_cast<float>(rightFftOut_[i][0]); // right real
                fftInputBuffer_[2][i] = static_cast<float>(leftFftOut_[i][1]); // left imag
                fftInputBuffer_[3][i] = static_cast<float>(rightFftOut_[i][1]); // right imag
            }
            
            // Use SL_Model's processFrame method with pre-allocated buffers (flattened pointer)
            // The embeddingBuffer_ is continuously updated by the background thread
            if (!slModel_->processFrame(reinterpret_cast<float*>(fftInputBuffer_), slModelOutputBuffer_, embeddingBuffer_)) {
                std::cerr << "Error in SL_Model processFrame" << std::endl;
                return false;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[SL_MODEL] Error in inference: " << e.what() << std::endl;
            return false;
        }
    }
    
    void createFakeOutput() {
        // Create fake output by mixing stereo FFT data for all SYSTEM_NUM_CLASSES classes
        for (int classIdx = 0; classIdx < SYSTEM_NUM_CLASSES; classIdx++) {
            int outputOffset = classIdx * 258;  // 258 = 129 real + 129 imag
            
            // Mix left and right channels for fake output (copy real parts)
            for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
                // Average left and right real parts
                slModelOutputBuffer_[outputOffset + i] = static_cast<float>(
                    (leftFftOut_[i][0] + rightFftOut_[i][0]) * 0.5);
            }
            
            // Mix left and right channels for fake output (copy imaginary parts)
            for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
                // Average left and right imaginary parts
                slModelOutputBuffer_[outputOffset + 129 + i] = static_cast<float>(
                    (leftFftOut_[i][1] + rightFftOut_[i][1]) * 0.5);
            }
        }
    }
    
    void processAllClasses() {
        // Process each class through IFFT and overlap-add
        float classOutputChunks[SYSTEM_NUM_CLASSES][CHUNK_SIZE];
        for (int classIdx = 0; classIdx < SYSTEM_NUM_CLASSES; classIdx++) {
            // Get pointer to this class's output data
            const float* classOutput = &slModelOutputBuffer_[classIdx * 258];
            
            // Perform IFFT for this class
            performIFFTForClass(classIdx, classOutput);
            
            // Perform overlap-add synthesis for this class
            performOverlapAddForClass(classIdx, classOutputChunks[classIdx]);
        }
        
        // Compute weighted sum of all classes into output frame
        std::fill(outputAudioFrame_, outputAudioFrame_ + CHUNK_SIZE, 0.0f);
        for (int classIdx = 0; classIdx < SYSTEM_NUM_CLASSES; classIdx++) {
            for (int i = 0; i < CHUNK_SIZE; i++) {
                outputAudioFrame_[i] += classOutputChunks[classIdx][i] * classWeights_[classIdx];
            }
        }
    }
    
    void performIFFTForClass(int classIdx, const float* classOutput) {
        // Split real and imaginary parts (first 129 real, next 129 imaginary)
        for (size_t i = 0; i < FFT_OUT_SIZE; i++) {
            ifftIn_[i][0] = static_cast<double>(classOutput[i]);           // Real part
            ifftIn_[i][1] = static_cast<double>(classOutput[i + 129]);     // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan_);
        
        // Store IFFT result in class-specific buffer (normalize by window size)
        for (size_t i = 0; i < NFFT; i++) {
            classIfftResult_[classIdx][i] = ifftOut_[i] / NFFT;
        }
    }
    
    void performOverlapAddForClass(int classIdx, float* outputChunk) {
        // Copy the last ISTFT_OUTPUT_SIZE frames to the class-specific context buffer
        double* ctxPtr = classIstftContextBuffers_[classIdx][classLookbackBufIdx_[classIdx]];
        std::copy(classIfftResult_[classIdx] + (NFFT - ISTFT_OUTPUT_SIZE),
                 classIfftResult_[classIdx] + NFFT,
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


#if !SL_FILL_EMBEDDING
    void updateFromServer() {
        std::cout << "[HumanFM] Background update thread started" << std::endl;
        
        while (!stopUpdateThread_) {
            try {
                // Get class levels (volume/weights) from server
                auto classLevels = humanfmClient_.getClassLevels();
                if (classLevels.size() == SYSTEM_NUM_CLASSES) {
                    for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
                        // Convert level (0-100) to weight (0.0-1.0)
                        classWeights_[i] = classLevels[i] / 100.0f;
                    }
                    std::cout << "[HumanFM] Updated class weights: ";
                    for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
                        std::cout << classWeights_[i] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // Get class detection status (embedding) from server
                auto classDetections = humanfmClient_.getClasses();
                if (classDetections.size() == SYSTEM_NUM_CLASSES) {
                    for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
                        // Convert detection (0 or 1) to embedding value
                        embeddingBuffer_[i] = static_cast<float>(classDetections[i]);
                    }
                    std::cout << "[HumanFM] Updated embedding: ";
                    for (int i = 0; i < SYSTEM_NUM_CLASSES; i++) {
                        std::cout << embeddingBuffer_[i] << " ";
                    }
                    std::cout << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[HumanFM] Error updating from server: " << e.what() << std::endl;
            }
            
            // Update every 2 seconds to stay in sync with server data
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        
        std::cout << "[HumanFM] Background update thread stopped" << std::endl;
    }
#endif


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
    
#if !SL_ENABLE_PASSTHROUGH && SL_ENABLE_INFERENCE
    if (!processor.initializeSLModel(argv[1])) {
        std::cerr << "Failed to initialize SL Model" << std::endl;
        return 1;
    }
#else
    #if SL_ENABLE_PASSTHROUGH
        std::cout << "[PASSTHROUGH] Skipping SL_Model initialization - passthrough mode enabled" << std::endl;
    #else
        std::cout << "[FFT_ONLY] Skipping SL_Model initialization - inference disabled, FFT processing only" << std::endl;
    #endif
#endif

    // Start audio processing
    if (!audioManager.start()) {
        std::cerr << "Failed to start audio manager" << std::endl;
        return 1;
    }

    std::cout << "Real-time audio processing started. Press Ctrl+C to stop." << std::endl;
#if SL_FILL_EMBEDDING
    std::cout << "[SL_FILL_EMBEDDING] Test mode - all weights and embeddings fixed at 1.0" << std::endl;
#else
    std::cout << "[HumanFM] Integration enabled - updating weights and embedding from server every 2 seconds" << std::endl;
#endif
#if !SL_FILL_EMBEDDING
    std::cout << "[HumanFM] Target classes: Speech, Music, Vehicle, Animal, Dog" << std::endl;
#else
    std::cout << "[SL_FILL_EMBEDDING] Target classes: Speech, Music, Vehicle, Animal, Dog (all set to 1.0)" << std::endl;
#endif

    // Main processing loop - single-threaded audio processing
    try {
        while (true) {
            // Check if we have input data
            if (audioManager.getAvailableInputSamples() >= CHUNK_SIZE * 2) {
                float stereoBuffer[CHUNK_SIZE * 2];
                if (audioManager.getInputData(stereoBuffer, CHUNK_SIZE * 2)) {
                    // Check if we have output space
                    if (audioManager.getAvailableOutputSpace() >= CHUNK_SIZE) {
                        float outputBuffer[CHUNK_SIZE];
                        
                        // Process audio in single thread with pre-allocated buffers
                        processor.processAudio(stereoBuffer, outputBuffer, CHUNK_SIZE * 2);
                        
                        // Write to audio output
                        audioManager.writeOutputData(outputBuffer, CHUNK_SIZE);
                    }
                }
            }
            
            // No sleep - audio I/O should be as responsive as possible
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in processing loop: " << e.what() << std::endl;
    }

    // Cleanup
    audioManager.stop();
    audioManager.cleanup();

    std::cout << "Real-time audio processing stopped." << std::endl;
    return 0;
} 
