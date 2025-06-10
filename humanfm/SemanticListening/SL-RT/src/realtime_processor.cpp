#include <iostream>
#include <cstring>
#include <algorithm>
#include <fftw3.h>
#include "audio_manager.h"
#include "audio_config.h"
#include "static_buffers.h"
#include "unified_ringbuffer.h"
#include "audio_data_structures.h"
#include "SL_Model.h"

// Configuration macros
#define NUM_CLASSES 5  // Number of output classes from the model

// Passthrough mode: when enabled, bypasses SL_Model inference and directly passes input to output
// Set to 1 to enable passthrough, 0 to use normal SL_Model processing
#define ENABLE_PASSTHROUGH 0

// Inference mode: when ENABLE_PASSTHROUGH is 0 and ENABLE_INFERENCE is 0, 
// we do all FFT/IFFT processing but bypass SL_Model inference (clone FFT output NUM_CLASSES times)
// Set to 1 to enable SL_Model inference, 0 to bypass inference but keep FFT processing
#define ENABLE_INFERENCE 1

class RealtimeProcessor {
public:
    RealtimeProcessor() : 
        ringBuffer_() {
        
        // Initialize windows if not already initialized
        static bool windowsInitialized = false;
        if (!windowsInitialized) {
            initializeWindows();
            windowsInitialized = true;
        }
        initializeFFTW();
        
        // Initialize class-specific overlap-add state
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            classLookbackBufIdx_[classIdx] = 0;
            std::memset(classCurrentBuffer_[classIdx], 0, BUFFER_SIZE * sizeof(double));
            std::memset(classIfftResult_[classIdx], 0, WINDOW_SIZE * sizeof(double));
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                std::memset(classIstftContextBuffers_[classIdx][bufIdx], 0, 
                           ISTFT_OUTPUT_SIZE * sizeof(double));
            }
        }
        
        // Initialize weighted sum coefficients
        for (int i = 0; i < NUM_CLASSES; i++) {
#if ENABLE_INFERENCE
            classWeights_[i] = 1.0f; // Full weights when inference is enabled
#else
            classWeights_[i] = 1.0f / NUM_CLASSES; // Average weights when inference is disabled
#endif
        }
        
        // Pre-allocate all processing buffers
        std::memset(fftInputBuffer_, 0, sizeof(fftInputBuffer_));
        std::memset(slModelOutputBuffer_, 0, sizeof(slModelOutputBuffer_));
        std::memset(currentAudioFrame_, 0, sizeof(currentAudioFrame_));
        std::memset(outputAudioFrame_, 0, sizeof(outputAudioFrame_));
        std::memset(leftChannelFrame_, 0, sizeof(leftChannelFrame_));
        std::memset(rightChannelFrame_, 0, sizeof(rightChannelFrame_));
        std::memset(leftFftOut_, 0, sizeof(leftFftOut_));
        std::memset(rightFftOut_, 0, sizeof(rightFftOut_));
    }

    ~RealtimeProcessor() {
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
        
        // For passthrough mode, mix stereo to mono for output compatibility
        for (size_t i = 0; i < samplesToProcess; i++) {
            currentAudioFrame_[i] = (leftChannelFrame_[i] + rightChannelFrame_[i]) * 0.5f;
        }
        
#if ENABLE_PASSTHROUGH
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
    // Ring buffer for sliding window
    SlidingWindowBuffer ringBuffer_;
    
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
    int8_t classLookbackBufIdx_[NUM_CLASSES];
    double classIstftContextBuffers_[NUM_CLASSES][ISTFT_LOOKBACK_BUFFERS][ISTFT_OUTPUT_SIZE];
    double classCurrentBuffer_[NUM_CLASSES][BUFFER_SIZE];
    double classIfftResult_[NUM_CLASSES][WINDOW_SIZE];
     
    // Weighted sum coefficients (pre-allocated)
    float classWeights_[NUM_CLASSES];
    
    // Pre-allocated processing buffers (no dynamic allocation during processing)
    float fftInputBuffer_[4 * 129];  // 4 channels * 129 frequency bins for SL_Model input
    float slModelOutputBuffer_[NUM_CLASSES * 258];  // NUM_CLASSES classes * 258 frequency bins for SL_Model output
    float currentAudioFrame_[CHUNK_SIZE];  // Current input audio frame
    float outputAudioFrame_[CHUNK_SIZE];   // Current output audio frame
    
    // Stereo channel buffers
    float leftChannelFrame_[CHUNK_SIZE];   // Left channel buffer
    float rightChannelFrame_[CHUNK_SIZE];  // Right channel buffer
    
    // Stereo FFT buffers
    fftw_complex leftFftOut_[FFT_SIZE];
    fftw_complex rightFftOut_[FFT_SIZE];
    
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
    
    // Single-threaded processing pipeline
    void processAudioFrame() {
        // Process left channel FFT
        // Apply analysis window to left channel
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            fftIn_[i] = static_cast<double>(leftChannelFrame_[i]) * g_analysisWindow[i];
        }
        // Zero-pad the rest of the window
        for (size_t i = CHUNK_SIZE; i < WINDOW_SIZE; i++) {
            fftIn_[i] = 0.0;
        }
        
        // Perform FFT on left channel
        fftw_execute(fftPlan_);
        
        // Store left channel FFT results
        for (size_t i = 0; i < FFT_SIZE; i++) {
            leftFftOut_[i][0] = fftOut_[i][0];  // Real part
            leftFftOut_[i][1] = fftOut_[i][1];  // Imaginary part
        }
        
        // Process right channel FFT
        // Apply analysis window to right channel
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            fftIn_[i] = static_cast<double>(rightChannelFrame_[i]) * g_analysisWindow[i];
        }
        // Zero-pad the rest of the window
        for (size_t i = CHUNK_SIZE; i < WINDOW_SIZE; i++) {
            fftIn_[i] = 0.0;
        }
        
        // Perform FFT on right channel
        fftw_execute(fftPlan_);
        
        // Store right channel FFT results
        for (size_t i = 0; i < FFT_SIZE; i++) {
            rightFftOut_[i][0] = fftOut_[i][0];  // Real part
            rightFftOut_[i][1] = fftOut_[i][1];  // Imaginary part
        }
        
        // Prepare FFT data for SL_Model or bypass
#if ENABLE_INFERENCE
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
            // Format: Left_Real, Left_Imag, Right_Real, Right_Imag (interleaved per frequency bin)
            for (size_t i = 0; i < FFT_SIZE; i++) {
                fftInputBuffer_[i * 4 + 0] = static_cast<float>(leftFftOut_[i][0]);   // Left real
                fftInputBuffer_[i * 4 + 1] = static_cast<float>(leftFftOut_[i][1]);   // Left imag
                fftInputBuffer_[i * 4 + 2] = static_cast<float>(rightFftOut_[i][0]);  // Right real
                fftInputBuffer_[i * 4 + 3] = static_cast<float>(rightFftOut_[i][1]);  // Right imag
            }
            
            // Use SL_Model's processFrame method with pre-allocated buffers
            if (!slModel_->processFrame(fftInputBuffer_, slModelOutputBuffer_)) {
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
        // Create fake output by mixing stereo FFT data for all NUM_CLASSES classes
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            int outputOffset = classIdx * 258;  // 258 = 129 real + 129 imag
            
            // Mix left and right channels for fake output (copy real parts)
            for (size_t i = 0; i < FFT_SIZE; i++) {
                // Average left and right real parts
                slModelOutputBuffer_[outputOffset + i] = static_cast<float>(
                    (leftFftOut_[i][0] + rightFftOut_[i][0]) * 0.5);
            }
            
            // Mix left and right channels for fake output (copy imaginary parts)
            for (size_t i = 0; i < FFT_SIZE; i++) {
                // Average left and right imaginary parts
                slModelOutputBuffer_[outputOffset + 129 + i] = static_cast<float>(
                    (leftFftOut_[i][1] + rightFftOut_[i][1]) * 0.5);
            }
        }
    }
    
    void processAllClasses() {
        // Process each class through IFFT and overlap-add
        float classOutputChunks[NUM_CLASSES][CHUNK_SIZE];
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            // Get pointer to this class's output data
            const float* classOutput = &slModelOutputBuffer_[classIdx * 258];
            
            // Perform IFFT for this class
            performIFFTForClass(classIdx, classOutput);
            
            // Perform overlap-add synthesis for this class
            performOverlapAddForClass(classIdx, classOutputChunks[classIdx]);
        }
        
        // Compute weighted sum of all classes into output frame
        std::fill(outputAudioFrame_, outputAudioFrame_ + CHUNK_SIZE, 0.0f);
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            for (int i = 0; i < CHUNK_SIZE; i++) {
                outputAudioFrame_[i] += classOutputChunks[classIdx][i] * classWeights_[classIdx];
            }
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
    if (!processor.initializeSLModel(argv[1])) {
        std::cerr << "Failed to initialize SL Model" << std::endl;
        return 1;
    }
#else
    #if ENABLE_PASSTHROUGH
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