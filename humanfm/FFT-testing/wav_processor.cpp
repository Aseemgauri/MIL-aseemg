#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <complex>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <chrono>
#include <fftw3.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "ringbuffer.cpp"
#include "audio_config.h"


// WAV file header structure
struct WAVHeader {
    // RIFF chunk
    char riff_header[4];      // "RIFF"
    uint32_t wav_size;        // File size - 8
    char wave_header[4];      // "WAVE"
    
    // Format chunk
    char fmt_header[4];       // "fmt "
    uint32_t fmt_chunk_size;  // Format chunk size
    uint16_t audio_format;    // Audio format (1 for PCM)
    uint16_t num_channels;    // Number of channels
    uint32_t sample_rate;     // Sample rate (16000 for 16kHz)
    uint32_t byte_rate;       // Byte rate
    uint16_t block_align;     // Block align
    uint16_t bit_depth;       // Bits per sample
};

// Structure for chunk headers
struct ChunkHeader {
    char id[4];
    uint32_t size;
};

class WAVProcessor {
public:
    // Constructor: Initializes the ring buffer and FFTW plans
    WAVProcessor() : ringBuffer(CHUNK_SIZE * sizeof(float)) {
        // Initialize windows if not already initialized
        static bool windowsInitialized = false;
        if (!windowsInitialized) {
            initializeWindows();
            windowsInitialized = true;
        }
        initializeFFTW();
        
        // Initialize class-specific overlap-add state
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            classLookbackBufIdx[classIdx] = 0;
            // Zero out all buffers for this class
            std::fill(classOutputChunks[classIdx], classOutputChunks[classIdx] + CHUNK_SIZE, 0.0f);
            std::fill(classCurrentBuffer[classIdx], classCurrentBuffer[classIdx] + BUFFER_SIZE, 0.0);
            std::fill(classIfftResult[classIdx], classIfftResult[classIdx] + WINDOW_SIZE, 0.0);
            std::fill(classFftReal[classIdx], classFftReal[classIdx] + FFT_SIZE, 0.0);
            std::fill(classFftImag[classIdx], classFftImag[classIdx] + FFT_SIZE, 0.0);
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                std::fill(classIstftContextBuffers[classIdx][bufIdx], 
                         classIstftContextBuffers[classIdx][bufIdx] + ISTFT_OUTPUT_SIZE, 0.0);
            }
        }
    }

    // Destructor: Cleans up FFTW resources and stops ONNX thread
    ~WAVProcessor() {
        stopONNXThread();
        cleanupFFTW();
    }

    // Main processing function: Reads input WAV file, processes audio data, and writes to output file
    bool processWAV(const std::string& inputFile, const std::string& outputPrefix, const std::string& modelDirectory) {
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
            return false;
        }

        // Open 5 output files
        for (int i = 0; i < 5; i++) {
            std::string outputFile = outputPrefix + "_class" + std::to_string(i) + ".wav";
            outputFiles[i].open(outputFile, std::ios::binary);
            if (!outputFiles[i].is_open()) {
            std::cerr << "Error: Could not open output file: " << outputFile << std::endl;
                inFile.close();
                // Close any already opened files
                for (int j = 0; j < i; j++) {
                    outputFiles[j].close();
                }
                return false;
            }
        }

        // Initialize ONNX model
        if (!initializeONNX(modelDirectory)) {
            inFile.close();
            for (int i = 0; i < 5; i++) {
                outputFiles[i].close();
            }
            return false;
        }
        
        // Start ONNX inference thread
        startONNXThread();

        // Read and validate main header
        if (!readAndValidateHeader(inFile)) {
            inFile.close();
            for (int i = 0; i < 5; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Find and read data chunk
        uint32_t dataSize;
        if (!findDataChunk(inFile, dataSize)) {
            inFile.close();
            for (int i = 0; i < 5; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Write headers to all output files (mono 16-bit PCM)
        for (int i = 0; i < 5; i++) {
            writeHeader(outputFiles[i]);

        // Write data chunk header
        ChunkHeader dataChunk = {};
        std::memcpy(dataChunk.id, "data", 4);
            dataChunk.size = dataSize / 2;  // Half the size since we're converting stereo to mono
            outputFiles[i].write(reinterpret_cast<char*>(&dataChunk), sizeof(ChunkHeader));
        }

        // Process audio data and write immediately
        if (!processAndWriteAudioData(inFile, dataSize)) {
            inFile.close();
            for (int i = 0; i < 5; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Update data chunk size in all output files
        for (int i = 0; i < 5; i++) {
            size_t dataWritten = static_cast<size_t>(outputFiles[i].tellp()) - (sizeof(WAVHeader) + sizeof(ChunkHeader));
            outputFiles[i].seekp(sizeof(WAVHeader) + 4);  // Skip to size field of data chunk
        uint32_t finalSize = static_cast<uint32_t>(dataWritten);
            outputFiles[i].write(reinterpret_cast<char*>(&finalSize), sizeof(uint32_t));
            outputFiles[i].close();
        }

        inFile.close();
        return true;
    }

    // Getter methods for FFT results
    const double* getFFTReal() const { return g_fftReal; }
    const double* getFFTImag() const { return g_fftImag; }

private:
    // Initialize ONNX runtime and model
    bool initializeONNX(const std::string& modelDirectory) {
        try {
            // Initialize ONNX Runtime environment
            onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "WAVProcessor");
            
            // Create session options
            sessionOptions = std::make_unique<Ort::SessionOptions>();
            sessionOptions->SetIntraOpNumThreads(1);
            sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // Load the model (modelDirectory is actually the direct path to the .onnx file)
            std::string modelPath = modelDirectory;
            onnxSession = std::make_unique<Ort::Session>(*onnxEnv, modelPath.c_str(), *sessionOptions);
            
            // Create memory info
            memoryInfo = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            
            // Initialize input buffers
            embeddingInput.resize(5, 1.0f);  // 1x5 filled with 1s
            convBufInput.resize(4 * 2 * 129, 0.0f);  // 1x4x2x129
            deconvBufInput.resize(32 * 2 * 129, 0.0f);  // 1x32x2x129
            
            // Initialize block buffers
            for (int i = 0; i < 6; i++) {
                blockBufsInput[i][0].resize(129 * 32, 0.0f);  // h0: 1x129x32
                blockBufsInput[i][1].resize(129 * 32, 0.0f);  // c0: 1x129x32
            }
            
            std::cout << "ONNX model loaded successfully from: " << modelPath << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing ONNX: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Start ONNX inference thread
    void startONNXThread() {
        stopOnnxThread = false;
        onnxThread = std::thread(&WAVProcessor::onnxInferenceLoop, this);
    }
    
    // Stop ONNX inference thread
    void stopONNXThread() {
        stopOnnxThread = true;
        fftQueueCondition.notify_all();
        if (onnxThread.joinable()) {
            onnxThread.join();
        }
    }
    
    // ONNX inference loop running in separate thread
    void onnxInferenceLoop() {
        while (!stopOnnxThread) {
            std::unique_lock<std::mutex> lock(fftQueueMutex);
            fftQueueCondition.wait(lock, [this] { return !fftDataQueue.empty() || stopOnnxThread; });
            
            if (stopOnnxThread) break;
            
            std::vector<float> fftData = fftDataQueue.front();
            fftDataQueue.pop();
            lock.unlock();
            
            // Run ONNX inference
            runONNXInference(fftData);
        }
    }
    
    // Run ONNX inference on FFT data
    void runONNXInference(const std::vector<float>& fftData) {
        std::cout << "[ONNX] Starting inference..." << std::endl;
        try {
            // Prepare input tensors
            std::vector<const char*> inputNames = {
                "mixture_tf", "embedding", "conv_buf",
                "block_bufs::buf0::h0", "block_bufs::buf0::c0",
                "block_bufs::buf1::h0", "block_bufs::buf1::c0",
                "block_bufs::buf2::h0", "block_bufs::buf2::c0",
                "block_bufs::buf3::h0", "block_bufs::buf3::c0",
                "block_bufs::buf4::h0", "block_bufs::buf4::c0",
                "block_bufs::buf5::h0", "block_bufs::buf5::c0",
                "deconv_buf"
            };
            
            std::vector<const char*> outputNames = {
                "output", "out::conv_buf",
                "out::block_bufs::buf0::h0", "out::block_bufs::buf0::c0",
                "out::block_bufs::buf1::h0", "out::block_bufs::buf1::c0",
                "out::block_bufs::buf2::h0", "out::block_bufs::buf2::c0",
                "out::block_bufs::buf3::h0", "out::block_bufs::buf3::c0",
                "out::block_bufs::buf4::h0", "out::block_bufs::buf4::c0",
                "out::block_bufs::buf5::h0", "out::block_bufs::buf5::c0",
                "out::deconv_buf"
            };
            
            // Create input tensors
            std::vector<Ort::Value> inputTensors;
            
            // mixture_tf: 1x4x1x129
            std::vector<int64_t> mixtureShape = {1, 4, 1, 129};
            inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo, 
                const_cast<float*>(fftData.data()), fftData.size(), mixtureShape.data(), mixtureShape.size()));
            
            // embedding: 1x5
            std::vector<int64_t> embeddingShape = {1, 5};
            inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo,
                embeddingInput.data(), embeddingInput.size(), embeddingShape.data(), embeddingShape.size()));
            
            // conv_buf: 1x4x2x129
            std::vector<int64_t> convBufShape = {1, 4, 2, 129};
            inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo,
                convBufInput.data(), convBufInput.size(), convBufShape.data(), convBufShape.size()));
            
            // block_bufs (6 blocks, each with h0 and c0): 1x129x32
            std::vector<int64_t> blockBufShape = {1, 129, 32};
            for (int i = 0; i < 6; i++) {
                inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo,
                    blockBufsInput[i][0].data(), blockBufsInput[i][0].size(), blockBufShape.data(), blockBufShape.size()));
                inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo,
                    blockBufsInput[i][1].data(), blockBufsInput[i][1].size(), blockBufShape.data(), blockBufShape.size()));
            }
            
            // deconv_buf: 1x32x2x129
            std::vector<int64_t> deconvBufShape = {1, 32, 2, 129};
            inputTensors.push_back(Ort::Value::CreateTensor<float>(*memoryInfo,
                deconvBufInput.data(), deconvBufInput.size(), deconvBufShape.data(), deconvBufShape.size()));
            
            // Run inference
            auto outputTensors = onnxSession->Run(Ort::RunOptions{nullptr}, inputNames.data(), 
                inputTensors.data(), inputTensors.size(), outputNames.data(), outputNames.size());
            
            // Extract main output (1x5x1x258)
            float* outputData = outputTensors[0].GetTensorMutableData<float>();
            
            // Debug: Print ONNX output statistics for each class
            for (int classIdx = 0; classIdx < 5; classIdx++) {
                float* classData = outputData + classIdx * 258;
                float minVal = classData[0], maxVal = classData[0], sum = 0.0f;
                for (int i = 0; i < 258; i++) {
                    minVal = std::min(minVal, classData[i]);
                    maxVal = std::max(maxVal, classData[i]);
                    sum += classData[i];
                }
                std::cout << "[ONNX] Class " << classIdx 
                          << " Range: [" << minVal << ", " << maxVal << "]"
                          << ", Mean: " << (sum / 258) << std::endl;
            }
            
            // Process each class output
            std::lock_guard<std::mutex> lock(outputMutex);
            for (int classIdx = 0; classIdx < 5; classIdx++) {
                std::vector<float> classOutput(258);
                std::copy(outputData + classIdx * 258, outputData + (classIdx + 1) * 258, classOutput.begin());
                classOutputs[classIdx].push_back(classOutput);
            }
            
            // Update state buffers for next iteration
            updateStateBuffers(outputTensors);
            
            std::cout << "[ONNX] Inference completed successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in ONNX inference: " << e.what() << std::endl;
        }
    }
    
    // Update state buffers from ONNX outputs
    void updateStateBuffers(std::vector<Ort::Value>& outputTensors) {
        // Copy output buffers back to input buffers for next iteration
        
        // out::conv_buf -> conv_buf
        float* convBufOut = outputTensors[1].GetTensorMutableData<float>();
        std::copy(convBufOut, convBufOut + convBufInput.size(), convBufInput.begin());
        
        // out::block_bufs -> block_bufs
        for (int i = 0; i < 6; i++) {
            float* h0Out = outputTensors[2 + i * 2].GetTensorMutableData<float>();
            float* c0Out = outputTensors[2 + i * 2 + 1].GetTensorMutableData<float>();
            
            std::copy(h0Out, h0Out + blockBufsInput[i][0].size(), blockBufsInput[i][0].begin());
            std::copy(c0Out, c0Out + blockBufsInput[i][1].size(), blockBufsInput[i][1].begin());
        }
        
        // out::deconv_buf -> deconv_buf
        float* deconvBufOut = outputTensors[14].GetTensorMutableData<float>();
        std::copy(deconvBufOut, deconvBufOut + deconvBufInput.size(), deconvBufInput.begin());
    }

    // Initializes FFTW plans and buffers for forward and inverse FFT
    void initializeFFTW() {
        // Initialize FFT
        fftIn = fftw_alloc_real(WINDOW_SIZE);
        fftOut = fftw_alloc_complex(FFT_SIZE);
        fftPlan = fftw_plan_dft_r2c_1d(WINDOW_SIZE, fftIn, fftOut, FFTW_ESTIMATE);
        
        // Initialize IFFT
        ifftIn = fftw_alloc_complex(FFT_SIZE);
        ifftOut = fftw_alloc_real(WINDOW_SIZE);
        ifftPlan = fftw_plan_dft_c2r_1d(WINDOW_SIZE, ifftIn, ifftOut, FFTW_ESTIMATE);
    }

    // Cleans up FFTW resources and frees allocated memory
    void cleanupFFTW() {
        fftw_destroy_plan(fftPlan);
        fftw_destroy_plan(ifftPlan);
        fftw_free(fftIn);
        fftw_free(fftOut);
        fftw_free(ifftIn);
        fftw_free(ifftOut);
    }

    // Reads and validates the WAV file header, checking format and sample rate
    bool readAndValidateHeader(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
        
        if (std::string(header.riff_header, 4) != "RIFF") {
            std::cerr << "Error: Invalid RIFF header. Expected 'RIFF', got '" 
                     << std::string(header.riff_header, 4) << "'" << std::endl;
            return false;
        }
        if (std::string(header.wave_header, 4) != "WAVE") {
            std::cerr << "Error: Invalid WAVE header. Expected 'WAVE', got '" 
                     << std::string(header.wave_header, 4) << "'" << std::endl;
            return false;
        }
        if (std::string(header.fmt_header, 4) != "fmt ") {
            std::cerr << "Error: Invalid format chunk header. Expected 'fmt ', got '" 
                     << std::string(header.fmt_header, 4) << "'" << std::endl;
            return false;
        }

        if (header.sample_rate != 16000) {
            std::cerr << "Error: Input file must be 16kHz" << std::endl;
            return false;
        }

        // Check for 16-bit PCM format
        if (header.audio_format != 1) {  // 1 is the format code for PCM
            std::cerr << "Error: Input file must be 16-bit PCM format (audio_format = 1)" << std::endl;
            return false;
        }
        if (header.bit_depth != 16) {
            std::cerr << "Error: Input file must be 16-bit PCM format (bits_per_sample = 16)" << std::endl;
            return false;
        }
        if (header.num_channels != 2) {
            std::cerr << "Error: Input file must be stereo (2 channels)" << std::endl;
            return false;
        }

        // Print WAV header information
        std::cout << "WAV File Information:" << std::endl;
        std::cout << "Sample Rate: " << header.sample_rate << " Hz" << std::endl;
        std::cout << "Bit Depth: " << header.bit_depth << " bits" << std::endl;
        std::cout << "Audio Format: " << (header.audio_format == 1 ? "PCM" : "Unknown") << std::endl;
        std::cout << "Number of Channels: " << header.num_channels << std::endl;
        std::cout << "Byte Rate: " << header.byte_rate << " bytes/sec" << std::endl;
        std::cout << "Block Align: " << header.block_align << " bytes" << std::endl;

        return true;
    }

    // Locates the data chunk in the WAV file and returns its size
    bool findDataChunk(std::ifstream& file, uint32_t& dataSize) {
        ChunkHeader chunkHeader;
        bool foundData = false;

        while (!foundData && file.good()) {
            file.read(reinterpret_cast<char*>(&chunkHeader), sizeof(ChunkHeader));
            
            if (std::string(chunkHeader.id, 4) == "data") {
                dataSize = chunkHeader.size;
                foundData = true;
            }
            else if (std::string(chunkHeader.id, 4) == "fact") {
                file.seekg(chunkHeader.size, std::ios::cur);
            }
            else {
                file.seekg(chunkHeader.size, std::ios::cur);
            }
        }

        if (!foundData) {
            std::cerr << "Error: Could not find data chunk in WAV file" << std::endl;
            return false;
        }

        return true;
    }

    // Processes audio data in chunks and writes to output files
    bool processAndWriteAudioData(std::ifstream& inFile, uint32_t dataSize) {
        size_t totalBytesRead = 0;
        
        while (totalBytesRead < dataSize) {
            if (!readAndProcessChunk(inFile, dataSize, totalBytesRead)) {
                return false;
            }
            
            // Convert float output back to int16 and write to respective output files
            for (int classIdx = 0; classIdx < 5; classIdx++) {
                int16_t outputBuffer[CHUNK_SIZE];
                
                for (int i = 0; i < CHUNK_SIZE; i++) {
                    // Clamp and convert to int16
                    float sample = classOutputChunks[classIdx][i];
                    sample = std::max(-1.0f, std::min(1.0f, sample));
                    outputBuffer[i] = static_cast<int16_t>(sample * 32767.0f);
                }
                
                // Debug: Print final int16 output statistics
                int16_t int16Min = outputBuffer[0], int16Max = outputBuffer[0];
                long long int16Sum = 0;
                int nonZeroCount = 0;
                for (int i = 0; i < CHUNK_SIZE; i++) {
                    int16Min = std::min(int16Min, outputBuffer[i]);
                    int16Max = std::max(int16Max, outputBuffer[i]);
                    int16Sum += outputBuffer[i];
                    if (outputBuffer[i] != 0) nonZeroCount++;
                }
                std::cout << "[OUTPUT] Class " << classIdx 
                          << " Range: [" << int16Min << ", " << int16Max << "]"
                          << ", Mean: " << (int16Sum / CHUNK_SIZE)
                          << ", NonZero: " << nonZeroCount << "/" << CHUNK_SIZE << std::endl;
                
                // Write class-specific data to its output file
                outputFiles[classIdx].write(reinterpret_cast<char*>(outputBuffer), CHUNK_SIZE * sizeof(int16_t));
            }
        }
        
        return true;
    }

    // Reads a chunk of audio data and processes it
    bool readAndProcessChunk(std::ifstream& file, uint32_t dataSize, size_t& totalBytesRead) {
        // Read stereo int16 samples
        size_t bytesToRead = std::min(static_cast<size_t>(CHUNK_SIZE * 2 * sizeof(int16_t)), 
                                      static_cast<size_t>(dataSize - totalBytesRead));
        
        int16_t stereoBuffer[CHUNK_SIZE * 2];  // Stereo samples
        file.read(reinterpret_cast<char*>(stereoBuffer), bytesToRead);
        
        int numStereoSamples = bytesToRead / (2 * sizeof(int16_t));
        
        // Convert stereo int16 to mono float and store in g_chunkBuffer
        float* floatChunk = reinterpret_cast<float*>(g_chunkBuffer);
        for (int i = 0; i < numStereoSamples; i++) {
            // Mix stereo to mono and convert to float
            float left = static_cast<float>(stereoBuffer[i * 2]) / 32768.0f;
            float right = static_cast<float>(stereoBuffer[i * 2 + 1]) / 32768.0f;
            floatChunk[i] = (left + right) * 0.5f;  // Average the channels
        }
        
        // Debug: Print input sample statistics
        if (numStereoSamples > 0) {
            float minVal = floatChunk[0], maxVal = floatChunk[0], sum = 0.0f;
            for (int i = 0; i < numStereoSamples; i++) {
                minVal = std::min(minVal, floatChunk[i]);
                maxVal = std::max(maxVal, floatChunk[i]);
                sum += floatChunk[i];
            }
            std::cout << "[INPUT] Samples: " << numStereoSamples 
                      << ", Range: [" << minVal << ", " << maxVal << "]"
                      << ", Mean: " << (sum / numStereoSamples) << std::endl;
        }
        
        // Pad the last chunk with zeros if necessary
        for (int i = numStereoSamples; i < CHUNK_SIZE; i++) {
            floatChunk[i] = 0.0f;
        }
        
        // Process the chunk
        if (!processChunk()) {
            return false;
        }
        
        totalBytesRead += bytesToRead;
        return true;
    }

    // Processes a single chunk of audio data through FFT and ONNX inference
    bool processChunk() {
        // Add the chunk to the ring buffer
        ringBuffer.write(g_chunkBuffer);
        
        // Get the window of samples for FFT analysis
        getWindowSamples();
        
        // Apply analysis window and copy to FFT input
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            float sample = *reinterpret_cast<const float*>(&g_windowBuffer[i * sizeof(float)]);
            fftIn[i] = static_cast<double>(sample) * g_analysisWindow[i];
        }
        
        // Perform FFT
        fftw_execute(fftPlan);
        
        // Store real and imaginary components
        for (size_t i = 0; i < FFT_SIZE; i++) {
            g_fftReal[i] = fftOut[i][0];  // Real part
            g_fftImag[i] = fftOut[i][1];  // Imaginary part
        }

        // Debug: Print FFT output statistics
        double fftRealMin = g_fftReal[0], fftRealMax = g_fftReal[0];
        double fftImagMin = g_fftImag[0], fftImagMax = g_fftImag[0];
        for (size_t i = 0; i < FFT_SIZE; i++) {
            fftRealMin = std::min(fftRealMin, g_fftReal[i]);
            fftRealMax = std::max(fftRealMax, g_fftReal[i]);
            fftImagMin = std::min(fftImagMin, g_fftImag[i]);
            fftImagMax = std::max(fftImagMax, g_fftImag[i]);
        }
        std::cout << "[FFT] Real: [" << fftRealMin << ", " << fftRealMax << "]"
                  << ", Imag: [" << fftImagMin << ", " << fftImagMax << "]" << std::endl;

        // Prepare FFT data for ONNX (format: Real{channel1} Real{channel2} Imaginary{Channel1} Imaginary{Channel2})
        // Since we have mono input, we duplicate for 4 channels as required
        std::vector<float> fftData(4 * 129);  // 4 channels * 129 frequency bins
        for (size_t i = 0; i < FFT_SIZE; i++) {
            // Real{channel1}, Real{channel2}, Imaginary{Channel1}, Imaginary{Channel2}
            fftData[i * 4 + 0] = static_cast<float>(g_fftReal[i]);  // Ch1 real
            fftData[i * 4 + 1] = static_cast<float>(g_fftReal[i]);  // Ch2 real (duplicate)
            fftData[i * 4 + 2] = static_cast<float>(g_fftImag[i]);  // Ch1 imag
            fftData[i * 4 + 3] = static_cast<float>(g_fftImag[i]);  // Ch2 imag (duplicate)
        }
        
        // Send FFT data to ONNX inference queue
        {
            std::lock_guard<std::mutex> lock(fftQueueMutex);
            fftDataQueue.push(fftData);
        }
        fftQueueCondition.notify_one();

        // Wait 6ms to simulate real-time audio feeding and give ONNX model time to process
        std::cout << "[TIMING] Waiting 6ms for ONNX processing..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(6));
        std::cout << "[TIMING] Resuming after 6ms delay" << std::endl;

        // Process available ONNX outputs for each class
        return processClassOutputs();
    }

    // Process ONNX outputs for each class and perform IFFT + overlap-add
    bool processClassOutputs() {
        std::lock_guard<std::mutex> lock(outputMutex);
        
        // Check if we have outputs available for all classes
        bool hasOutputs = true;
        std::cout << "[QUEUE] Output queue sizes: ";
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            std::cout << "C" << classIdx << ":" << classOutputs[classIdx].size() << " ";
            if (classOutputs[classIdx].empty()) {
                hasOutputs = false;
            }
        }
        std::cout << std::endl;
        
        if (!hasOutputs) {
            // No outputs available yet, fill with zeros
            std::cout << "[DEBUG] No ONNX outputs available yet, filling with zeros" << std::endl;
            for (int classIdx = 0; classIdx < 5; classIdx++) {
                std::fill(classOutputChunks[classIdx], classOutputChunks[classIdx] + CHUNK_SIZE, 0.0f);
            }
            return true;
        }
        
        std::cout << "[DEBUG] Processing ONNX outputs for all 5 classes" << std::endl;
        
        // Process each class
        for (int classIdx = 0; classIdx < 5; classIdx++) {
            std::vector<float> classOutput = classOutputs[classIdx].front();
            classOutputs[classIdx].erase(classOutputs[classIdx].begin());
            
            // Split real and imaginary parts into class-specific buffers (first 129 real, next 129 imaginary)
            for (size_t i = 0; i < FFT_SIZE; i++) {
                classFftReal[classIdx][i] = static_cast<double>(classOutput[i]);           // Real part
                classFftImag[classIdx][i] = static_cast<double>(classOutput[i + 129]);     // Imaginary part
            }
            
            // Perform IFFT for this class
            if (!performIFFTForClass(classIdx)) {
            return false;
        }

            // Perform overlap-add synthesis for this class
            if (!performOverlapAddForClass(classIdx)) {
            return false;
            }
        }

        return true;
    }

    // Performs inverse FFT on the processed frequency domain data
    bool performIFFTForClass(int classIdx) {
        // Prepare for IFFT using class-specific FFT data
        for (size_t i = 0; i < FFT_SIZE; i++) {
            ifftIn[i][0] = classFftReal[classIdx][i];  // Real part
            ifftIn[i][1] = classFftImag[classIdx][i];  // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan);
        
        // Store IFFT result in class-specific buffer (normalize by window size)
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            classIfftResult[classIdx][i] = ifftOut[i] / WINDOW_SIZE;
        }
        
        // Debug: Print IFFT result statistics
        double ifftMin = classIfftResult[classIdx][0], ifftMax = classIfftResult[classIdx][0];
        double ifftSum = 0.0;
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            ifftMin = std::min(ifftMin, classIfftResult[classIdx][i]);
            ifftMax = std::max(ifftMax, classIfftResult[classIdx][i]);
            ifftSum += classIfftResult[classIdx][i];
        }
        std::cout << "[IFFT] Class " << classIdx 
                  << " Range: [" << ifftMin << ", " << ifftMax << "]"
                  << ", Mean: " << (ifftSum / WINDOW_SIZE) << std::endl;

        return true;
    }

    // Performs overlap-add synthesis on the IFFT result for a specific class
    bool performOverlapAddForClass(int classIdx) {
        // Store the current IFFT result in the class-specific current buffer
        std::copy(classIfftResult[classIdx], classIfftResult[classIdx] + BUFFER_SIZE, classCurrentBuffer[classIdx]);

        // Copy the last ISTFT_OUTPUT_SIZE frames to the class-specific context buffer
        double* ctxPtr = classIstftContextBuffers[classIdx][classLookbackBufIdx[classIdx]];
        std::copy(classIfftResult[classIdx] + (WINDOW_SIZE - ISTFT_OUTPUT_SIZE),
                 classIfftResult[classIdx] + WINDOW_SIZE,
                 ctxPtr);

        // Do overlap-add for the current chunk using class-specific buffers
        for (int j = 0; j < CHUNK_SIZE; j++) {
            // Accumulate result from all relevant lookback buffers for this class
            double result = 0.0;
            for (int lookbackOffset = 0; lookbackOffset < ISTFT_LOOKBACK_BUFFERS; lookbackOffset++) {
                // Get the correct buffer index (going backwards in time)
                int bufIdx = (classLookbackBufIdx[classIdx] - lookbackOffset + ISTFT_LOOKBACK_BUFFERS) % ISTFT_LOOKBACK_BUFFERS;
                double* historyPtr = classIstftContextBuffers[classIdx][bufIdx];
                
                // Calculate the index in the history buffer
                int ctxIdx = j + lookbackOffset * CHUNK_SIZE;
                if (ctxIdx < ISTFT_OUTPUT_SIZE) {
                    // Apply synthesis window and accumulate
                    result += historyPtr[ctxIdx] * g_synthesisWindow[ctxIdx];
                }
            }
            
            // Store the result in class-specific buffer
            classOutputChunks[classIdx][j] = static_cast<float>(result);
        }

        // Advance the class-specific lookback buffer index
        classLookbackBufIdx[classIdx] = (classLookbackBufIdx[classIdx] + 1) % ISTFT_LOOKBACK_BUFFERS;
        
        // Debug: Print overlap-add output statistics
        float overlapMin = classOutputChunks[classIdx][0], overlapMax = classOutputChunks[classIdx][0];
        float overlapSum = 0.0f;
        for (int i = 0; i < CHUNK_SIZE; i++) {
            overlapMin = std::min(overlapMin, classOutputChunks[classIdx][i]);
            overlapMax = std::max(overlapMax, classOutputChunks[classIdx][i]);
            overlapSum += classOutputChunks[classIdx][i];
        }
        std::cout << "[OVERLAP] Class " << classIdx 
                  << " Range: [" << overlapMin << ", " << overlapMax << "]"
                  << ", Mean: " << (overlapSum / CHUNK_SIZE) << std::endl;

        return true;
    }

    // Retrieves window samples from the ring buffer for processing
    void getWindowSamples() {
        const std::vector<char>& buffer = ringBuffer.getBuffer();
        size_t writePos = ringBuffer.getWritePosition();
        
        for (size_t i = 0; i < WINDOW_SIZE * sizeof(float); i++) {
            size_t pos = (writePos - WINDOW_SIZE * sizeof(float) + i + buffer.size()) % buffer.size();
            g_windowBuffer[i] = buffer[pos];
        }
    }

    // Writes the WAV header to the output file (mono 16-bit PCM)
    void writeHeader(std::ofstream& file) {
        WAVHeader outputHeader = {};
        
        // Set RIFF chunk
        std::memcpy(outputHeader.riff_header, "RIFF", 4);
        outputHeader.wav_size = 0;  // Will be updated later
        std::memcpy(outputHeader.wave_header, "WAVE", 4);
        
        // Set format chunk
        std::memcpy(outputHeader.fmt_header, "fmt ", 4);
        outputHeader.fmt_chunk_size = 16;
        outputHeader.audio_format = 1;  // PCM
        outputHeader.num_channels = 1;  // Mono
        outputHeader.sample_rate = 16000;
        outputHeader.bit_depth = 16;
        outputHeader.byte_rate = outputHeader.sample_rate * outputHeader.num_channels * outputHeader.bit_depth / 8;
        outputHeader.block_align = outputHeader.num_channels * outputHeader.bit_depth / 8;
        
        file.write(reinterpret_cast<char*>(&outputHeader), sizeof(WAVHeader));
    }

    WAVHeader header;
    RingBuffer ringBuffer;
    
    // Output files for 5 classes
    std::ofstream outputFiles[5];
    
    // FFTW variables
    double* fftIn;
    fftw_complex* fftOut;
    fftw_plan fftPlan;
    
    // IFFTW variables
    fftw_complex* ifftIn;
    double* ifftOut;
    fftw_plan ifftPlan;
    
    // ONNX-related variables
    std::unique_ptr<Ort::Env> onnxEnv;
    std::unique_ptr<Ort::Session> onnxSession;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;
    
    // Threading for ONNX inference
    std::thread onnxThread;
    std::atomic<bool> stopOnnxThread{false};
    std::mutex fftQueueMutex;
    std::condition_variable fftQueueCondition;
    std::queue<std::vector<float>> fftDataQueue;
    
    // ONNX output storage for each class
    std::vector<std::vector<float>> classOutputs[5];
    std::mutex outputMutex;
    
    // ONNX input/output buffers
    std::vector<float> embeddingInput;
    std::vector<float> convBufInput;
    std::vector<float> blockBufsInput[6][2];  // 6 blocks, 2 buffers each (h0, c0)
    std::vector<float> deconvBufInput;
    
    // Class-specific output buffers for overlap-add
    float classOutputChunks[5][CHUNK_SIZE];
    
    // Class-specific overlap-add state buffers
    int8_t classLookbackBufIdx[5];
    double classIstftContextBuffers[5][ISTFT_LOOKBACK_BUFFERS][ISTFT_OUTPUT_SIZE];
    double classCurrentBuffer[5][BUFFER_SIZE];
    double classIfftResult[5][WINDOW_SIZE];
    
    // Class-specific FFTW buffers to avoid interference
    double classFftReal[5][FFT_SIZE];
    double classFftImag[5][FFT_SIZE];
};

// Main entry point: Processes command line arguments and runs the WAV processor
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <input_wav_file> <output_prefix> <model_file.onnx>" << std::endl;
        std::cout << "Will create 5 output files: <output_prefix>_class0.wav through <output_prefix>_class4.wav" << std::endl;
        return 1;
    }

    WAVProcessor processor;
    
    if (!processor.processWAV(argv[1], argv[2], argv[3])) {
        return 1;
    }

    std::cout << "Successfully processed WAV file" << std::endl;
    return 0;
} 