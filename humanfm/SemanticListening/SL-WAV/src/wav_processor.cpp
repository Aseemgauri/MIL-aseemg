#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <complex>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fftw3.h>
#include "ringbuffer.h"
#include "audio_config.h"
#include "SL_Model.h"

// Number of output classes (only hardcoded value allowed)
#define NUM_CLASSES 5

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

// Model dimensions discovered at initialization
struct ModelDimensions {
    int fft_data_size;      // channels * freq_bins (e.g., 4 * 129 = 516)
    int output_data_size;   // classes * output_freq_bins (e.g., NUM_CLASSES * 258 = 1290)
    int class_output_size;  // output_freq_bins (e.g., 258)
    int queue_size;         // Based on processing requirements (e.g., 8)
    
    ModelDimensions() : fft_data_size(0), output_data_size(0), class_output_size(0), queue_size(8) {}
};

// Global model dimensions (discovered once, used for all static allocations)
static ModelDimensions g_modelDims;



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
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            classLookbackBufIdx[classIdx] = 0;
            // Zero out all buffers for this class
            memset(classOutputChunks[classIdx], 0, CHUNK_SIZE * sizeof(float));
            memset(classCurrentBuffer[classIdx], 0, BUFFER_SIZE * sizeof(double));
            memset(classIfftResult[classIdx], 0, WINDOW_SIZE * sizeof(double));
            memset(classFftReal[classIdx], 0, FFT_SIZE * sizeof(double));
            memset(classFftImag[classIdx], 0, FFT_SIZE * sizeof(double));
            for (int bufIdx = 0; bufIdx < ISTFT_LOOKBACK_BUFFERS; bufIdx++) {
                memset(classIstftContextBuffers[classIdx][bufIdx], 0, ISTFT_OUTPUT_SIZE * sizeof(double));
            }
        }
        
        // Initialize SL Model to null state
        slModelInitialized = false;
        
        // Buffers will be initialized after model dimensions are discovered
        buffersInitialized = false;
    }

    // Destructor: Cleans up FFTW resources
    ~WAVProcessor() {
        cleanupFFTW();
        // No need to delete slModel since it's now static
    }

    // Main processing function: Reads input WAV file, processes audio data, and writes to output file
    bool processWAV(const std::string& inputFile, const std::string& outputDirectory, const std::string& modelDirectory) {
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
            return false;
        }

        // Initialize SL Model first to discover dimensions
        if (!initializeSLModel(modelDirectory)) {
            inFile.close();
            return false;
        }
        
        // Now initialize buffers based on discovered model dimensions
        if (!initializeBuffers()) {
            inFile.close();
            return false;
        }

        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDirectory)) {
            inFile.close();
            return false;
        }

        // Extract base filename from input path for output files
        std::string baseFilename = extractBaseFilename(inputFile);

        // Open NUM_CLASSES output files in the specified directory
        for (int i = 0; i < NUM_CLASSES; i++) {
            std::string outputFile = outputDirectory + "/" + baseFilename + "_class" + std::to_string(i) + ".wav";
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

        // SL Model is now ready for synchronous inference

        // Read and validate main header
        if (!readAndValidateHeader(inFile)) {
            inFile.close();
            for (int i = 0; i < NUM_CLASSES; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Find and read data chunk
        uint32_t dataSize;
        if (!findDataChunk(inFile, dataSize)) {
            inFile.close();
            for (int i = 0; i < NUM_CLASSES; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Write headers to all output files (mono 32-bit float)
        for (int i = 0; i < NUM_CLASSES; i++) {
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
            for (int i = 0; i < NUM_CLASSES; i++) {
                outputFiles[i].close();
            }
            return false;
        }

        // Update data chunk size in all output files
        for (int i = 0; i < NUM_CLASSES; i++) {
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
    // Helper function to create output directory
    bool createOutputDirectory(const std::string& directory) {
        // Try to create directory using system command
        std::string command = "mkdir -p \"" + directory + "\"";
        int result = system(command.c_str());
        if (result != 0) {
            std::cerr << "Error: Could not create output directory: " << directory << std::endl;
            return false;
        }
        std::cout << "Output directory: " << directory << std::endl;
        return true;
    }
    
    // Helper function to extract base filename from path
    std::string extractBaseFilename(const std::string& filepath) {
        // Find the last slash or backslash
        size_t lastSlash = filepath.find_last_of("/\\");
        std::string filename = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);
        
        // Remove the .wav extension if present
        size_t lastDot = filename.find_last_of('.');
        if (lastDot != std::string::npos && filename.substr(lastDot) == ".wav") {
            filename = filename.substr(0, lastDot);
        }
        
        return filename;
    }
    
    // Initialize SL Model and discover dimensions
    bool initializeSLModel(const std::string& modelDirectory) {
        if (slModelInitialized) return true;
        
        try {
            // Initialize the static model (modelDirectory is actually the direct path to the .onnx file)
            std::string modelPath = modelDirectory;
            new (&slModel) SL_Model(modelPath);  // Placement new on static storage
            slModelInitialized = true;
            
            // Discover model dimensions
            g_modelDims.fft_data_size = slModel.getInputChannels() * slModel.getFFTSize();
            g_modelDims.class_output_size = slModel.getFFTSize() * 2;  // Complex output (real + imag)
            g_modelDims.output_data_size = slModel.getNumClasses() * g_modelDims.class_output_size;
            
            std::cout << "SL Model loaded successfully from: " << modelPath << std::endl;
            std::cout << "Discovered dimensions:" << std::endl;
            std::cout << "  FFT data size: " << g_modelDims.fft_data_size << std::endl;
            std::cout << "  Class output size: " << g_modelDims.class_output_size << std::endl;
            std::cout << "  Total output size: " << g_modelDims.output_data_size << std::endl;
            std::cout << "  Queue size: " << g_modelDims.queue_size << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing SL Model: " << e.what() << std::endl;
            slModelInitialized = false;
            return false;
        }
    }
    
    // Initialize buffers based on discovered model dimensions
    bool initializeBuffers() {
        if (buffersInitialized) return true;
        
        // Calculate total memory needed (no queues needed for blocking inference)
        int fftDataSize = g_modelDims.fft_data_size;
        int outputDataSize = g_modelDims.output_data_size;
        int classOutputSize = g_modelDims.class_output_size;
        
        int totalMemoryNeeded = fftDataSize + outputDataSize + classOutputSize;
        
        if (totalMemoryNeeded > MAX_MEMORY_POOL_SIZE) {
            std::cerr << "Error: Model requires " << totalMemoryNeeded 
                     << " floats but memory pool is only " << MAX_MEMORY_POOL_SIZE << std::endl;
            return false;
        }
        
        std::cout << "Memory allocation:" << std::endl;
        std::cout << "  Total needed: " << totalMemoryNeeded << " floats" << std::endl;
        std::cout << "  Pool size: " << MAX_MEMORY_POOL_SIZE << " floats" << std::endl;
        
        // Initialize memory pool
        memset(memoryPool, 0, MAX_MEMORY_POOL_SIZE * sizeof(float));
        
        // Partition memory pool
        float* currentPtr = memoryPool;
        
        fftDataBuffer = currentPtr;
        currentPtr += fftDataSize;
        
        outputDataBuffer = currentPtr;
        currentPtr += outputDataSize;
        
        tempClassOutput = currentPtr;
        currentPtr += classOutputSize;
        
        buffersInitialized = true;
        return true;
    }
    

    
    // Run SL Model inference on FFT data (blocking/synchronous)
    bool runSLModelInference(const float* fftData) {
        std::cout << "[SL Model] Starting blocking inference..." << std::endl;
        try {
            // Use static output buffer with exact size
            memset(outputDataBuffer, 0, g_modelDims.output_data_size * sizeof(float));
            
            // Use SLModel's processFrame method
            if (!slModel.processFrame(const_cast<float*>(fftData), outputDataBuffer)) {
                std::cerr << "Error in SL Model processFrame" << std::endl;
                return false;
            }
            
            // Debug: Print SL Model output statistics for each class (processing stereo input)
            for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
                float* classData = outputDataBuffer + classIdx * g_modelDims.class_output_size;
                float minVal = classData[0], maxVal = classData[0], sum = 0.0f;
                for (int i = 0; i < g_modelDims.class_output_size; i++) {
                    minVal = std::min(minVal, classData[i]);
                    maxVal = std::max(maxVal, classData[i]);
                    sum += classData[i];
                }
                std::cout << "[SL Model] Class " << classIdx 
                          << " Range: [" << minVal << ", " << maxVal << "]"
                          << ", Mean: " << (sum / g_modelDims.class_output_size) << std::endl;
            }
            
            std::cout << "[SL Model] Inference completed successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in SL Model inference: " << e.what() << std::endl;
            return false;
        }
    }

    // Initializes FFTW plans and buffers for forward and inverse FFT
    void initializeFFTW() {
        // Initialize FFT using static buffers
        fftPlan = fftw_plan_dft_r2c_1d(WINDOW_SIZE, fftInBuffer, fftOutBuffer, FFTW_ESTIMATE);
        
        // Initialize IFFT using static buffers
        ifftPlan = fftw_plan_dft_c2r_1d(WINDOW_SIZE, ifftInBuffer, ifftOutBuffer, FFTW_ESTIMATE);
    }

    // Cleans up FFTW resources
    void cleanupFFTW() {
        fftw_destroy_plan(fftPlan);
        fftw_destroy_plan(ifftPlan);
        // No need to free static buffers
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

        // Check for 32-bit float format
        if (header.audio_format != 3) {  // 3 is the format code for IEEE float
            std::cerr << "Error: Input file must be 32-bit float format (audio_format = 3)" << std::endl;
            return false;
        }
        if (header.bit_depth != 32) {
            std::cerr << "Error: Input file must be 32-bit float format (bits_per_sample = 32)" << std::endl;
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
        std::cout << "Audio Format: " << (header.audio_format == 3 ? "IEEE Float" : "Unknown") << std::endl;
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
            
            // Write float32 output directly to respective output files
            for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
                // Debug: Print final float32 output statistics
                float floatMin = classOutputChunks[classIdx][0], floatMax = classOutputChunks[classIdx][0];
                float floatSum = 0.0f;
                int nonZeroCount = 0;
                for (int i = 0; i < CHUNK_SIZE; i++) {
                    floatMin = std::min(floatMin, classOutputChunks[classIdx][i]);
                    floatMax = std::max(floatMax, classOutputChunks[classIdx][i]);
                    floatSum += classOutputChunks[classIdx][i];
                    if (classOutputChunks[classIdx][i] != 0.0f) nonZeroCount++;
                }
                std::cout << "[OUTPUT] Class " << classIdx 
                          << " Range: [" << floatMin << ", " << floatMax << "]"
                          << ", Mean: " << (floatSum / CHUNK_SIZE)
                          << ", NonZero: " << nonZeroCount << "/" << CHUNK_SIZE << std::endl;
                
                // Write class-specific float32 data to its output file
                outputFiles[classIdx].write(reinterpret_cast<char*>(classOutputChunks[classIdx]), CHUNK_SIZE * sizeof(float));
            }
        }
        
        return true;
    }

    // Reads a chunk of audio data and processes it
    bool readAndProcessChunk(std::ifstream& file, uint32_t dataSize, size_t& totalBytesRead) {
        // Read stereo float32 samples
        size_t bytesToRead = std::min(static_cast<size_t>(CHUNK_SIZE * 2 * sizeof(float)), 
                                      static_cast<size_t>(dataSize - totalBytesRead));
        
        if (bytesToRead == 0) {
            return true;  // No more data to read
        }
        
        // Read stereo data into static buffer
        file.read(reinterpret_cast<char*>(stereoBuffer), bytesToRead);
        size_t samplesRead = bytesToRead / sizeof(float);
        
        if (file.gcount() != static_cast<std::streamsize>(bytesToRead)) {
            std::cerr << "Error: Could not read expected amount of data" << std::endl;
            return false;
        }
        
        totalBytesRead += bytesToRead;
        
        // Separate stereo channels instead of converting to mono
        size_t stereoSamples = samplesRead / 2;
        for (size_t i = 0; i < stereoSamples; i++) {
            leftChunkBuffer[i] = stereoBuffer[i * 2];      // Left channel
            rightChunkBuffer[i] = stereoBuffer[i * 2 + 1]; // Right channel
        }
        
        // Fill remaining samples with zeros if needed
        for (size_t i = stereoSamples; i < CHUNK_SIZE; i++) {
            leftChunkBuffer[i] = 0.0f;
            rightChunkBuffer[i] = 0.0f;
        }
        
        // Debug: Print stereo chunk statistics
        float leftMin = leftChunkBuffer[0], leftMax = leftChunkBuffer[0], leftSum = 0.0f;
        float rightMin = rightChunkBuffer[0], rightMax = rightChunkBuffer[0], rightSum = 0.0f;
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            leftMin = std::min(leftMin, leftChunkBuffer[i]);
            leftMax = std::max(leftMax, leftChunkBuffer[i]);
            leftSum += leftChunkBuffer[i];
            rightMin = std::min(rightMin, rightChunkBuffer[i]);
            rightMax = std::max(rightMax, rightChunkBuffer[i]);
            rightSum += rightChunkBuffer[i];
        }
        std::cout << "[INPUT] Left: [" << leftMin << ", " << leftMax << "], Mean: " << (leftSum / CHUNK_SIZE)
                  << " | Right: [" << rightMin << ", " << rightMax << "], Mean: " << (rightSum / CHUNK_SIZE) << std::endl;
        
        return processChunk();
    }

    // Processes a single chunk of audio data through FFT and SL Model inference
    bool processChunk() {
        // Process left channel FFT
        // Apply analysis window to left channel
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            fftInBuffer[i] = static_cast<double>(leftChunkBuffer[i]) * g_analysisWindow[i];
        }
        // Zero-pad the rest of the window
        for (size_t i = CHUNK_SIZE; i < WINDOW_SIZE; i++) {
            fftInBuffer[i] = 0.0;
        }
        
        // Perform FFT on left channel
        fftw_execute(fftPlan);
        
        // Store left channel FFT results
        for (size_t i = 0; i < FFT_SIZE; i++) {
            leftFftReal[i] = fftOutBuffer[i][0];  // Real part
            leftFftImag[i] = fftOutBuffer[i][1];  // Imaginary part
        }
        
        // Process right channel FFT
        // Apply analysis window to right channel
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            fftInBuffer[i] = static_cast<double>(rightChunkBuffer[i]) * g_analysisWindow[i];
        }
        // Zero-pad the rest of the window
        for (size_t i = CHUNK_SIZE; i < WINDOW_SIZE; i++) {
            fftInBuffer[i] = 0.0;
        }
        
        // Perform FFT on right channel
        fftw_execute(fftPlan);
        
        // Store right channel FFT results
        for (size_t i = 0; i < FFT_SIZE; i++) {
            rightFftReal[i] = fftOutBuffer[i][0];  // Real part
            rightFftImag[i] = fftOutBuffer[i][1];  // Imaginary part
        }

        // Debug: Print stereo FFT output statistics
        double leftRealMin = leftFftReal[0], leftRealMax = leftFftReal[0];
        double leftImagMin = leftFftImag[0], leftImagMax = leftFftImag[0];
        double rightRealMin = rightFftReal[0], rightRealMax = rightFftReal[0];
        double rightImagMin = rightFftImag[0], rightImagMax = rightFftImag[0];
        for (size_t i = 0; i < FFT_SIZE; i++) {
            leftRealMin = std::min(leftRealMin, leftFftReal[i]);
            leftRealMax = std::max(leftRealMax, leftFftReal[i]);
            leftImagMin = std::min(leftImagMin, leftFftImag[i]);
            leftImagMax = std::max(leftImagMax, leftFftImag[i]);
            rightRealMin = std::min(rightRealMin, rightFftReal[i]);
            rightRealMax = std::max(rightRealMax, rightFftReal[i]);
            rightImagMin = std::min(rightImagMin, rightFftImag[i]);
            rightImagMax = std::max(rightImagMax, rightFftImag[i]);
        }
        std::cout << "[FFT] Left Real: [" << leftRealMin << ", " << leftRealMax << "]"
                  << ", Left Imag: [" << leftImagMin << ", " << leftImagMax << "]" << std::endl;
        std::cout << "[FFT] Right Real: [" << rightRealMin << ", " << rightRealMax << "]"
                  << ", Right Imag: [" << rightImagMin << ", " << rightImagMax << "]" << std::endl;

        // Prepare FFT data for SL Model using static buffer with exact size
        // Format: Left_Real, Left_Imag, Right_Real, Right_Imag (interleaved per frequency bin)
        for (size_t i = 0; i < FFT_SIZE; i++) {
            fftDataBuffer[i * 4 + 0] = static_cast<float>(leftFftReal[i]);   // Left real
            fftDataBuffer[i * 4 + 1] = static_cast<float>(leftFftImag[i]);   // Left imag
            fftDataBuffer[i * 4 + 2] = static_cast<float>(rightFftReal[i]);  // Right real
            fftDataBuffer[i * 4 + 3] = static_cast<float>(rightFftImag[i]);  // Right imag
        }
        
        // Run SL Model inference synchronously (blocking)
        if (!runSLModelInference(fftDataBuffer)) {
            std::cerr << "Error: SL Model inference failed" << std::endl;
            return false;
        }

        // Process SL Model outputs for each class immediately
        return processClassOutputs();
    }

    // Process SL Model outputs for each class and perform IFFT + overlap-add
    bool processClassOutputs() {
        std::cout << "[DEBUG] Processing SL Model outputs for all " << NUM_CLASSES << " classes" << std::endl;
        
        // Process each class output directly from outputDataBuffer
        for (int classIdx = 0; classIdx < NUM_CLASSES; classIdx++) {
            // Get class output directly from the output buffer
            float* classData = outputDataBuffer + classIdx * g_modelDims.class_output_size;
            
            // Split real and imaginary parts into class-specific buffers (first 129 real, next 129 imaginary)
            for (size_t i = 0; i < FFT_SIZE; i++) {
                classFftReal[classIdx][i] = static_cast<double>(classData[i]);           // Real part
                classFftImag[classIdx][i] = static_cast<double>(classData[i + 129]);     // Imaginary part
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
            ifftInBuffer[i][0] = classFftReal[classIdx][i];  // Real part
            ifftInBuffer[i][1] = classFftImag[classIdx][i];  // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan);
        
        // Store IFFT result in class-specific buffer (normalize by window size)
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            classIfftResult[classIdx][i] = ifftOutBuffer[i] / WINDOW_SIZE;
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
        memcpy(classCurrentBuffer[classIdx], classIfftResult[classIdx], BUFFER_SIZE * sizeof(double));

        // Copy the last ISTFT_OUTPUT_SIZE frames to the class-specific context buffer
        double* ctxPtr = classIstftContextBuffers[classIdx][classLookbackBufIdx[classIdx]];
        memcpy(ctxPtr, classIfftResult[classIdx] + (WINDOW_SIZE - ISTFT_OUTPUT_SIZE), ISTFT_OUTPUT_SIZE * sizeof(double));

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
        auto bufferWrapper = ringBuffer.getBuffer();
        const char* buffer = bufferWrapper.data;
        size_t bufferSize = bufferWrapper.size;
        size_t writePos = ringBuffer.getWritePosition();
        
        for (size_t i = 0; i < WINDOW_SIZE * sizeof(float); i++) {
            size_t pos = (writePos - WINDOW_SIZE * sizeof(float) + i + bufferSize) % bufferSize;
            g_windowBuffer[i] = buffer[pos];
        }
    }

    // Writes the WAV header to the output file (mono 32-bit float)
    void writeHeader(std::ofstream& file) {
        WAVHeader outputHeader = {};
        
        // Set RIFF chunk
        std::memcpy(outputHeader.riff_header, "RIFF", 4);
        outputHeader.wav_size = 0;  // Will be updated later
        std::memcpy(outputHeader.wave_header, "WAVE", 4);
        
        // Set format chunk
        std::memcpy(outputHeader.fmt_header, "fmt ", 4);
        outputHeader.fmt_chunk_size = 16;
        outputHeader.audio_format = 3;  // IEEE Float
        outputHeader.num_channels = 1;  // Mono
        outputHeader.sample_rate = 16000;
        outputHeader.bit_depth = 32;
        outputHeader.byte_rate = outputHeader.sample_rate * outputHeader.num_channels * outputHeader.bit_depth / 8;
        outputHeader.block_align = outputHeader.num_channels * outputHeader.bit_depth / 8;
        
        file.write(reinterpret_cast<char*>(&outputHeader), sizeof(WAVHeader));
    }

    WAVHeader header;
    RingBuffer ringBuffer;
    
    // Output files for 5 classes
    std::ofstream outputFiles[NUM_CLASSES];
    
    // FFTW variables (static buffers instead of dynamic allocation)
    double fftInBuffer[WINDOW_SIZE];
    fftw_complex fftOutBuffer[FFT_SIZE];
    fftw_plan fftPlan;
    
    // IFFTW variables (static buffers instead of dynamic allocation)
    fftw_complex ifftInBuffer[FFT_SIZE];
    double ifftOutBuffer[WINDOW_SIZE];
    fftw_plan ifftPlan;
    
    // SL Model-related variables (using static storage instead of pointer)
    alignas(SL_Model) char slModelStorage[sizeof(SL_Model)];
    SL_Model& slModel = reinterpret_cast<SL_Model&>(slModelStorage);
    bool slModelInitialized;
    
    // Class-specific output buffers for overlap-add
    float classOutputChunks[NUM_CLASSES][CHUNK_SIZE];
    
    // Class-specific overlap-add state buffers
    int8_t classLookbackBufIdx[NUM_CLASSES];
    double classIstftContextBuffers[NUM_CLASSES][ISTFT_LOOKBACK_BUFFERS][ISTFT_OUTPUT_SIZE];
    double classCurrentBuffer[NUM_CLASSES][BUFFER_SIZE];
    double classIfftResult[NUM_CLASSES][WINDOW_SIZE];
    
    // Class-specific FFTW buffers to avoid interference
    double classFftReal[NUM_CLASSES][FFT_SIZE];
    double classFftImag[NUM_CLASSES][FFT_SIZE];
    
    // Stereo FFT buffers
    double leftFftReal[FFT_SIZE];
    double leftFftImag[FFT_SIZE];
    double rightFftReal[FFT_SIZE];
    double rightFftImag[FFT_SIZE];
    
    // Static buffers sized exactly based on discovered model dimensions
    static const int MAX_MEMORY_POOL_SIZE = 65536;  // Large enough for any reasonable model
    float memoryPool[MAX_MEMORY_POOL_SIZE];
    
    // Pointers to partitioned memory (set after model discovery)
    float* fftDataBuffer;
    float* outputDataBuffer;
    float* tempClassOutput;
    
    float stereoBuffer[CHUNK_SIZE * 2];  // For reading stereo input
    float leftChunkBuffer[CHUNK_SIZE];   // Left channel buffer
    float rightChunkBuffer[CHUNK_SIZE];  // Right channel buffer
    
    bool buffersInitialized;
};

// Main entry point: Processes command line arguments and runs the WAV processor
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <input_wav_file> <output_directory> <model_file.onnx>" << std::endl;
        std::cout << "Input file must be stereo 32-bit float WAV at 16kHz" << std::endl;
        std::cout << "Will create " << NUM_CLASSES << " output files: <output_directory>/<basename>_class0.wav through <output_directory>/<basename>_class" << (NUM_CLASSES - 1) << ".wav" << std::endl;
        std::cout << "Output files will be mono 32-bit float WAV at 16kHz" << std::endl;
        std::cout << "Example: " << argv[0] << " input.wav ./output_dir model.onnx" << std::endl;
        std::cout << "  -> Creates: ./output_dir/input_class0.wav, ./output_dir/input_class1.wav, etc." << std::endl;
        return 1;
    }

    WAVProcessor processor;
    
    if (!processor.processWAV(argv[1], argv[2], argv[3])) {
        return 1;
    }

    std::cout << "Successfully processed WAV file" << std::endl;
    return 0;
} 