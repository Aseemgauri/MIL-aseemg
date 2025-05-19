#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <complex>
#include <algorithm>  // For std::max and std::min
#include <fftw3.h>
#include "ringbuffer.cpp"

// Define the constants
#define CHUNK_SIZE 1024  // Example chunk size, adjust as needed
#define LB_SIZE 1024     // Lookback size
#define LF_SIZE 1024     // Look forward size
#define WINDOW_SIZE (LB_SIZE + CHUNK_SIZE + LF_SIZE)

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
    WAVProcessor() : ringBuffer(CHUNK_SIZE * sizeof(float)) {  // Adjust buffer size for float
        initializeBuffers();
        initializeFFTW();
    }

    ~WAVProcessor() {
        cleanupFFTW();
    }

    bool readWAV(const std::string& inputFile) {
        std::ifstream file(inputFile, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
            return false;
        }

        // Read and validate main header
        if (!readAndValidateHeader(file)) {
            return false;
        }

        // Find and read data chunk
        uint32_t dataSize;
        if (!findDataChunk(file, dataSize)) {
            return false;
        }

        // Process audio data
        if (!processAudioData(file, dataSize)) {
            return false;
        }

        file.close();
        return true;
    }

    bool writeWAV(const std::string& outputFile) {
        std::ofstream file(outputFile, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open output file: " << outputFile << std::endl;
            return false;
        }

        // Debug output for data size
        std::cout << "Writing " << processedData.size() << " bytes to output file" << std::endl;
        std::cout << "Number of samples: " << processedData.size() / sizeof(float) << std::endl;

        // Write header and data
        writeHeader(file);
        writeDataChunk(file);
        writeAudioData(file);

        file.close();
        return true;
    }

    // Getter methods for FFT results
    const std::vector<double>& getFFTReal() const { return fftReal; }
    const std::vector<double>& getFFTImag() const { return fftImag; }

private:
    void initializeBuffers() {
        // Initialize ring buffer with zeros
        std::vector<char> zeroChunk(CHUNK_SIZE * sizeof(float), 0);
        for (int i = 0; i < 3; i++) {
            ringBuffer.write(zeroChunk.data());
        }

        // Initialize synthesis buffers
        previousBuffer.resize(WINDOW_SIZE - LB_SIZE, 0.0);  // CHUNK_SIZE + LF_SIZE
        currentBuffer.resize(WINDOW_SIZE - LB_SIZE, 0.0);   // CHUNK_SIZE + LF_SIZE
        synthesisBuffer.resize(WINDOW_SIZE, 0.0);
    }

    void initializeFFTW() {
        // Initialize FFT
        fftIn = fftw_alloc_real(WINDOW_SIZE);
        fftOut = fftw_alloc_complex(WINDOW_SIZE/2 + 1);
        fftPlan = fftw_plan_dft_r2c_1d(WINDOW_SIZE, fftIn, fftOut, FFTW_ESTIMATE);
        
        // Initialize IFFT
        ifftIn = fftw_alloc_complex(WINDOW_SIZE/2 + 1);
        ifftOut = fftw_alloc_real(WINDOW_SIZE);
        ifftPlan = fftw_plan_dft_c2r_1d(WINDOW_SIZE, ifftIn, ifftOut, FFTW_ESTIMATE);
        
        // Initialize FFT result arrays
        fftReal.resize(WINDOW_SIZE/2 + 1);
        fftImag.resize(WINDOW_SIZE/2 + 1);
        
        // Initialize IFFT result array
        ifftResult.resize(WINDOW_SIZE);
    }

    void cleanupFFTW() {
        fftw_destroy_plan(fftPlan);
        fftw_destroy_plan(ifftPlan);
        fftw_free(fftIn);
        fftw_free(fftOut);
        fftw_free(ifftIn);
        fftw_free(ifftOut);
    }

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

    bool processAudioData(std::ifstream& file, uint32_t dataSize) {
        std::vector<char> chunk(CHUNK_SIZE * sizeof(float));
        size_t totalBytesRead = 0;
        
        while (totalBytesRead < dataSize) {
            if (!readAndProcessChunk(file, chunk, dataSize, totalBytesRead)) {
                return false;
            }
        }
        return true;
    }

    bool readAndProcessChunk(std::ifstream& file, std::vector<char>& chunk, 
                           uint32_t dataSize, size_t& totalBytesRead) {
        // Read a chunk from the file
        size_t bytesToRead = (CHUNK_SIZE * sizeof(float) < (dataSize - totalBytesRead)) ? 
                            CHUNK_SIZE * sizeof(float) : 
                            (dataSize - totalBytesRead);
        file.read(chunk.data(), bytesToRead);
        
        // Pad the last chunk with zeros if necessary
        if (bytesToRead < CHUNK_SIZE * sizeof(float)) {
            std::fill(chunk.begin() + bytesToRead, chunk.end(), 0);
        }
        
        // Process the chunk
        if (!processChunk(chunk)) {
            return false;
        }
        
        totalBytesRead += bytesToRead;
        return true;
    }

    bool processChunk(const std::vector<char>& chunk) {
        // Add the chunk to the ring buffer
        ringBuffer.write(chunk.data());
        
        // Get the window of samples for FFT analysis
        std::vector<char> window = getWindowSamples();
        
        // Convert char buffer to float samples
        std::vector<float> floatWindow(WINDOW_SIZE);
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            // Interpret the bytes as float values
            floatWindow[i] = *reinterpret_cast<const float*>(&window[i * sizeof(float)]);
        }
        
        // Convert float samples to double for FFT
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            fftIn[i] = static_cast<double>(floatWindow[i]);
        }
        
        // Perform FFT
        fftw_execute(fftPlan);
        
        // Store real and imaginary components
        for (size_t i = 0; i < WINDOW_SIZE/2 + 1; i++) {
            fftReal[i] = fftOut[i][0];  // Real part
            fftImag[i] = fftOut[i][1];  // Imaginary part
        }

        // Perform IFFT
        if (!performIFFT()) {
            return false;
        }

        // Perform overlap-add synthesis
        if (!performOverlapAdd()) {
            return false;
        }

        return true;
    }

    bool performIFFT() {
        // Prepare for IFFT
        for (size_t i = 0; i < WINDOW_SIZE/2 + 1; i++) {
            ifftIn[i][0] = fftReal[i];  // Real part
            ifftIn[i][1] = fftImag[i];  // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan);
        
        // Store IFFT result (normalize by window size)
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            ifftResult[i] = ifftOut[i] / WINDOW_SIZE;
        }

        return true;
    }

    bool performOverlapAdd() {
        // Store the current IFFT result in the current buffer
        for (size_t i = 0; i < WINDOW_SIZE - LB_SIZE; i++) {
            currentBuffer[i] = ifftResult[i];
        }

        // Initialize synthesis buffer with zeros
        std::fill(synthesisBuffer.begin(), synthesisBuffer.end(), 0.0);

        // Copy non-overlapping section from previous buffer
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            synthesisBuffer[i] = previousBuffer[i];
        }

        // Blend overlapping section (LF_SIZE samples) with 0.5/0.5 weighting
        for (size_t i = 0; i < LF_SIZE; i++) {
            synthesisBuffer[CHUNK_SIZE + i] = 0.5 * previousBuffer[CHUNK_SIZE + i] + 0.5 * currentBuffer[i];
        }

        // Copy remaining non-overlapping section from current buffer
        for (size_t i = LF_SIZE; i < WINDOW_SIZE - LB_SIZE; i++) {
            synthesisBuffer[CHUNK_SIZE + i] = currentBuffer[i];
        }

        // Take the middle CHUNK_SIZE samples from synthesis buffer
        // These should be from index LB_SIZE to LB_SIZE + CHUNK_SIZE
        std::vector<float> outputChunk(CHUNK_SIZE);
        for (size_t i = 0; i < CHUNK_SIZE; i++) {
            outputChunk[i] = static_cast<float>(synthesisBuffer[LB_SIZE + i]);
        }

        // Store the processed chunk as float values
        size_t oldSize = processedData.size();
        processedData.insert(processedData.end(), 
                           reinterpret_cast<char*>(outputChunk.data()),
                           reinterpret_cast<char*>(outputChunk.data() + CHUNK_SIZE));
        
        // Verify data was written correctly
        if (processedData.size() != oldSize + CHUNK_SIZE * sizeof(float)) {
            std::cerr << "Error: Data size mismatch in processedData" << std::endl;
            return false;
        }

        // Update previous buffer for next iteration
        previousBuffer = currentBuffer;

        return true;
    }

    std::vector<char> getWindowSamples() {
        std::vector<char> window(WINDOW_SIZE * sizeof(float));  // Adjust size for float
        const std::vector<char>& buffer = ringBuffer.getBuffer();
        size_t writePos = ringBuffer.getWritePosition();
        
        for (size_t i = 0; i < WINDOW_SIZE * sizeof(float); i++) {
            size_t pos = (writePos - WINDOW_SIZE * sizeof(float) + i + buffer.size()) % buffer.size();
            window[i] = buffer[pos];
        }
        
        return window;
    }

    void writeHeader(std::ofstream& file) {
        file.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    }

    void writeDataChunk(std::ofstream& file) {
        ChunkHeader dataChunk = {};
        std::memcpy(dataChunk.id, "data", 4);
        dataChunk.size = processedData.size();
        file.write(reinterpret_cast<char*>(&dataChunk), sizeof(ChunkHeader));
    }

    void writeAudioData(std::ofstream& file) {
        file.write(processedData.data(), processedData.size());
    }

    WAVHeader header;
    RingBuffer ringBuffer;
    std::vector<char> processedData;
    
    // FFTW variables
    double* fftIn;
    fftw_complex* fftOut;
    fftw_plan fftPlan;
    
    // IFFTW variables
    fftw_complex* ifftIn;
    double* ifftOut;
    fftw_plan ifftPlan;
    
    // FFT results
    std::vector<double> fftReal;  // Real components
    std::vector<double> fftImag;  // Imaginary components
    
    // IFFT result
    std::vector<double> ifftResult;  // Time domain signal
    
    // Synthesis buffers
    std::vector<double> previousBuffer;  // Previous window (CHUNK_SIZE + LF_SIZE)
    std::vector<double> currentBuffer;   // Current window (CHUNK_SIZE + LF_SIZE)
    std::vector<double> synthesisBuffer; // Combined window (WINDOW_SIZE)
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_wav_file> <output_wav_file>" << std::endl;
        return 1;
    }

    WAVProcessor processor;
    
    if (!processor.readWAV(argv[1])) {
        return 1;
    }

    if (!processor.writeWAV(argv[2])) {
        return 1;
    }

    std::cout << "Successfully processed WAV file" << std::endl;
    return 0;
} 