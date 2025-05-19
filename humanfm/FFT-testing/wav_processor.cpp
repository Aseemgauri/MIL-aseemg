#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <complex>
#include <algorithm>
#include <fftw3.h>
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
    }

    // Destructor: Cleans up FFTW resources
    ~WAVProcessor() {
        cleanupFFTW();
    }

    // Main processing function: Reads input WAV file, processes audio data, and writes to output file
    bool processWAV(const std::string& inputFile, const std::string& outputFile) {
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open input file: " << inputFile << std::endl;
            return false;
        }

        std::ofstream outFile(outputFile, std::ios::binary);
        if (!outFile.is_open()) {
            std::cerr << "Error: Could not open output file: " << outputFile << std::endl;
            inFile.close();
            return false;
        }

        // Read and validate main header
        if (!readAndValidateHeader(inFile)) {
            inFile.close();
            outFile.close();
            return false;
        }

        // Find and read data chunk
        uint32_t dataSize;
        if (!findDataChunk(inFile, dataSize)) {
            inFile.close();
            outFile.close();
            return false;
        }

        // Write header to output file
        writeHeader(outFile);

        // Write data chunk header
        ChunkHeader dataChunk = {};
        std::memcpy(dataChunk.id, "data", 4);
        dataChunk.size = dataSize;  // We'll update this later
        outFile.write(reinterpret_cast<char*>(&dataChunk), sizeof(ChunkHeader));

        // Process audio data and write immediately
        if (!processAndWriteAudioData(inFile, outFile, dataSize)) {
            inFile.close();
            outFile.close();
            return false;
        }

        // Update data chunk size in output file
        size_t dataWritten = static_cast<size_t>(outFile.tellp()) - (sizeof(WAVHeader) + sizeof(ChunkHeader));
        outFile.seekp(sizeof(WAVHeader) + 4);  // Skip to size field of data chunk
        uint32_t finalSize = static_cast<uint32_t>(dataWritten);
        outFile.write(reinterpret_cast<char*>(&finalSize), sizeof(uint32_t));

        inFile.close();
        outFile.close();
        return true;
    }

    // Getter methods for FFT results
    const double* getFFTReal() const { return g_fftReal; }
    const double* getFFTImag() const { return g_fftImag; }

private:
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

    // Processes audio data in chunks and writes to output file
    bool processAndWriteAudioData(std::ifstream& inFile, std::ofstream& outFile, uint32_t dataSize) {
        size_t totalBytesRead = 0;
        
        while (totalBytesRead < dataSize) {
            if (!readAndProcessChunk(inFile, dataSize, totalBytesRead)) {
                return false;
            }
            
            // Write the processed chunk immediately
            outFile.write(g_chunkBuffer, CHUNK_SIZE * sizeof(float));
        }
        
        return true;
    }

    // Reads a chunk of audio data and processes it
    bool readAndProcessChunk(std::ifstream& file, uint32_t dataSize, size_t& totalBytesRead) {
        // Read a chunk from the file
        size_t bytesToRead = (CHUNK_SIZE * sizeof(float) < (dataSize - totalBytesRead)) ? 
                            CHUNK_SIZE * sizeof(float) : 
                            (dataSize - totalBytesRead);
        file.read(g_chunkBuffer, bytesToRead);
        
        // Pad the last chunk with zeros if necessary
        if (bytesToRead < CHUNK_SIZE * sizeof(float)) {
            std::fill(g_chunkBuffer + bytesToRead, g_chunkBuffer + CHUNK_SIZE * sizeof(float), 0);
        }
        
        // Process the chunk
        if (!processChunk()) {
            return false;
        }
        
        totalBytesRead += bytesToRead;
        return true;
    }

    // Processes a single chunk of audio data through FFT and IFFT
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

    // Performs inverse FFT on the processed frequency domain data
    bool performIFFT() {
        // Prepare for IFFT
        for (size_t i = 0; i < FFT_SIZE; i++) {
            ifftIn[i][0] = g_fftReal[i];  // Real part
            ifftIn[i][1] = g_fftImag[i];  // Imaginary part
        }
        
        // Compute IFFT
        fftw_execute(ifftPlan);
        
        // Store IFFT result (normalize by window size)
        for (size_t i = 0; i < WINDOW_SIZE; i++) {
            g_ifftResult[i] = ifftOut[i] / WINDOW_SIZE;
        }

        return true;
    }

    // Performs overlap-add synthesis on the IFFT result
    bool performOverlapAdd() {
        // Store the current IFFT result in the current buffer
        std::copy(g_ifftResult, g_ifftResult + BUFFER_SIZE, g_currentBuffer);

        // Copy the last ISTFT_OUTPUT_SIZE frames to the current context buffer
        double* ctxPtr = getIstftContextBuffer(g_lookbackBufIdx);
        std::copy(g_ifftResult + (WINDOW_SIZE - ISTFT_OUTPUT_SIZE),
                 g_ifftResult + WINDOW_SIZE,
                 ctxPtr);

        // Do overlap-add for the current chunk
        float* outputChunk = reinterpret_cast<float*>(g_chunkBuffer);
        for (int j = 0; j < CHUNK_SIZE; j++) {
            // Accumulate result from all relevant lookback buffers
            double result = 0.0;
            for (int8_t historyBufIdx = g_lookbackBufIdx;
                 historyBufIdx > g_lookbackBufIdx - ISTFT_LOOKBACK_BUFFERS;
                 historyBufIdx--) {
                
                // Get the lookback buffer
                double* historyPtr = getIstftContextBuffer(historyBufIdx);
                
                // Calculate the index in the history buffer
                int ctxIdx = j + (g_lookbackBufIdx - historyBufIdx) * CHUNK_SIZE;
                if (ctxIdx < ISTFT_OUTPUT_SIZE) {
                    // Apply synthesis window and accumulate
                    result += historyPtr[ctxIdx] * g_synthesisWindow[ctxIdx];
                }
            }
            
            // Store the result
            outputChunk[j] = static_cast<float>(result);
        }

        // Advance the lookback buffer index
        g_lookbackBufIdx = (g_lookbackBufIdx + 1) % ISTFT_LOOKBACK_BUFFERS;

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

    // Writes the WAV header to the output file
    void writeHeader(std::ofstream& file) {
        file.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    }

    WAVHeader header;
    RingBuffer ringBuffer;
    
    // FFTW variables
    double* fftIn;
    fftw_complex* fftOut;
    fftw_plan fftPlan;
    
    // IFFTW variables
    fftw_complex* ifftIn;
    double* ifftOut;
    fftw_plan ifftPlan;
};

// Main entry point: Processes command line arguments and runs the WAV processor
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_wav_file> <output_wav_file>" << std::endl;
        return 1;
    }

    WAVProcessor processor;
    
    if (!processor.processWAV(argv[1], argv[2])) {
        return 1;
    }

    std::cout << "Successfully processed WAV file" << std::endl;
    return 0;
} 