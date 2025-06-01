#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>

// WAV file header structure
struct WAVHeader {
    // RIFF chunk
    char riff_header[4];      // "RIFF"
    uint32_t wav_size;        // File size - 8
    char wave_header[4];      // "WAVE"
    
    // Format chunk
    char fmt_header[4];       // "fmt "
    uint32_t fmt_chunk_size;  // Format chunk size
    uint16_t audio_format;    // Audio format (3 for IEEE float)
    uint16_t num_channels;    // Number of channels
    uint32_t sample_rate;     // Sample rate
    uint32_t byte_rate;       // Byte rate
    uint16_t block_align;     // Block align
    uint16_t bit_depth;       // Bits per sample
};

// Function to generate a sine wave sample
float generateSineWave(float frequency, float amplitude, float phase, int sampleIndex, int sampleRate) {
    return amplitude * std::sin(2.0f * M_PI * frequency * sampleIndex / sampleRate + phase);
}

int main() {
    const char* outputFile = "test_sine_waves.wav";
    const int sampleRate = 16000;  // 16 kHz
    const float duration = 5.0f;   // 5 seconds
    const int numChannels = 1;     // Mono
    const int bitDepth = 32;       // 32-bit float
    
    // Calculate number of samples
    const int numSamples = static_cast<int>(sampleRate * duration);
    
    // Create WAV header
    WAVHeader header = {};
    std::memcpy(header.riff_header, "RIFF", 4);
    std::memcpy(header.wave_header, "WAVE", 4);
    std::memcpy(header.fmt_header, "fmt ", 4);
    
    header.fmt_chunk_size = 16;  // Size of format chunk
    header.audio_format = 3;     // 3 for IEEE float
    header.num_channels = numChannels;
    header.sample_rate = sampleRate;
    header.bit_depth = bitDepth;
    header.byte_rate = sampleRate * numChannels * (bitDepth / 8);
    header.block_align = numChannels * (bitDepth / 8);
    
    // Calculate total file size
    header.wav_size = 36 + (numSamples * numChannels * (bitDepth / 8));  // 36 = header size - 8
    
    // Generate audio data
    std::vector<float> audioData(numSamples);
    
    // Generate multiple sine waves
    for (int i = 0; i < numSamples; i++) {
        // Generate multiple sine waves with different frequencies
        float sample = 0.0f;
        sample += generateSineWave(440.0f, 0.5f, 0.0f, i, sampleRate);    // 440 Hz
        sample += generateSineWave(880.0f, 0.3f, 0.0f, i, sampleRate);    // 880 Hz
        sample += generateSineWave(1320.0f, 0.2f, 0.0f, i, sampleRate);   // 1320 Hz
        sample += generateSineWave(1760.0f, 0.1f, 0.0f, i, sampleRate);   // 1760 Hz
        
        // Normalize to prevent clipping
        sample *= 0.5f;  // Scale down to prevent overflow
        
        audioData[i] = sample;
    }
    
    // Write WAV file
    std::ofstream file(outputFile, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file: " << outputFile << std::endl;
        return 1;
    }
    
    // Write header
    file.write(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    
    // Write "data" chunk header
    char dataHeader[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize = numSamples * numChannels * (bitDepth / 8);
    file.write(dataHeader, 4);
    file.write(reinterpret_cast<char*>(&dataSize), 4);
    
    // Write audio data
    file.write(reinterpret_cast<char*>(audioData.data()), audioData.size() * sizeof(float));
    
    file.close();
    
    std::cout << "Generated WAV file: " << outputFile << std::endl;
    std::cout << "Duration: " << duration << " seconds" << std::endl;
    std::cout << "Sample Rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "Bit Depth: " << bitDepth << " bits (float)" << std::endl;
    std::cout << "Channels: " << numChannels << std::endl;
    std::cout << "Sine wave components:" << std::endl;
    std::cout << "  - 440 Hz (amplitude: 0.5)" << std::endl;
    std::cout << "  - 880 Hz (amplitude: 0.3)" << std::endl;
    std::cout << "  - 1320 Hz (amplitude: 0.2)" << std::endl;
    std::cout << "  - 1760 Hz (amplitude: 0.1)" << std::endl;
    
    return 0;
} 