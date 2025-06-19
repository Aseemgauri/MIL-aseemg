#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "SL_Model.h"

using namespace std::chrono;

// Generate realistic FFT data that mimics real audio FFT characteristics
void generateRealisticFFTData(float* input_fft, int input_channels, int fft_size) {
    // FFT data format: [left_real, right_real, left_imag, right_imag]
    // Each channel has fft_size elements (129 for this model)
    
    for (int ch = 0; ch < input_channels; ch++) {
        float* channel_data = input_fft + ch * fft_size;
        
        for (int freq = 0; freq < fft_size; freq++) {
            // Generate realistic frequency-domain characteristics
            // Lower frequencies typically have higher magnitudes
            float frequency_factor = 1.0f / (1.0f + freq * 0.1f);
            
            // Add some randomness but maintain realistic ranges
            float magnitude = frequency_factor * (0.5f + 0.5f * static_cast<float>(rand()) / RAND_MAX);
            
            // Add some phase variation
            float phase = 2.0f * M_PI * static_cast<float>(rand()) / RAND_MAX;
            
            // Convert to real/imaginary parts
            if (ch % 2 == 0) {  // Real parts (left_real, right_real)
                channel_data[freq] = magnitude * cos(phase);
            } else {  // Imaginary parts (left_imag, right_imag)
                channel_data[freq] = magnitude * sin(phase);
            }
            
            // Ensure DC component (freq=0) is real and positive for real signals
            if (freq == 0 && ch % 2 == 1) {
                channel_data[freq] = 0.0f;  // Imaginary part of DC should be zero
            }
        }
    }
}

bool test_runtime(const std::string &model_path) {
    try {
        // Initialize model
        SL_Model model(model_path);
        std::cout << "Initialized SL Model successfully" << std::endl;
        
        // Get model parameters
        int num_classes = model.getNumClasses();
        int fft_size = model.getFFTSize();
        int input_channels = model.getInputChannels();
        
        std::cout << "Model parameters:" << std::endl;
        std::cout << "  Number of classes: " << num_classes << std::endl;
        std::cout << "  FFT size: " << fft_size << std::endl;
        std::cout << "  Input channels: " << input_channels << std::endl;
        
        // Prepare input data (realistic FFT values for testing)
        int input_size = input_channels * fft_size;
        float* input_fft = new float[input_size];
        float* output_fft = new float[num_classes * fft_size * 2]; // *2 for complex data
        
        // Reset model state
        model.resetBuffers();
        
        // Warm up run with initial realistic FFT data
        generateRealisticFFTData(input_fft, input_channels, fft_size);
        model.processFrame(input_fft, output_fft);
        
        const int RUNS = 10000;
        float times[RUNS];
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        
        std::cout << "Running " << RUNS << " inference iterations..." << std::endl;
        
        for (int i = 0; i < RUNS; i++) {
            // Generate new realistic FFT data for each iteration
            generateRealisticFFTData(input_fft, input_channels, fft_size);
            
            start = high_resolution_clock::now();
            bool success = model.processFrame(input_fft, output_fft);
            stop = high_resolution_clock::now();
            
            if (!success) {
                std::cerr << "Error: Inference failed at iteration " << i << std::endl;
                delete[] input_fft;
                delete[] output_fft;
                return false;
            }
            
            times[i] = duration_cast<microseconds>(stop - start).count();
        }
        
        // Calculate mean time
        float mean_time = 0;
        for (int i = 0; i < RUNS; i++) {
            mean_time += times[i];
        }
        mean_time /= RUNS;
        
        // Calculate standard deviation
        float variance = 0;
        for (int i = 0; i < RUNS; i++) {
            float diff = times[i] - mean_time;
            variance += diff * diff;
        }
        variance /= RUNS;
        float std_dev = sqrt(variance);
        
        // Sort times for quartile calculation
        std::vector<float> sorted_times(times, times + RUNS);
        std::sort(sorted_times.begin(), sorted_times.end());
        
        // Calculate quartiles
        int n = RUNS;
        float q1 = sorted_times[n / 4];                    // 1st quartile (25th percentile)
        float median = sorted_times[n / 2];                // Median (50th percentile)
        float q3 = sorted_times[3 * n / 4];                // 3rd quartile (75th percentile)
        
        // Write results to file
        std::ofstream out("runtime_test_results.txt");
        if (out.is_open()) {
            for (int i = 0; i < RUNS; i++) {
                out << std::setprecision(10) << std::fixed << times[i] << "\n";
            }
            out.close();
            std::cout << "Results written to runtime_test_results.txt" << std::endl;
        } else {
            std::cerr << "Warning: Could not write results to file" << std::endl;
        }
        
        // Print comprehensive statistics
        std::cout << "Runtime test completed successfully" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Performance Statistics (microseconds):" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Mean:        " << std::setprecision(2) << std::fixed << mean_time << " μs" << std::endl;
        std::cout << "Std Dev:     " << std::setprecision(2) << std::fixed << std_dev << " μs" << std::endl;
        std::cout << "Min:         " << std::setprecision(2) << std::fixed << sorted_times[0] << " μs" << std::endl;
        std::cout << "Q1 (25%):    " << std::setprecision(2) << std::fixed << q1 << " μs" << std::endl;
        std::cout << "Median (50%): " << std::setprecision(2) << std::fixed << median << " μs" << std::endl;
        std::cout << "Q3 (75%):    " << std::setprecision(2) << std::fixed << q3 << " μs" << std::endl;
        std::cout << "Max:         " << std::setprecision(2) << std::fixed << sorted_times[n-1] << " μs" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Calculate additional useful statistics
        float range = sorted_times[n-1] - sorted_times[0];
        float iqr = q3 - q1;  // Interquartile range
        float cv = (std_dev / mean_time) * 100.0f;  // Coefficient of variation (percentage)
        
        std::cout << "Additional Statistics:" << std::endl;
        std::cout << "Range:       " << std::setprecision(2) << std::fixed << range << " μs" << std::endl;
        std::cout << "IQR:         " << std::setprecision(2) << std::fixed << iqr << " μs" << std::endl;
        std::cout << "CV:          " << std::setprecision(2) << std::fixed << cv << "%" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Cleanup
        delete[] input_fft;
        delete[] output_fft;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in runtime test: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_file_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " ./pretrained/tfgridnet.onnx" << std::endl;
        return -1;
    }
    
    std::string model_path = argv[1];
    
    // Check if model file exists
    std::ifstream file_check(model_path);
    if (!file_check.good()) {
        std::cerr << "Error: Model file not found at " << model_path << std::endl;
        std::cerr << "Please provide the full path to the ONNX model file" << std::endl;
        return -1;
    }
    file_check.close();
    
    std::cout << "Starting SL Model runtime test..." << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    
    bool success = test_runtime(model_path);
    
    if (success) {
        std::cout << "[SUCCESS] Runtime test completed" << std::endl;
        return 0;
    } else {
        std::cout << "[FAILED] Runtime test failed" << std::endl;
        return -1;
    }
} 