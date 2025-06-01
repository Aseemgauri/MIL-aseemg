#include "SL_Model.h"
#include <iostream>
#include <cstring>
#include <algorithm>

SL_Model::SL_Model(const std::string &model_path) : InferenceWrapper(model_path) {
    std::cout << "Initializing SL Model with: " << model_path << std::endl;
    
    initializeModelParams();
    identifyStateBuffers();
    initializeStateBuffers();
    
    std::cout << "SL Model initialized successfully" << std::endl;
    std::cout << "Number of classes: " << num_classes_ << std::endl;
    std::cout << "FFT size: " << fft_size_ << std::endl;
    std::cout << "Input channels: " << input_channels_ << std::endl;
}

SL_Model::~SL_Model() {
    if (current_mixture_) {
        delete[] current_mixture_;
    }
    if (current_embedding_) {
        delete[] current_embedding_;
    }
}

void SL_Model::initializeModelParams() {
    // Get input shapes to determine model parameters
    mixture_shape_ = get_input_shape("mixture_tf");
    embedding_shape_ = get_input_shape("embedding");
    output_shape_ = get_output_shape("output");
    
    // Extract model parameters from shapes
    // mixture_tf shape: [batch, channels, time, freq] = [1, 4, 1, 129]
    input_channels_ = static_cast<int>(mixture_shape_[1]);
    fft_size_ = static_cast<int>(mixture_shape_[3]);
    
    // embedding shape: [batch, num_classes] = [1, 5]
    num_classes_ = static_cast<int>(embedding_shape_[1]);
    
    // Allocate input buffers
    int mixture_elements = calculate_product(mixture_shape_);
    int embedding_elements = calculate_product(embedding_shape_);
    
    current_mixture_ = new float[mixture_elements]();
    current_embedding_ = new float[embedding_elements]();
    
    // Initialize embedding to zeros (will be set per class during processing)
    std::fill(current_embedding_, current_embedding_ + embedding_elements, 0.0f);
}

void SL_Model::identifyStateBuffers() {
    // Identify state buffers by looking for inputs that are not mixture_tf or embedding
    for (const auto &input_pair : input_names_map) {
        const std::string &input_name = input_pair.first;
        if (input_name != "mixture_tf" && input_name != "embedding") {
            state_buffer_names_.push_back(input_name);
            std::cout << "Identified state buffer: " << input_name << std::endl;
        }
    }
}

void SL_Model::initializeStateBuffers() {
    // Initialize all state buffers to zero
    for (const std::string &buffer_name : state_buffer_names_) {
        clear_input(buffer_name);
    }
}

void SL_Model::resetBuffers() {
    // Reset all state buffers to zero
    initializeStateBuffers();
    
    // Reset mixture and embedding inputs
    clear_input("mixture_tf");
    clear_input("embedding");
}

bool SL_Model::processFrame(float* inputFFT, float* outputFFT) {
    if (!inputFFT || !outputFFT) {
        std::cerr << "Error: Null input or output FFT data" << std::endl;
        return false;
    }
    
    try {
        // Set mixture input
        // Input FFT data is expected to be in the format [channels * freq]
        // We need to reshape it to [1, channels, 1, freq]
        std::memcpy(current_mixture_, inputFFT, 
                   input_channels_ * fft_size_ * sizeof(float));
        set_input("mixture_tf", current_mixture_);
        
        // Set embedding to all ones (or zeros - depends on model expectation)
        // For multi-class output, we typically use all ones or a specific pattern
        std::fill(current_embedding_, current_embedding_ + num_classes_, 1.0f);
        set_input("embedding", current_embedding_);
        
        // Run inference
        infer();
        
        // Get output
        Ort::Value &output = get_output("output");
        const float* output_data = static_cast<const float*>(output.GetTensorRawData());
        
        // Copy output data for all classes
        // Output shape is [1, num_classes, 1, freq*2] where freq*2 accounts for complex data
        int output_freq_size = static_cast<int>(output_shape_[3]);
        int total_output_size = num_classes_ * output_freq_size;
        
        std::memcpy(outputFFT, output_data, total_output_size * sizeof(float));
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in processFrame: " << e.what() << std::endl;
        return false;
    }
}

void SL_Model::infer() {
    // Run the base inference
    run_session();
    
    // Update state buffers with outputs
    updateStateBuffers();
}

void SL_Model::updateStateBuffers() {
    // Update state buffers with corresponding outputs
    for (const std::string &buffer_name : state_buffer_names_) {
        std::string output_name = "out::" + buffer_name;
        
        try {
            Ort::Value &output_val = get_output(output_name);
            const float* output_data = static_cast<const float*>(output_val.GetTensorRawData());
            set_input(buffer_name, output_data);
        } catch (const std::exception& e) {
            // Some outputs might not exist, which is okay
            std::cout << "Note: Could not update buffer " << buffer_name 
                     << " (output " << output_name << " not found)" << std::endl;
        }
    }
} 