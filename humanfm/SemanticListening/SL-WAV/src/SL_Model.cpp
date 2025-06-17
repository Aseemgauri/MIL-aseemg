#include "SL_Model.h"
#include <iostream>
#include <cstring>
#include <algorithm>

SL_Model::SL_Model(const std::string &model_path) : InferenceWrapper(model_path) {
    std::cout << "Initializing SL Model with: " << model_path << std::endl;
    
    // Initialize state buffer management
    num_state_buffers_ = 0;
    max_name_length_ = 0;
    state_buffer_names_ = nullptr;
    memset(nameMemoryPool, 0, sizeof(nameMemoryPool));
    
    initializeModelParams();
    identifyStateBuffers();
    allocateNameStorage();
    initializeStateBuffers();
    
    std::cout << "SL Model initialized successfully" << std::endl;
    std::cout << "Number of classes: " << num_classes_ << std::endl;
    std::cout << "FFT size: " << FFT_OUT_SIZE_ << std::endl;
    std::cout << "Input channels: " << input_channels_ << std::endl;
    std::cout << "State buffers: " << num_state_buffers_ << std::endl;
    std::cout << "Max name length: " << max_name_length_ << std::endl;
}

SL_Model::~SL_Model() {
    if (current_mixture_) {
        delete[] current_mixture_;
    }
    if (current_embedding_) {
        delete[] current_embedding_;
    }
    if (state_buffer_names_) {
        delete[] state_buffer_names_;
    }
}

void SL_Model::initializeModelParams() {
    // Get input shapes to determine model parameters
    auto mixture_shape_vec = get_input_shape("mixture_tf");
    auto embedding_shape_vec = get_input_shape("embedding");
    auto output_shape_vec = get_output_shape("output");
    
    // Copy vector data to fixed arrays
    for (size_t i = 0; i < mixture_shape_vec.size() && i < 4; i++) {
        mixture_shape_[i] = mixture_shape_vec[i];
    }
    for (size_t i = 0; i < embedding_shape_vec.size() && i < 2; i++) {
        embedding_shape_[i] = embedding_shape_vec[i];
    }
    for (size_t i = 0; i < output_shape_vec.size() && i < 4; i++) {
        output_shape_[i] = output_shape_vec[i];
    }
    
    // Extract model parameters from shapes
    // mixture_tf shape: [batch, channels, time, freq] = [1, 4, 1, 129]
    input_channels_ = static_cast<int>(mixture_shape_[1]);
    FFT_OUT_SIZE_ = static_cast<int>(mixture_shape_[3]);
    
    // embedding shape: [batch, num_classes] = [1, 5]
    num_classes_ = static_cast<int>(embedding_shape_[1]);
    
    // Allocate input buffers
    int mixture_elements = calculate_product(mixture_shape_vec);
    int embedding_elements = calculate_product(embedding_shape_vec);
    
    current_mixture_ = new float[mixture_elements]();
    current_embedding_ = new float[embedding_elements]();
    
    // Initialize embedding to zeros (will be set per class during processing)
    memset(current_embedding_, 0, embedding_elements * sizeof(float));
}

void SL_Model::identifyStateBuffers() {
    // First pass: count state buffers and find max name length
    num_state_buffers_ = 0;
    max_name_length_ = 0;
    
    for (const auto &input_pair : input_names_map) {
        const std::string &input_name = input_pair.first;
        if (input_name != "mixture_tf" && input_name != "embedding") {
            num_state_buffers_++;
            max_name_length_ = std::max(max_name_length_, static_cast<int>(input_name.length()) + 1);
        }
    }
    
    std::cout << "Discovered " << num_state_buffers_ << " state buffers" << std::endl;
    std::cout << "Maximum name length: " << max_name_length_ << std::endl;
}

void SL_Model::allocateNameStorage() {
    if (num_state_buffers_ == 0) return;
    
    // Calculate memory needed
    int pointersSize = num_state_buffers_ * sizeof(char*);
    int namesSize = num_state_buffers_ * max_name_length_;
    int totalSize = pointersSize + namesSize;
    
    if (totalSize > MAX_MEMORY_POOL_SIZE) {
        std::cerr << "Error: Need " << totalSize << " bytes but pool is " << MAX_MEMORY_POOL_SIZE << std::endl;
        return;
    }
    
    // Allocate pointer array
    state_buffer_names_ = new char*[num_state_buffers_];
    
    // Set up pointers into memory pool
    char* namePtr = nameMemoryPool;
    int bufferIdx = 0;
    
    for (const auto &input_pair : input_names_map) {
        const std::string &input_name = input_pair.first;
        if (input_name != "mixture_tf" && input_name != "embedding") {
            state_buffer_names_[bufferIdx] = namePtr;
            strncpy(namePtr, input_name.c_str(), max_name_length_ - 1);
            namePtr[max_name_length_ - 1] = '\0';
            namePtr += max_name_length_;
            bufferIdx++;
            std::cout << "Stored state buffer: " << state_buffer_names_[bufferIdx - 1] << std::endl;
        }
    }
}

void SL_Model::initializeStateBuffers() {
    // Initialize all state buffers to zero
    for (int i = 0; i < num_state_buffers_; i++) {
        clear_input(state_buffer_names_[i]);
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
    // inputFFT can be a 2D array float[4][FFT_OUT_SIZE] passed as &inputFFT[0][0]
    if (!inputFFT || !outputFFT) {
        std::cerr << "Error: Null input or output FFT data" << std::endl;
        return false;
    }
    try {
        // Set mixture input
        // Input FFT data is expected to be in the format [channels][freq] as a flat buffer
        std::memcpy(current_mixture_, inputFFT, input_channels_ * FFT_OUT_SIZE_ * sizeof(float));
        set_input("mixture_tf", current_mixture_);
        
        // Set embedding to all ones (or zeros - depends on model expectation)
        // For multi-class output, we typically use all ones or a specific pattern
        for (int i = 0; i < num_classes_; i++) {
            current_embedding_[i] = 1.0f;
        }
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
    for (int i = 0; i < num_state_buffers_; i++) {
        std::string buffer_name(state_buffer_names_[i]);
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