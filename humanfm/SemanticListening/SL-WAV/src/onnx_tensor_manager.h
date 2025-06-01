#ifndef ONNX_TENSOR_MANAGER_H
#define ONNX_TENSOR_MANAGER_H

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <iostream>
#include "audio_config.h"

// ONNX tensor manager that discovers model structure then allocates exact memory needed
class ONNXTensorManager {
private:
    // Model structure (discovered first)
    size_t num_inputs_;
    size_t num_outputs_;
    
    // Discovered model information (temporary during discovery)
    struct ModelInfo {
        char name[64];
        int64_t shape[8];
        size_t shape_size;
        size_t buffer_size;
        int buffer_type;  // 0=mixture, 1=embedding, 2=state
        int state_index;  // For state buffers
    };
    
    // Exact memory allocation based on discovered sizes
    char* input_names_storage_;
    char* output_names_storage_;
    int64_t* input_shapes_storage_;
    int64_t* output_shapes_storage_;
    size_t* input_shape_sizes_;
    size_t* output_shape_sizes_;
    size_t* input_buffer_sizes_;
    size_t* output_buffer_sizes_;
    
    // Data buffers (exact sizes)
    alignas(MEMORY_ALIGNMENT) float* mixture_buffer_;
    alignas(MEMORY_ALIGNMENT) float* embedding_buffer_;
    alignas(MEMORY_ALIGNMENT) float** state_buffers_;
    
    // Model structure indices
    int mixture_input_idx_;
    int embedding_input_idx_;
    int* state_buffer_indices_;
    size_t num_state_buffers_;
    
    // ONNX tensors
    Ort::Value* input_tensors_;
    bool* tensors_created_;
    
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    bool initialized_;
    
    // Helper functions
    size_t calculateTensorSize(const int64_t* shape, size_t shape_size) {
        size_t size = 1;
        for (size_t i = 0; i < shape_size; i++) {
            size *= static_cast<size_t>(shape[i]);
        }
        return size;
    }
    
    bool nameMatches(const char* name1, const char* name2) {
        return strcmp(name1, name2) == 0;
    }

public:
    ONNXTensorManager() : 
        num_inputs_(0), num_outputs_(0),
        input_names_storage_(nullptr), output_names_storage_(nullptr),
        input_shapes_storage_(nullptr), output_shapes_storage_(nullptr),
        input_shape_sizes_(nullptr), output_shape_sizes_(nullptr),
        input_buffer_sizes_(nullptr), output_buffer_sizes_(nullptr),
        mixture_buffer_(nullptr), embedding_buffer_(nullptr), state_buffers_(nullptr),
        mixture_input_idx_(-1), embedding_input_idx_(-1),
        state_buffer_indices_(nullptr), num_state_buffers_(0),
        input_tensors_(nullptr), tensors_created_(nullptr),
        initialized_(false) {
        
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    
    // Two-phase initialization: discover then allocate exact memory
    bool initialize(Ort::Session* session) {
        if (!session || initialized_) return false;
        
        try {
            // Phase 1: Discover model structure
            if (!discoverModelStructure(session)) {
                return false;
            }
            
            // Phase 2: Allocate exact memory needed
            if (!allocateMemory()) {
                return false;
            }
            
            // Phase 3: Create tensors
            if (!createTensors()) {
                return false;
            }
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error initializing tensor manager: " << e.what() << std::endl;
            cleanup();
            return false;
        }
    }
    
private:
    bool discoverModelStructure(Ort::Session* session) {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Discover inputs
        num_inputs_ = session->GetInputCount();
        ModelInfo* input_info = new ModelInfo[num_inputs_];
        
        size_t total_name_chars = 0;
        size_t total_shape_elements = 0;
        num_state_buffers_ = 0;
        
        for (size_t i = 0; i < num_inputs_; i++) {
            // Get input name
            auto input_name_ptr = session->GetInputNameAllocated(i, allocator);
            const char* name = input_name_ptr.get();
            strncpy(input_info[i].name, name, 63);
            input_info[i].name[63] = '\0';
            total_name_chars += strlen(input_info[i].name) + 1;
            
            // Get input shape
            auto type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            input_info[i].shape_size = shape.size();
            for (size_t j = 0; j < input_info[i].shape_size; j++) {
                input_info[i].shape[j] = shape[j];
            }
            total_shape_elements += input_info[i].shape_size;
            
            // Calculate buffer size
            input_info[i].buffer_size = calculateTensorSize(input_info[i].shape, input_info[i].shape_size);
            
            // Classify input type
            if (nameMatches(input_info[i].name, "mixture_tf")) {
                input_info[i].buffer_type = 0;
                mixture_input_idx_ = i;
            } else if (nameMatches(input_info[i].name, "embedding")) {
                input_info[i].buffer_type = 1;
                embedding_input_idx_ = i;
            } else {
                input_info[i].buffer_type = 2;
                input_info[i].state_index = num_state_buffers_;
                num_state_buffers_++;
            }
        }
        
        // Discover outputs
        num_outputs_ = session->GetOutputCount();
        ModelInfo* output_info = new ModelInfo[num_outputs_];
        
        for (size_t i = 0; i < num_outputs_; i++) {
            // Get output name
            auto output_name_ptr = session->GetOutputNameAllocated(i, allocator);
            const char* name = output_name_ptr.get();
            strncpy(output_info[i].name, name, 63);
            output_info[i].name[63] = '\0';
            total_name_chars += strlen(output_info[i].name) + 1;
            
            // Get output shape
            auto type_info = session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            
            output_info[i].shape_size = shape.size();
            for (size_t j = 0; j < output_info[i].shape_size; j++) {
                output_info[i].shape[j] = shape[j];
            }
            total_shape_elements += output_info[i].shape_size;
            
            output_info[i].buffer_size = calculateTensorSize(output_info[i].shape, output_info[i].shape_size);
        }
        
        // Now allocate exact memory for metadata
        input_names_storage_ = new char[total_name_chars];
        output_names_storage_ = new char[total_name_chars];
        input_shapes_storage_ = new int64_t[total_shape_elements];
        output_shapes_storage_ = new int64_t[total_shape_elements];
        input_shape_sizes_ = new size_t[num_inputs_];
        output_shape_sizes_ = new size_t[num_outputs_];
        input_buffer_sizes_ = new size_t[num_inputs_];
        output_buffer_sizes_ = new size_t[num_outputs_];
        state_buffer_indices_ = new int[num_inputs_];
        
        // Copy discovered information to allocated storage
        char* name_ptr = input_names_storage_;
        int64_t* shape_ptr = input_shapes_storage_;
        
        for (size_t i = 0; i < num_inputs_; i++) {
            // Copy name
            strcpy(name_ptr, input_info[i].name);
            name_ptr += strlen(input_info[i].name) + 1;
            
            // Copy shape
            for (size_t j = 0; j < input_info[i].shape_size; j++) {
                shape_ptr[j] = input_info[i].shape[j];
            }
            shape_ptr += input_info[i].shape_size;
            
            // Copy metadata
            input_shape_sizes_[i] = input_info[i].shape_size;
            input_buffer_sizes_[i] = input_info[i].buffer_size;
            
            if (input_info[i].buffer_type == 2) {
                state_buffer_indices_[i] = input_info[i].state_index;
            } else {
                state_buffer_indices_[i] = -1;
            }
        }
        
        name_ptr = output_names_storage_;
        shape_ptr = output_shapes_storage_;
        
        for (size_t i = 0; i < num_outputs_; i++) {
            // Copy name
            strcpy(name_ptr, output_info[i].name);
            name_ptr += strlen(output_info[i].name) + 1;
            
            // Copy shape
            for (size_t j = 0; j < output_info[i].shape_size; j++) {
                shape_ptr[j] = output_info[i].shape[j];
            }
            shape_ptr += output_info[i].shape_size;
            
            // Copy metadata
            output_shape_sizes_[i] = output_info[i].shape_size;
            output_buffer_sizes_[i] = output_info[i].buffer_size;
        }
        
        delete[] input_info;
        delete[] output_info;
        return true;
    }
    
    bool allocateMemory() {
        // Allocate exact data buffers based on discovered sizes
        if (mixture_input_idx_ >= 0) {
            mixture_buffer_ = new float[input_buffer_sizes_[mixture_input_idx_]];
            memset(mixture_buffer_, 0, input_buffer_sizes_[mixture_input_idx_] * sizeof(float));
        }
        
        if (embedding_input_idx_ >= 0) {
            embedding_buffer_ = new float[input_buffer_sizes_[embedding_input_idx_]];
            // Initialize embedding to ones
            for (size_t i = 0; i < input_buffer_sizes_[embedding_input_idx_]; i++) {
                embedding_buffer_[i] = 1.0f;
            }
        }
        
        if (num_state_buffers_ > 0) {
            state_buffers_ = new float*[num_state_buffers_];
            for (size_t i = 0; i < num_inputs_; i++) {
                if (state_buffer_indices_[i] >= 0) {
                    state_buffers_[state_buffer_indices_[i]] = new float[input_buffer_sizes_[i]];
                    memset(state_buffers_[state_buffer_indices_[i]], 0, input_buffer_sizes_[i] * sizeof(float));
                }
            }
        }
        
        // Allocate tensor arrays
        input_tensors_ = new Ort::Value[num_inputs_];
        tensors_created_ = new bool[num_inputs_];
        memset(tensors_created_, false, num_inputs_ * sizeof(bool));
        
        return true;
    }
    
    bool createTensors() {
        const char* name_ptr = input_names_storage_;
        const int64_t* shape_ptr = input_shapes_storage_;
        
        for (size_t i = 0; i < num_inputs_; i++) {
            float* buffer_ptr = nullptr;
            
            if (i == mixture_input_idx_) {
                buffer_ptr = mixture_buffer_;
            } else if (i == embedding_input_idx_) {
                buffer_ptr = embedding_buffer_;
            } else if (state_buffer_indices_[i] >= 0) {
                buffer_ptr = state_buffers_[state_buffer_indices_[i]];
            } else {
                std::cerr << "Unknown input type: " << name_ptr << std::endl;
                return false;
            }
            
            // Create tensor using placement new
            new (&input_tensors_[i]) Ort::Value(
                Ort::Value::CreateTensor<float>(
                    *memory_info_,
                    buffer_ptr,
                    input_buffer_sizes_[i],
                    shape_ptr,
                    input_shape_sizes_[i]
                )
            );
            tensors_created_[i] = true;
            
            // Advance pointers
            name_ptr += strlen(name_ptr) + 1;
            shape_ptr += input_shape_sizes_[i];
        }
        
        return true;
    }
    
    void cleanup() {
        // Clean up all allocated memory
        delete[] input_names_storage_;
        delete[] output_names_storage_;
        delete[] input_shapes_storage_;
        delete[] output_shapes_storage_;
        delete[] input_shape_sizes_;
        delete[] output_shape_sizes_;
        delete[] input_buffer_sizes_;
        delete[] output_buffer_sizes_;
        delete[] state_buffer_indices_;
        delete[] mixture_buffer_;
        delete[] embedding_buffer_;
        
        if (state_buffers_) {
            for (size_t i = 0; i < num_state_buffers_; i++) {
                delete[] state_buffers_[i];
            }
            delete[] state_buffers_;
        }
        
        if (input_tensors_) {
            for (size_t i = 0; i < num_inputs_; i++) {
                if (tensors_created_[i]) {
                    input_tensors_[i].~Value();
                }
            }
            delete[] input_tensors_;
        }
        delete[] tensors_created_;
        
        // Reset pointers
        input_names_storage_ = nullptr;
        output_names_storage_ = nullptr;
        input_shapes_storage_ = nullptr;
        output_shapes_storage_ = nullptr;
        input_shape_sizes_ = nullptr;
        output_shape_sizes_ = nullptr;
        input_buffer_sizes_ = nullptr;
        output_buffer_sizes_ = nullptr;
        state_buffer_indices_ = nullptr;
        mixture_buffer_ = nullptr;
        embedding_buffer_ = nullptr;
        state_buffers_ = nullptr;
        input_tensors_ = nullptr;
        tensors_created_ = nullptr;
    }

public:
    // Set mixture input data
    bool setMixtureInput(const float* data, size_t size) {
        if (mixture_input_idx_ < 0 || !mixture_buffer_) return false;
        
        size_t copy_size = std::min(size, input_buffer_sizes_[mixture_input_idx_]);
        memcpy(mixture_buffer_, data, copy_size * sizeof(float));
        return true;
    }
    
    // Get pre-allocated input tensors
    Ort::Value* getInputTensors() {
        return initialized_ ? input_tensors_ : nullptr;
    }
    
    // Get input names as C-style array
    const char* const* getInputNames() const {
        if (!initialized_) return nullptr;
        
        static const char** names = nullptr;
        if (!names) {
            names = new const char*[num_inputs_];
            const char* name_ptr = input_names_storage_;
            for (size_t i = 0; i < num_inputs_; i++) {
                names[i] = name_ptr;
                name_ptr += strlen(name_ptr) + 1;
            }
        }
        return names;
    }
    
    // Get output names as C-style array  
    const char* const* getOutputNames() const {
        if (!initialized_) return nullptr;
        
        static const char** names = nullptr;
        if (!names) {
            names = new const char*[num_outputs_];
            const char* name_ptr = output_names_storage_;
            for (size_t i = 0; i < num_outputs_; i++) {
                names[i] = name_ptr;
                name_ptr += strlen(name_ptr) + 1;
            }
        }
        return names;
    }
    
    // Get tensor counts
    size_t getInputCount() const { return num_inputs_; }
    size_t getOutputCount() const { return num_outputs_; }
    
    // Update state buffers after inference
    void updateStateBuffers(const std::vector<Ort::Value>& output_tensors) {
        if (output_tensors.size() != num_outputs_ || !initialized_) return;
        
        try {
            const char* output_name_ptr = output_names_storage_;
            
            for (size_t out_idx = 0; out_idx < num_outputs_; out_idx++) {
                // Check if this output corresponds to a state buffer input
                if (strncmp(output_name_ptr, "out::", 5) == 0) {
                    const char* input_name = output_name_ptr + 5;  // Skip "out::"
                    
                    // Find corresponding input
                    const char* input_name_ptr = input_names_storage_;
                    for (size_t in_idx = 0; in_idx < num_inputs_; in_idx++) {
                        if (nameMatches(input_name_ptr, input_name) && 
                            state_buffer_indices_[in_idx] >= 0) {
                            
                            // Copy output data to state buffer
                            const float* output_data = output_tensors[out_idx].GetTensorData<float>();
                            size_t buffer_size = input_buffer_sizes_[in_idx];
                            
                            memcpy(state_buffers_[state_buffer_indices_[in_idx]], 
                                   output_data, 
                                   buffer_size * sizeof(float));
                            break;
                        }
                        input_name_ptr += strlen(input_name_ptr) + 1;
                    }
                }
                output_name_ptr += strlen(output_name_ptr) + 1;
            }
        } catch (const std::exception& e) {
            // Handle error silently to avoid disrupting audio processing
        }
    }
    
    // Reset all state buffers to zero
    void resetStateBuffers() {
        if (!initialized_) return;
        
        for (size_t i = 0; i < num_inputs_; i++) {
            if (state_buffer_indices_[i] >= 0) {
                memset(state_buffers_[state_buffer_indices_[i]], 0, 
                       input_buffer_sizes_[i] * sizeof(float));
            }
        }
    }
    
    // Print discovered model information
    void printModelInfo() const {
        if (!initialized_) return;
        
        std::cout << "=== Discovered Model Structure ===" << std::endl;
        std::cout << "Inputs (" << num_inputs_ << "):" << std::endl;
        
        const char* input_name_ptr = input_names_storage_;
        const int64_t* input_shape_ptr = input_shapes_storage_;
        
        for (size_t i = 0; i < num_inputs_; i++) {
            std::cout << "  " << input_name_ptr << ": [";
            for (size_t j = 0; j < input_shape_sizes_[i]; j++) {
                std::cout << input_shape_ptr[j];
                if (j < input_shape_sizes_[i] - 1) std::cout << ", ";
            }
            std::cout << "] (" << input_buffer_sizes_[i] << " elements)";
            
            if (i == mixture_input_idx_) std::cout << " [MIXTURE]";
            else if (i == embedding_input_idx_) std::cout << " [EMBEDDING]";
            else if (state_buffer_indices_[i] >= 0) std::cout << " [STATE_" << state_buffer_indices_[i] << "]";
            
            std::cout << std::endl;
            
            input_name_ptr += strlen(input_name_ptr) + 1;
            input_shape_ptr += input_shape_sizes_[i];
        }
        
        std::cout << "Outputs (" << num_outputs_ << "):" << std::endl;
        
        const char* output_name_ptr = output_names_storage_;
        const int64_t* output_shape_ptr = output_shapes_storage_;
        
        for (size_t i = 0; i < num_outputs_; i++) {
            std::cout << "  " << output_name_ptr << ": [";
            for (size_t j = 0; j < output_shape_sizes_[i]; j++) {
                std::cout << output_shape_ptr[j];
                if (j < output_shape_sizes_[i] - 1) std::cout << ", ";
            }
            std::cout << "] (" << output_buffer_sizes_[i] << " elements)" << std::endl;
            
            output_name_ptr += strlen(output_name_ptr) + 1;
            output_shape_ptr += output_shape_sizes_[i];
        }
        
        std::cout << "State buffers: " << num_state_buffers_ << std::endl;
    }
    
    ~ONNXTensorManager() {
        cleanup();
    }
};

#endif // ONNX_TENSOR_MANAGER_H 