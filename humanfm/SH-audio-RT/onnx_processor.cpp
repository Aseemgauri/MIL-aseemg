#include "onnx_processor.h"
#include <iostream>

OnnxProcessor::OnnxProcessor(const std::string& modelPath) : 
    env_(ORT_LOGGING_LEVEL_WARNING, "onnx_processor") {
    std::cout << "Initializing ONNX processor with model: " << modelPath << std::endl;
    if (!initializeModel(modelPath)) {
        throw std::runtime_error("Failed to initialize ONNX model");
    }
    std::cout << "Model initialized successfully" << std::endl;
    initializeBuffers();
    std::cout << "Buffers initialized successfully" << std::endl;
}

OnnxProcessor::~OnnxProcessor() {
    // Clean up allocated buffer data
    for (auto& pair : buffer_data_) {
        delete[] pair.second;
    }
}

bool OnnxProcessor::initializeModel(const std::string& modelPath) {
    try {
        std::cout << "Creating session options..." << std::endl;
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        std::cout << "Loading ONNX model..." << std::endl;
        session_ = Ort::Session(env_, modelPath.c_str(), session_options);
        std::cout << "Model loaded successfully" << std::endl;
        
        std::cout << "Creating memory info..." << std::endl;
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Get input and output names from the model
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_.GetInputCount();
        size_t num_outputs = session_.GetOutputCount();
        
        std::cout << "Model has " << num_inputs << " inputs and " << num_outputs << " outputs" << std::endl;
        
        input_names_.clear();
        output_names_.clear();
        input_names_.reserve(num_inputs);
        output_names_.reserve(num_outputs);
        
        std::cout << "Getting input names and types..." << std::endl;
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_.GetInputNameAllocated(i, allocator);
            if (name.get() == nullptr || name.get()[0] == '\0') {
                throw std::runtime_error("Empty input name at index " + std::to_string(i));
            }
            // Store the name as a string to ensure proper memory management
            input_names_.push_back(std::string(name.get()));
            std::cout << "Input " << i << ": " << input_names_.back() << std::endl;
            
            // Get input type info
            Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            
            std::cout << "Type: ";
            switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    std::cout << "FLOAT";
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    std::cout << "INT64";
                    break;
                default:
                    std::cout << "UNKNOWN(" << type << ")";
            }
            std::cout << std::endl;
            
            // Print shape information
            std::vector<int64_t> shape = tensor_info.GetShape();
            std::cout << "Shape: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "Getting output names and types..." << std::endl;
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_.GetOutputNameAllocated(i, allocator);
            if (name.get() == nullptr || name.get()[0] == '\0') {
                throw std::runtime_error("Empty output name at index " + std::to_string(i));
            }
            // Store the name as a string to ensure proper memory management
            output_names_.push_back(std::string(name.get()));
            std::cout << "Output " << i << ": " << output_names_.back() << std::endl;
            
            // Get output type info
            Ort::TypeInfo type_info = session_.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            
            std::cout << "Type: ";
            switch (type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    std::cout << "FLOAT";
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    std::cout << "INT64";
                    break;
                default:
                    std::cout << "UNKNOWN(" << type << ")";
            }
            std::cout << std::endl;
            
            // Print shape information
            std::vector<int64_t> shape = tensor_info.GetShape();
            std::cout << "Shape: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                std::cout << shape[j];
                if (j < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

void OnnxProcessor::initializeBuffers() {
    std::cout << "Initializing conv_buf..." << std::endl;
    float* conv_buf_data = new float[4 * 2 * 129]();  // Initialize to zeros
    buffer_data_["conv_buf"] = conv_buf_data;
    buffer_tensors_["conv_buf"] = Ort::Value::CreateTensor<float>(
        memory_info_, conv_buf_data, 4 * 2 * 129, conv_buf_dims_.data(), conv_buf_dims_.size());
    
    std::cout << "Initializing deconv_buf..." << std::endl;
    float* deconv_buf_data = new float[32 * 3 * 129]();  // Initialize to zeros
    buffer_data_["deconv_buf"] = deconv_buf_data;
    buffer_tensors_["deconv_buf"] = Ort::Value::CreateTensor<float>(
        memory_info_, deconv_buf_data, 32 * 3 * 129, deconv_buf_dims_.data(), deconv_buf_dims_.size());
    
    std::cout << "Initializing block buffers..." << std::endl;
    const char* block_buf_names[] = {
        "block_bufs::buf0::h0", "block_bufs::buf0::c0",
        "block_bufs::buf1::h0", "block_bufs::buf1::c0",
        "block_bufs::buf2::h0", "block_bufs::buf2::c0",
        "block_bufs::buf3::h0", "block_bufs::buf3::c0",
        "block_bufs::buf4::h0", "block_bufs::buf4::c0",
        "block_bufs::buf5::h0", "block_bufs::buf5::c0"
    };
    
    for (const char* name : block_buf_names) {
        std::cout << "Creating block buffer: " << name << std::endl;
        float* buf_data = new float[129 * 32]();  // Initialize to zeros
        buffer_data_[name] = buf_data;
        buffer_tensors_[name] = Ort::Value::CreateTensor<float>(
            memory_info_, buf_data, 129 * 32, block_buf_dims_.data(), block_buf_dims_.size());
    }
}

void OnnxProcessor::updateBuffers(const std::vector<Ort::Value>& output_tensors) {
    std::cout << "Updating buffers with " << output_tensors.size() << " output tensors" << std::endl;
    for (size_t i = 0; i < output_tensors.size(); i++) {
        std::string output_name = output_names_[i];
        std::cout << "Processing output: " << output_name << std::endl;
        if (output_name.substr(0, 4) == "out::") {
            std::string input_name = output_name.substr(4);
            std::cout << "Found buffer update for: " << input_name << std::endl;
            if (buffer_tensors_.find(input_name) != buffer_tensors_.end() && 
                buffer_data_.find(input_name) != buffer_data_.end()) {
                const float* output_data = output_tensors[i].GetTensorData<float>();
                size_t num_elements = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();
                std::cout << "Updating buffer with " << num_elements << " elements" << std::endl;
                
                // Update the existing buffer data in place
                std::memcpy(buffer_data_[input_name], output_data, num_elements * sizeof(float));
                
                // The tensor already points to this data, so no need to recreate it
            }
        }
    }
}

bool OnnxProcessor::processFrame(float* inputFFT, float* outputFFT) {
    try {
        std::cout << "Starting frame processing..." << std::endl;
        
        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;
        
        std::cout << "Creating mixture_tf input tensor..." << std::endl;
        // Verify input data is not null
        if (inputFFT == nullptr) {
            throw std::runtime_error("inputFFT is null");
        }
        
        // Create and verify mixture_tf tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, inputFFT, 4 * 129, input_dims_.data(), input_dims_.size());
        if (!input_tensor.IsTensor()) {
            throw std::runtime_error("Failed to create mixture_tf tensor");
        }
        input_tensors.push_back(std::move(input_tensor));
        input_names.push_back(input_names_[0].c_str());
        
        std::cout << "Creating embedding input tensor..." << std::endl;
        float embedding_data[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, embedding_data, 5, embedding_dims_.data(), embedding_dims_.size());
        if (!embedding_tensor.IsTensor()) {
            throw std::runtime_error("Failed to create embedding tensor");
        }
        input_tensors.push_back(std::move(embedding_tensor));
        input_names.push_back(input_names_[1].c_str());
        
        std::cout << "Adding buffer inputs..." << std::endl;
        for (size_t i = 2; i < input_names_.size(); ++i) {
            const std::string& name = input_names_[i];
            std::cout << "Adding buffer: " << name << std::endl;
            if (buffer_tensors_.find(name) == buffer_tensors_.end()) {
                throw std::runtime_error("Buffer not found: " + name);
            }
            if (!buffer_tensors_[name].IsTensor()) {
                throw std::runtime_error("Invalid buffer tensor: " + name);
            }
            
            // Use the existing buffer tensor directly (no copying needed)
            // We need to create a new tensor that references the same data
            float* buffer_data = buffer_data_[name];
            size_t num_elements;
            std::vector<int64_t>* dims;
            
            if (name == "conv_buf") {
                num_elements = 4 * 2 * 129;
                dims = &conv_buf_dims_;
            } else if (name == "deconv_buf") {
                num_elements = 32 * 3 * 129;
                dims = &deconv_buf_dims_;
            } else if (name.find("block_bufs") != std::string::npos) {
                num_elements = 129 * 32;
                dims = &block_buf_dims_;
            } else {
                throw std::runtime_error("Unknown buffer type: " + name);
            }
            
            Ort::Value new_tensor = Ort::Value::CreateTensor<float>(
                memory_info_, buffer_data, num_elements, dims->data(), dims->size());
            
            input_tensors.push_back(std::move(new_tensor));
            input_names.push_back(name.c_str());
        }
        
        std::cout << "Verifying input tensor count..." << std::endl;
        if (input_tensors.size() != input_names_.size()) {
            throw std::runtime_error("Input tensor count mismatch. Expected: " + 
                                   std::to_string(input_names_.size()) + 
                                   ", Got: " + std::to_string(input_tensors.size()));
        }
        
        std::cout << "Running inference..." << std::endl;
        try {
            std::vector<const char*> output_names;
            for (const auto& name : output_names_) {
                output_names.push_back(name.c_str());
            }
            
            auto output_tensors = session_.Run(
                Ort::RunOptions{nullptr}, 
                input_names.data(), 
                input_tensors.data(), 
                input_tensors.size(),
                output_names.data(), 
                output_names.size());
            std::cout << "Inference completed successfully" << std::endl;
                
            std::cout << "Updating buffers..." << std::endl;
            updateBuffers(output_tensors);
            
            std::cout << "Processing output..." << std::endl;
            bool output_found = false;
            for (size_t i = 0; i < output_tensors.size(); i++) {
                if (output_names_[i] == "output") {
                    const float* output_data = output_tensors[i].GetTensorData<float>();
                    if (output_data == nullptr) {
                        throw std::runtime_error("Output tensor data is null");
                    }
                    
                    // Debug prints for output tensor
                    std::cout << "Output tensor shape: [";
                    auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
                    for (size_t j = 0; j < shape.size(); ++j) {
                        std::cout << shape[j];
                        if (j < shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                    
                    // Print first few values of output tensor
                    std::cout << "First 10 output values: ";
                    for (int j = 0; j < 10 && j < 5 * 258; ++j) {
                        std::cout << output_data[j] << " ";
                    }
                    std::cout << std::endl;
                    
                    // Print min/max values
                    float min_val = output_data[0];
                    float max_val = output_data[0];
                    for (int j = 0; j < 5 * 258; ++j) {
                        min_val = std::min(min_val, output_data[j]);
                        max_val = std::max(max_val, output_data[j]);
                    }
                    std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
                    
                    std::memcpy(outputFFT, output_data, 5 * 258 * sizeof(float));
                    std::cout << "Output copied successfully" << std::endl;
                    output_found = true;
                    break;
                }
            }
            if (!output_found) {
                throw std::runtime_error("Output tensor not found in results");
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
            throw;
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error during inference: " << e.what() << std::endl;
        return false;
    }
} 