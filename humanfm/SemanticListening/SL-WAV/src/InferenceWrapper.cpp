#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <algorithm>
#include <iostream>
#include "InferenceWrapper.h"

const static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, ORT_ENV_NAME);

InferenceWrapper::InferenceWrapper(){
    
}

int calculate_product(const std::vector<std::int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

template <typename T>
Ort::Value arr_to_tensor(T *arr, const std::vector<std::int64_t>& shape){
    Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(
        mem_info, arr, calculate_product(shape), shape.data(), shape.size());
    return tensor;
}

InferenceWrapper::InferenceWrapper(const std::string &model_file){
    // Create ORT Session
    Ort::SessionOptions session_options;
    session = new Ort::Session(env, model_file.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Initialize inputs
    int num_inputs = session->GetInputCount();
    std::vector<std::int64_t> input_shape;

    input_names_in_order = new const char*[num_inputs];
    
    for (std::size_t i = 0; i < num_inputs; i++) {
      // Store input names
      std::string input_name = session->GetInputNameAllocated(i, allocator).get();
      input_names_map[input_name] = i;

      // Make a reference to map key
      input_names_in_order[i] = input_names_map.find(input_name)->first.c_str();
      
      input_shape = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      auto total_number_elements = calculate_product(input_shape);

      // Initialize input tensors with zeros
      float* input_tensor_values = new float[total_number_elements];
      memset(input_tensor_values, 0, sizeof(float) * total_number_elements);
      input_tensors.emplace_back(arr_to_tensor(input_tensor_values, input_shape));	
    }

    // Initialize outputs
    int num_outputs = session->GetOutputCount();

    output_names_in_order = new const char*[num_outputs];

    for (std::size_t i = 0; i < num_outputs; i++) {
      // Store output names
      std::string output_name = session->GetOutputNameAllocated(i, allocator).get();
      output_names_map[output_name] = i;

      // Make a reference to map key
      output_names_in_order[i] = output_names_map.find(output_name)->first.c_str();

      // std::cout << output_name << std::endl;
    }
}

void InferenceWrapper::set_input(const std::string &input_name, const float* data){
  if(input_names_map.find(input_name) == input_names_map.end()){
    printf("ERROR KEY %s NOT FOUND IN INPUTS\n", input_name.c_str());
    return;
  }

  std::vector<int64_t> shape = get_input_shape(input_name);
  auto total_number_elements = calculate_product(shape);
  
  // Get raw data and update it
  int index = input_names_map[input_name];
  float* tensor_buf = (float*) input_tensors[index].GetTensorMutableRawData();

  // Copy new data into old tensor data
  mempcpy(tensor_buf, data, total_number_elements * sizeof(float));
}

Ort::Value& InferenceWrapper::get_output(const std::string &output_name){
  if(output_names_map.find(output_name) == output_names_map.end()){
    printf("ERROR KEY %s NOT FOUND IN OUTPUTS\n", output_name.c_str());
  }

  int index = output_names_map[output_name];
  return output_tensors[index];
}

void InferenceWrapper::run_session(){
  output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names_in_order, input_tensors.data(),
                                input_names_map.size(), output_names_in_order, output_names_map.size());
}

void InferenceWrapper::clear_input(const std::string &input_name){
  std::vector<int64_t> shape = get_input_shape(input_name);
  auto total_number_elements = calculate_product(shape);
  
  // Get raw data and zero it directly
  int index = input_names_map[input_name];
  float* tensor_buf = (float*) input_tensors[index].GetTensorMutableRawData();
  memset(tensor_buf, 0, total_number_elements * sizeof(float));
}

std::vector<std::int64_t> InferenceWrapper::get_input_shape(const std::string &input_name){
  int index = input_names_map[input_name];
  return session->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<std::int64_t> InferenceWrapper::get_output_shape(const std::string &output_name){
  int index = output_names_map[output_name];
  return session->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

void InferenceWrapper::infer(){
  this->run_session();
}

InferenceWrapper::~InferenceWrapper(){
  // Free input tensor data that was allocated during initialization
  for (auto& tensor : input_tensors) {
    if (tensor.IsTensor()) {
      float* tensor_data = static_cast<float*>(tensor.GetTensorMutableRawData());
      delete[] tensor_data;
    }
  }
  
  input_tensors.clear();
  output_tensors.clear();
  
  input_names_map.clear();
  output_names_map.clear();
  
  delete [] output_names_in_order;
  delete [] input_names_in_order;

  delete session;
} 