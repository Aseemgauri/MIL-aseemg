#ifndef __INFERENCE_ENGINE__
#define __INFERENCE_ENGINE__

#include <map>
#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>


#define ORT_ENV_NAME "ClickNTrack"


int calculate_product(const std::vector<std::int64_t>& v);

class InferenceWrapper{
    public:
        InferenceWrapper();
        
        InferenceWrapper(const std::string &model_file);

        void set_input(const std::string &input_name, const float* data);
        Ort::Value& get_output(const std::string &output_name);

        void clear_input(const std::string &input_name);

        std::vector<std::int64_t> get_output_shape(const std::string &output_name);
        std::vector<std::int64_t> get_input_shape(const std::string &input_name);

        ~InferenceWrapper();

        virtual void infer();

    protected:
        void run_session();

        std::map<std::string, int> input_names_map;
        const char** input_names_in_order;

        std::map<std::string, int> output_names_map;
        const char** output_names_in_order;

    private:
        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;
        
        Ort::Session *session;
};


#endif
