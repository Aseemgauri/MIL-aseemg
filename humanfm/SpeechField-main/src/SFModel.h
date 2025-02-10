#ifndef __SF_MODEL_H__
#define __SF_MODEL_H__

#include <string>
#include "InferenceWrapper.h"


class SFModel : public InferenceWrapper{
    public:
        SFModel(const std::string &model_path);
        
        void set_embedding(int idx);
        void feed_audio_chunk(const float* data, int num_channels, int samples_per_channel);
        void infer();
        void reset_state();
        
        int get_enrollment_length();
        int get_chunk_length();
        int get_pad_length();
        
        int get_num_input_channels();
        int get_num_output_channels();
    
        float* current_frame;
        float* current_embedding;
    private:
        std::vector<int64_t> frame_shape;
        std::vector<int64_t> embedding_shape;
        std::vector<std::string> ctx_buf_names; // Context buffer names
        int total_num_elements;
        int total_num_classes;
        float* processed;

        int num_input_channels;
        int num_output_channels;
        int chunk_length;
        int pad_length;

};

#endif
