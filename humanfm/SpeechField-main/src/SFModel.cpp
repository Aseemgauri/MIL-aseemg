#include "SFModel.h"
#include <stdio.h>
#include <algorithm>
#include <iostream>


SFModel::SFModel(const std::string &model_path) : InferenceWrapper(model_path){
    frame_shape = this->get_input_shape("mixture");
    total_num_elements = calculate_product(frame_shape);

    current_frame = new float[total_num_elements];

    // This is not a great way to implement it
    // Maybe change context buffers to be prepended with ctx_buf:: or something
    for(auto &k : input_names_map){
        std::string input_name = k.first;
        if(input_name != "mixture"){
            ctx_buf_names.emplace_back(input_name);
        }
    }

    auto output_shape = this->get_output_shape("filtered_output");
    this->pad_length = (frame_shape[2] - output_shape[2]);
    this->chunk_length = output_shape[2];
    
    this->num_output_channels = output_shape[1];
    this->num_input_channels = frame_shape[1];

    this->reset_state();
}
        
void SFModel::feed_audio_chunk(const float* data, int num_channels, int samples_per_channel){
    if(num_channels != frame_shape[1]){
        printf("[TSEModel] ERROR: Invalid number of channels - Expected %lld, Got %d", frame_shape[1], num_channels);
        return;
    }

    float* frame_channel_start;
    const float* data_channel_start;
    // If input data would is longer than frame, copy last few samples into entire current frame
    if(samples_per_channel >= frame_shape[2]){
        for(int i = 0; i < num_channels; i++){
            frame_channel_start = current_frame + i * frame_shape[2];
            data_channel_start = data + i * samples_per_channel + (samples_per_channel - frame_shape[2]);

            memcpy(frame_channel_start, data_channel_start, frame_shape[2] * sizeof(float));
        }
    }else{
        // Shift to make room for incoming samples
        std::shift_left(current_frame, current_frame + total_num_elements, samples_per_channel);
        for(int i = 0; i < num_channels; i++){
            frame_channel_start = current_frame  + i * frame_shape[2];
            frame_channel_start += (frame_shape[2] - samples_per_channel);
            
            data_channel_start = data + i * samples_per_channel;

            memcpy(frame_channel_start, data_channel_start, samples_per_channel * sizeof(float));
        }
    }

    set_input("mixture", current_frame);
}

void SFModel::reset_state(){    
    clear_input("mixture");
    
    for(const auto &ctx_buf : ctx_buf_names){
        clear_input(ctx_buf);
    }
}

void SFModel::infer(){
    this->run_session();
    
    std::string ctx_buf_out_name;
    for(const auto &ctx_buf : ctx_buf_names){
        ctx_buf_out_name = "out::" + ctx_buf;
        
        Ort::Value &ctx_out_val = get_output(ctx_buf_out_name);
        set_input(ctx_buf, (const float*) ctx_out_val.GetTensorRawData());
    }


    // For runtime test, this was (and should be) uncommented
    //Ort::Value &filtered = get_output("filtered_output");

    //processed = (float*) filtered.GetTensorRawData();
}

int SFModel::get_chunk_length(){
    return this->chunk_length;
}

int SFModel::get_pad_length(){
    return this->pad_length;
}

int SFModel::get_num_input_channels(){
    return this->num_input_channels;
}

int SFModel::get_num_output_channels(){
    return this->num_output_channels;
}
