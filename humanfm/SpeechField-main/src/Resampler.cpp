#include "Resampler.h"


ResamplerWrapper::ResamplerWrapper(){

}

ResamplerWrapper::~ResamplerWrapper(){
    zresampler.clear();
}

ResamplerWrapper::ResamplerWrapper(int input_sr, int output_sr, int num_ch){
    // Setup resampler with hlen = 16
    zresampler.setup(input_sr, output_sr, num_ch, 16);

    // Initialise the resampler for zero delay.
    zresampler.inp_count = zresampler.inpsize () - 1;
    zresampler.inp_data = 0;
    zresampler.out_count = 999999;
    zresampler.out_data = 0;
    zresampler.process ();

    upsampling = output_sr > input_sr;

    if(upsampling){
        up_ratio = output_sr / input_sr;
    }else{
        down_ratio = input_sr / output_sr;
    }
}

int ResamplerWrapper::resample(float* output, float* input, int input_nframes){
    zresampler.inp_data = input;
    zresampler.inp_count = input_nframes;
    
    zresampler.out_data = output;
    
    int output_frames = calculate_output_frames(input_nframes);
    zresampler.out_count = output_frames;
    
    zresampler.process();
    
    return output_frames;
}

int ResamplerWrapper::calculate_output_frames(int input_frames){
    return upsampling ? input_frames * up_ratio : input_frames / down_ratio;
}

int ResamplerWrapper::calculate_input_frames(int output_frames){
    return upsampling ? output_frames / up_ratio : output_frames * down_ratio;
}