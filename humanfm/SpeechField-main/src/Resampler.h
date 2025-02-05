#ifndef __RESAMPLER_H__
#define __RESAMPLER_H__

#include "zita-resampler/resampler.h"

class ResamplerWrapper{
    public:
        ResamplerWrapper();
        ResamplerWrapper(int input_sr, int output_sr, int num_ch);
        ~ResamplerWrapper();

        // Returns the number of frames written to the output buffer
        int resample(float* output, float* input, int samples_per_channel);
        
        int calculate_output_frames(int input_frames);
        int calculate_input_frames(int output_frames);
    
    private:
        Resampler zresampler;
        int down_ratio;
        int up_ratio;
        bool upsampling;
};

#endif