#include "utils.h"
#include "AudioFile.h"


int read_wavfile(std::string path, std::vector<std::vector<float>> &audio){
    AudioFile<float> a = AudioFile<float>(path);
    int num_channels = a.getNumChannels();
    audio.resize(num_channels);
    for(int j = 0; j < num_channels; j++){
        for(int i = 0; i < a.getNumSamplesPerChannel(); i++){
            audio[j].push_back(a.samples[j][i]);
        }
    }
    return a.getSampleRate();
}

void write_wavfile(std::string path, const std::vector<std::vector<float>> &audio, int sr){
    int num_channels = audio.size();
    if(num_channels > 0){
        int samples_per_channel = audio[0].size();
        
        AudioFile<float> b = AudioFile<float>();
        b.setNumChannels(num_channels);
        b.setNumSamplesPerChannel(samples_per_channel);
        b.setSampleRate(sr);
        for(int j = 0; j < num_channels; j++){
            for(int i = 0; i < b.getNumSamplesPerChannel(); i++){
                b.samples[j][i] = audio[j][i];
            }
        }
        b.save(path, AudioFileFormat::Wave);
    }
}
