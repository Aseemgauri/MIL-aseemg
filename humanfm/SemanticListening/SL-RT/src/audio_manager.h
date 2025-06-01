#ifndef AUDIO_MANAGER_H
#define AUDIO_MANAGER_H

#include <jack/jack.h>
#include <atomic>
#include "unified_ringbuffer.h"

class AudioManager {
public:
    AudioManager();
    ~AudioManager();
    
    // Initialize JACK client and ports
    bool initialize();
    
    // Start/stop audio processing
    bool start();
    void stop();
    void cleanup();
    
    // Audio data interface
    bool getInputData(float* buffer, size_t num_samples);
    bool writeOutputData(const float* buffer, size_t num_samples);
    
    // Buffer status
    size_t getAvailableInputSamples() const;
    size_t getAvailableOutputSpace() const;
    
    // Audio parameters
    int getSampleRate() const;
    int getBlockSize() const;
    
    // Public members for JACK callbacks
    std::atomic<bool> is_running;
    jack_port_t* input_ports[2];   // 2 input ports (pre-allocated)
    jack_port_t* output_ports[2];  // 2 output ports for stereo (pre-allocated)
    AudioIOBuffer input_ringbuffer;
    AudioIOBuffer output_ringbuffer;
    
    // Pre-allocated buffers for JACK callback (no dynamic allocation)
    static constexpr size_t MAX_JACK_BUFFER_SIZE = 4096;  // Should be larger than any JACK buffer size
    float interleaved_buffer[MAX_JACK_BUFFER_SIZE * 2];   // For stereo interleaving
    float output_buffer[MAX_JACK_BUFFER_SIZE];            // For output data
    
private:
    jack_client_t* jack_client;
    int sampling_rate;
    int block_size;
};

// JACK callback functions
int jack_callback_process(jack_nframes_t nframes, void *arg);
void jack_callback_shutdown(void *arg);

#endif // AUDIO_MANAGER_H 