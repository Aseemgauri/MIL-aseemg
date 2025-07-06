#ifndef AUDIO_MANAGER_H
#define AUDIO_MANAGER_H

#include <jack/jack.h>
#include <atomic>
#include "unified_ringbuffer.h"
#include "sl_config.h"  // Include the static configuration (auto-generated)

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
    jack_port_t* input_ports[SL_INPUT_PORTS];   // Input ports (from configuration)
    jack_port_t* output_ports[SL_OUTPUT_PORTS];  // Output ports for stereo (from configuration)
    AudioIOBuffer left_input_ringbuffer;   // Separate buffer for left channel
    AudioIOBuffer right_input_ringbuffer;  // Separate buffer for right channel
    AudioIOBuffer output_ringbuffer;
    
    // Pre-allocated buffers for JACK callback (no dynamic allocation)
    static constexpr size_t MAX_JACK_BUFFER_SIZE = SL_MAX_JACK_BUFFER_SIZE;  // From configuration
    float left_buffer[MAX_JACK_BUFFER_SIZE];              // For left channel storage
    float right_buffer[MAX_JACK_BUFFER_SIZE];             // For right channel storage
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