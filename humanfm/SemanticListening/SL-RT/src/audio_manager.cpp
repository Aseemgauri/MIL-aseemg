#include <jack/jack.h>
#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "audio_manager.h"
#include "unified_ringbuffer.h"

#define JACK_CLIENT_NAME "SH-audio-RT"
#define AUDIO_BUFFER_SIZE 1000000

using namespace std::chrono;

// Global variables for timing
auto process_start_time = high_resolution_clock::now();
auto process_finish_time = high_resolution_clock::now();

// JACK callback function for audio processing
int jack_callback_process(jack_nframes_t nframes, void *arg) {
    process_start_time = high_resolution_clock::now();
    
    AudioManager *audio_manager = (AudioManager *)(arg);
    
    // If not running, zero out outputs and return
    if (!audio_manager->is_running) {
        for (int i = 0; i < 2; i++) {
            jack_default_audio_sample_t *out = (jack_default_audio_sample_t *)jack_port_get_buffer(audio_manager->output_ports[i], nframes);
            memset(out, 0, nframes * sizeof(jack_default_audio_sample_t));
        }
        return 0;
    }
    
    // Read input from microphones and write to ring buffer (interleaved stereo)
    jack_default_audio_sample_t *in0 = (jack_default_audio_sample_t *)jack_port_get_buffer(audio_manager->input_ports[0], nframes);
    jack_default_audio_sample_t *in1 = (jack_default_audio_sample_t *)jack_port_get_buffer(audio_manager->input_ports[1], nframes);
    
    // Interleave stereo samples using pre-allocated buffer (no dynamic allocation)
    for (jack_nframes_t i = 0; i < nframes; i++) {
        audio_manager->interleaved_buffer[i * 2] = in0[i];     // Left channel
        audio_manager->interleaved_buffer[i * 2 + 1] = in1[i]; // Right channel
    }
    
    // Write interleaved data to ring buffer
    audio_manager->input_ringbuffer.write(reinterpret_cast<char*>(audio_manager->interleaved_buffer), 
                                         nframes * 2 * sizeof(float));
    
    // Read processed output from output ring buffer and clone to both stereo channels
    jack_default_audio_sample_t *out_left = (jack_default_audio_sample_t *)jack_port_get_buffer(audio_manager->output_ports[0], nframes);
    jack_default_audio_sample_t *out_right = (jack_default_audio_sample_t *)jack_port_get_buffer(audio_manager->output_ports[1], nframes);
    
    // Try to read from output ring buffer
    size_t available_bytes = audio_manager->output_ringbuffer.getAvailableBytes();
    size_t samples_available = available_bytes / sizeof(float);
    size_t samples_to_read = std::min((size_t)nframes, samples_available);
    
    if (samples_to_read > 0) {
        // Read available samples using pre-allocated buffer (no dynamic allocation)
        if (audio_manager->output_ringbuffer.read(reinterpret_cast<char*>(audio_manager->output_buffer), 
                                                 samples_to_read * sizeof(float))) {
            // Clone mono output to both stereo channels
            for (size_t i = 0; i < samples_to_read; i++) {
                out_left[i] = audio_manager->output_buffer[i];   // Left channel
                out_right[i] = audio_manager->output_buffer[i];  // Right channel (same as left)
            }
        } else {
            // Failed to read, fill with zeros
            for (size_t i = 0; i < samples_to_read; i++) {
                out_left[i] = 0.0f;
                out_right[i] = 0.0f;
            }
        }
    }
    
    // Fill remaining with zeros if needed
    for (size_t i = samples_to_read; i < nframes; i++) {
        out_left[i] = 0.0f;
        out_right[i] = 0.0f;
    }
    
    process_finish_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(process_finish_time - process_start_time);
    
    // Print timing info occasionally
    static int counter = 0;
    if (++counter % 1000 == 0) {
        std::cout << "[AudioManager] Process duration: " << duration.count() << "us, frames: " << nframes << std::endl;
    }
    
    return 0;
}

// JACK shutdown callback
void jack_callback_shutdown(void *arg) {
    AudioManager *audio_manager = (AudioManager *)(arg);
    std::cerr << "[AudioManager] JACK server shutdown!" << std::endl;
    audio_manager->is_running = false;
}

// AudioManager implementation
AudioManager::AudioManager() : 
    input_ringbuffer(),
    output_ringbuffer(),
    is_running(false),
    jack_client(nullptr),
    sampling_rate(16000),
    block_size(0) {
    
    // Initialize input and output ports arrays
    for (int i = 0; i < 2; i++) {
        input_ports[i] = nullptr;
        output_ports[i] = nullptr;
    }
}

AudioManager::~AudioManager() {
    stop();
    cleanup();
}

bool AudioManager::initialize() {
    // Open JACK client - do NOT auto-start server
    jack_status_t status;
    jack_client = jack_client_open(JACK_CLIENT_NAME, JackNoStartServer, &status);
    
    if (jack_client == nullptr) {
        std::cerr << "[AudioManager] Failed to open JACK client - make sure JACK server is running" << std::endl;
        if (status & JackServerFailed) {
            std::cerr << "[AudioManager] JACK server not running. Please start jackd manually." << std::endl;
        }
        return false;
    }
    
    if (status & JackServerStarted) {
        std::cout << "[AudioManager] JACK server started" << std::endl;
    }
    
    if (status & JackNameNotUnique) {
        const char *client_name = jack_get_client_name(jack_client);
        std::cout << "[AudioManager] Unique name assigned: " << client_name << std::endl;
    }
    
    // Get sampling rate and block size
    sampling_rate = jack_get_sample_rate(jack_client);
    block_size = jack_get_buffer_size(jack_client);
    
    std::cout << "[AudioManager] JACK sample rate: " << sampling_rate << " Hz" << std::endl;
    std::cout << "[AudioManager] JACK buffer size: " << block_size << " frames" << std::endl;
    
    // Verify sampling rate
    if (sampling_rate != 16000) {
        std::cerr << "[AudioManager] Warning: Expected 16kHz, got " << sampling_rate << "Hz" << std::endl;
    }
    
    // Set up callbacks
    jack_set_process_callback(jack_client, jack_callback_process, this);
    jack_on_shutdown(jack_client, jack_callback_shutdown, this);
    
    // Create input ports (2 microphones)
    for (int i = 0; i < 2; i++) {
        std::string port_name = "input_" + std::to_string(i);
        input_ports[i] = jack_port_register(jack_client, port_name.c_str(), 
                                          JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
        if (input_ports[i] == nullptr) {
            std::cerr << "[AudioManager] Failed to register input port " << i << std::endl;
            return false;
        }
    }
    
    // Create output ports (2 speakers for stereo)
    for (int i = 0; i < 2; i++) {
        std::string port_name = "output_" + std::to_string(i);
        output_ports[i] = jack_port_register(jack_client, port_name.c_str(), 
                                           JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
        if (output_ports[i] == nullptr) {
            std::cerr << "[AudioManager] Failed to register output port " << i << std::endl;
            return false;
        }
    }
    
    std::cout << "[AudioManager] Ports registered successfully" << std::endl;
    return true;
}

bool AudioManager::start() {
    if (jack_client == nullptr) {
        std::cerr << "[AudioManager] Client not initialized" << std::endl;
        return false;
    }
    
    // Activate the client
    if (jack_activate(jack_client)) {
        std::cerr << "[AudioManager] Failed to activate JACK client" << std::endl;
        return false;
    }
    
    // Get physical ports
    const char **physical_inputs = jack_get_ports(jack_client, nullptr, nullptr, 
                                                  JackPortIsOutput);
    const char **physical_outputs = jack_get_ports(jack_client, nullptr, nullptr, 
                                                   JackPortIsInput);
    
    if (physical_inputs == nullptr) {
        std::cerr << "[AudioManager] No physical input ports found" << std::endl;
        return false;
    }
    
    if (physical_outputs == nullptr) {
        std::cerr << "[AudioManager] No physical output ports found" << std::endl;
        jack_free(physical_inputs);
        return false;
    }
    
    // Count available ports
    int num_physical_inputs = 0;
    int num_physical_outputs = 0;
    
    while (physical_inputs[num_physical_inputs] != nullptr) {
        num_physical_inputs++;
    }
    
    while (physical_outputs[num_physical_outputs] != nullptr) {
        num_physical_outputs++;
    }
    
    std::cout << "[AudioManager] Found " << num_physical_inputs << " physical input ports" << std::endl;
    std::cout << "[AudioManager] Found " << num_physical_outputs << " physical output ports" << std::endl;
    
    // Check if we have at least 2 inputs and 2 outputs
    if (num_physical_inputs < 2) {
        std::cerr << "[AudioManager] Need at least 2 microphone inputs, found " << num_physical_inputs << std::endl;
        jack_free(physical_inputs);
        jack_free(physical_outputs);
        return false;
    }
    
    if (num_physical_outputs < 2) {
        std::cerr << "[AudioManager] Need at least 2 speaker outputs, found " << num_physical_outputs << std::endl;
        jack_free(physical_inputs);
        jack_free(physical_outputs);
        return false;
    }
    
    // Connect input ports (first 2 microphones)
    for (int i = 0; i < 2; i++) {
        std::string our_port = jack_get_client_name(jack_client) + std::string(":input_") + std::to_string(i);
        if (jack_connect(jack_client, physical_inputs[i], our_port.c_str())) {
            std::cerr << "[AudioManager] Failed to connect input port " << i << std::endl;
        } else {
            std::cout << "[AudioManager] Connected " << physical_inputs[i] << " -> " << our_port << std::endl;
        }
    }
    
    // Connect output ports (first 2 speakers for stereo)
    for (int i = 0; i < 2; i++) {
        std::string our_output_port = jack_get_client_name(jack_client) + std::string(":output_") + std::to_string(i);
        if (jack_connect(jack_client, our_output_port.c_str(), physical_outputs[i])) {
            std::cerr << "[AudioManager] Failed to connect output port " << i << std::endl;
        } else {
            std::cout << "[AudioManager] Connected " << our_output_port << " -> " << physical_outputs[i] << std::endl;
        }
    }
    
    jack_free(physical_inputs);
    jack_free(physical_outputs);
    
    is_running = true;
    std::cout << "[AudioManager] Audio processing started" << std::endl;
    
    return true;
}

void AudioManager::stop() {
    is_running = false;
    std::cout << "[AudioManager] Audio processing stopped" << std::endl;
}

void AudioManager::cleanup() {
    if (jack_client != nullptr) {
        jack_client_close(jack_client);
        jack_client = nullptr;
        std::cout << "[AudioManager] JACK client closed" << std::endl;
    }
}

// Get input data from ring buffer (stereo interleaved)
bool AudioManager::getInputData(float* buffer, size_t num_samples) {
    size_t bytes_needed = num_samples * sizeof(float);
    size_t available_bytes = input_ringbuffer.getAvailableBytes();
    
    if (available_bytes < bytes_needed) {
        return false; // Not enough data available
    }
    
    return input_ringbuffer.read(reinterpret_cast<char*>(buffer), bytes_needed);
}

// Write output data to ring buffer
bool AudioManager::writeOutputData(const float* buffer, size_t num_samples) {
    size_t bytes_to_write = num_samples * sizeof(float);
    return output_ringbuffer.write(reinterpret_cast<const char*>(buffer), bytes_to_write);
}

// Get available input samples
size_t AudioManager::getAvailableInputSamples() const {
    return input_ringbuffer.getAvailableBytes() / sizeof(float);
}

// Get available space in output buffer
size_t AudioManager::getAvailableOutputSpace() const {
    return output_ringbuffer.getAvailableSpace() / sizeof(float);
}

int AudioManager::getSampleRate() const {
    return sampling_rate;
}

int AudioManager::getBlockSize() const {
    return block_size;
} 