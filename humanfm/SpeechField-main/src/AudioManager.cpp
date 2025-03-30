#include <jack/jack.h>
#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <unistd.h>  // for usleep

#include "AudioManager.h"

// On some platforms, scaling is required
// ALSA input is different from output (32-bit vs 24-bit)
#define SCALE_REQUIRED false

// Set to 1 to use a dummy mic port when only one physical mic is available
// Set to 0 when you have a real 2-mic setup
#define USE_DUMMY_MIC 0

// Set to 1 to enable passthrough from input to output (for testing)
#define ENABLE_PASSTHROUGH 0

using namespace std::chrono;
auto finish_time = high_resolution_clock::now();
auto start_time = high_resolution_clock::now();

// Declare the dummy mic port globally
#if USE_DUMMY_MIC
jack_port_t *dummy_mic_port = NULL;
#endif
const char *target_physical_port = NULL;

void PCM32_to_PCM24(float *arr, int num_samples){
    #if SCALE_REQUIRED
        
        for(int i = 0; i < num_samples; i++){
            arr[i] /= (1 << 10);
            
        }
    #endif
}

float aux_arr[100000];

int jack_callback_process(jack_nframes_t nframes, void *arg){
    jack_default_audio_sample_t *in, *out;    
    AudioManager *audio_manager = (AudioManager *) (arg);

    // If not running, zero out all output buffers and return
    if (!audio_manager->is_running) {
        for(int i = 0; i < audio_manager->get_num_output_channels(); i++){
            out = (jack_default_audio_sample_t *) jack_port_get_buffer(audio_manager->output_ports[i], nframes);
            memset(out, 0, nframes * sizeof(jack_default_audio_sample_t));
        }
        return 0;
    }

    start_time = high_resolution_clock::now();
    
    #if USE_DUMMY_MIC
    // Fill the dummy mic port buffer with silence
    if (dummy_mic_port != NULL) {
        jack_default_audio_sample_t *dummy_buffer = (jack_default_audio_sample_t *)jack_port_get_buffer(dummy_mic_port, nframes);
        memset(dummy_buffer, 0, sizeof(jack_default_audio_sample_t) * nframes);
    }
    #endif

    float *buf_start;
    float *current_write_buffer = audio_manager->get_input_buffer_ptr(audio_manager->current_input_buffer_step);
    
    // Number of samples to write to the buffer
    uint32_t samples_to_write = nframes;
    // Samples remaining to fill the current buffer
    uint32_t samples_remaining;
    // Whether or not to release the current buffer
    bool release_buffer = false;

    // Go over each input channel
    for(int i = 0; i < audio_manager->get_num_input_channels(); i++){
        in = (jack_default_audio_sample_t *) jack_port_get_buffer(audio_manager->input_ports[i], nframes);

        // At each channel, we want to write as many samples as the Jack client receives
        // Writing starts at the i-th channel with an offset given by the current buffer index
        samples_to_write = nframes;
        buf_start = current_write_buffer + i * audio_manager->input_samples_per_channel + audio_manager->current_input_buffer_idx;
        
        // Calculate remaining samples in current buffer
        samples_remaining = audio_manager->input_samples_per_channel - audio_manager->current_input_buffer_idx;
        
        // If the buffer has fewer samples required to fill it, only do a partial write and release the buffer
        // Then, start writing in the next buffer
        if(samples_remaining <= samples_to_write){
            // Write remaining samples to current buffer
            memcpy(buf_start, in, samples_remaining * sizeof(jack_default_audio_sample_t));

            // Probe input audio for debugging
            if(audio_manager->is_io_monitoring){
                int bytes_to_store = sizeof(float) * samples_remaining;
                if(rb_enough_space(&audio_manager->input_debug_v[i], bytes_to_store)){
                    rb_put(&audio_manager->input_debug_v[i], (uint8_t*) buf_start, bytes_to_store);
                }
            }
            
            // Advance the read buffer
            in += samples_remaining;
            
            // Start writing at the beginning of the other buffer
            buf_start = audio_manager->get_input_buffer_ptr(!audio_manager->current_input_buffer_step) + i * audio_manager->input_samples_per_channel;
            
            // We wrote some samples, now we need to write fewer samples
            samples_to_write = samples_to_write - samples_remaining;

            // Indicate that the buffer should be released
            release_buffer = true;
        }
        
        if(samples_to_write > 0){
            // Write remaining samples
            memcpy(buf_start, in, samples_to_write * sizeof(jack_default_audio_sample_t));

            // Probe input audio for debugging
            if(audio_manager->is_io_monitoring){
                int bytes_to_store = sizeof(float) * samples_to_write;
                if(rb_enough_space(&audio_manager->input_debug_v[i], bytes_to_store)){
                    rb_put(&audio_manager->input_debug_v[i], (uint8_t*) buf_start, bytes_to_store);
                }
            }
        }
    }

    // Release buffer if we have to
    if(release_buffer){
        audio_manager->release_buffer();
    }

    // Advance the current writing index by the amount of samples we've written
    audio_manager->advance_idx(nframes);
    
    // Process output channels
    for(int i = 0; i < audio_manager->get_num_output_channels(); i++){
        ringbuffer_t *output_rb = &audio_manager->output_buffers[i];
        uint32_t samples_remaining = rb_size(output_rb) / sizeof(jack_default_audio_sample_t);
        
        // Read as many samples as needed/available from buffer
        uint32_t samples_to_read = std::min(samples_remaining, (uint32_t)nframes);
        
        // Load data from ringbuffer into auxilliary array
        rb_get_fast(output_rb, (uint8_t*) aux_arr, sizeof(jack_default_audio_sample_t) * samples_to_read);
        rb_advance(output_rb, sizeof(jack_default_audio_sample_t) * samples_to_read);

        // Store output samples to output debug vector
        if(audio_manager->is_io_monitoring){
            int bytes_to_store = sizeof(float) * samples_to_read;
            if(rb_enough_space(&audio_manager->output_debug_v[i], bytes_to_store)){
                rb_put(&audio_manager->output_debug_v[i], (uint8_t*) aux_arr, bytes_to_store);
            }
        }

        // Get output port buffer
        out = (jack_default_audio_sample_t *) jack_port_get_buffer(
            audio_manager->output_ports[i], nframes);

        // Copy data directly to output buffer
        memcpy(out, aux_arr, samples_to_read * sizeof(jack_default_audio_sample_t));
        
        // Fill the rest with zeros if needed
        if(samples_to_read < nframes){
            memset(out + samples_to_read, 0, sizeof(jack_default_audio_sample_t) * (nframes - samples_to_read));
        }
    }

    finish_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(finish_time - start_time);
    printf("Duration %dus\n", duration);

    #if ENABLE_PASSTHROUGH
    // Passthrough from each input to its corresponding output (for testing)
    for(int i = 0; i < std::min(audio_manager->get_num_input_channels(), audio_manager->get_num_output_channels()); i++) {
        in = (jack_default_audio_sample_t *) jack_port_get_buffer(audio_manager->input_ports[i], nframes);
        out = (jack_default_audio_sample_t *) jack_port_get_buffer(audio_manager->output_ports[i], nframes);
        memcpy(out, in, nframes * sizeof(jack_default_audio_sample_t));
    }
    #endif

    return 0;
}

int jack_callback_shutdown(void *arg){
    (void)arg;
    fprintf(stderr, "[AudioManager] SHUTDOWN CALLBACK");

    return 0;
}

const float* AudioManager::wait_for_input_buffer(){
    this->input_buffer_ready.lock();
    
    this->input_buffer_access_lock.lock();
    int num_channels = this->input_port_order.size(); // Number of input channels
    const float* buffer = this->get_input_buffer_ptr(!current_input_buffer_step);
    this->input_buffer_access_lock.unlock();

    return buffer;
}

void AudioManager::write_audio(const float *data, int num_samples, int channel_index){
    if(!rb_put(&output_buffers[channel_index], (uint8_t*) data, sizeof(float) * num_samples)){
        printf("NO SPACE IN OUTPUT BUFFER!\n");
    }
}

AudioManager::AudioManager(){
    const char *client_name = JACK_CLIENT_NAME;
    jack_status_t status;

    // Initialize member variables
    mic_ports = NULL;
    spk_ports = NULL;
    output_buffers = NULL;
    input_buffers = NULL;
    current_input_buffer_step = 0;
    current_input_buffer_idx = 0;
    input_samples_per_channel = 0;
    input_samples_per_buffer = 0;
    is_io_monitoring = false;
    num_physical_input_channels = 0;
    num_physical_output_channels = 0;
    sampling_rate = 0;
    block_size = 0;

    // Open JACK client
    jack_client = jack_client_open(client_name, JackNullOption, &status);
    if (jack_client == NULL) {
        fprintf(stderr, "[AudioManager] jack_client_open() failed, status = 0x%2.0x\n", status);
        if (status & JackServerFailed) {
            fprintf(stderr, "Unable to connect to JACK server\n");
        }
        return;
    }

    // Get JACK parameters
    block_size = jack_get_buffer_size(jack_client);
    sampling_rate = jack_get_sample_rate(jack_client);
    bool realtime = jack_is_realtime(jack_client);
    
    fprintf(stderr, "[AudioManager] JACK Sampling rate: %d\n", sampling_rate);
    if (sampling_rate != 44100) {
        fprintf(stderr, "Error: 44100 Hz is required.\n");
        jack_client_close(jack_client);
        return;
    }
    fprintf(stderr, "[AudioManager] JACK Block size: %d\n", block_size);
    fprintf(stderr, "[AudioManager] Realtime? %d\n", realtime);
    
    // Check if JACK Client name was reassigned
    if (status & JackNameNotUnique) {
        client_name = jack_get_client_name(jack_client);
        fprintf (stderr, "[AudioManager] unique name `%s' assigned\n", client_name);
    }
    
    // Check if JACK Server had been started
    if (status & JackServerStarted) {
        fprintf(stderr, "[AudioManager] JACK server started\n");
    }

    printf("Querying physical ports...\n");
    // Discover physical ports
    query_physical_ports();
    printf("Physical ports queried\n");

    // Register callbacks
    if (jack_set_process_callback(jack_client, jack_callback_process, (void*) this) != 0) {
        fprintf(stderr, "[AudioManager] Failed to set process callback\n");
        jack_client_close(jack_client);
        return;
    }
}

void AudioManager::query_physical_ports(){
    // Get physical input ports (microphones)
    const char **outputs = jack_get_ports(jack_client, NULL, NULL, JackPortIsPhysical|JackPortIsOutput);
    if (outputs == NULL) {
        fprintf(stderr, "[AudioManager] No physical output ports found\n");
        return;
    }

    #if USE_DUMMY_MIC
    // Register a dummy output port that will provide silence
    // This makes the system think there are 2 microphones available
    dummy_mic_port = jack_port_register(jack_client, "dummy_mic", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (dummy_mic_port == NULL) {
        fprintf(stderr, "Failed to register dummy microphone port\n");
    } else {
        fprintf(stderr, "Successfully registered dummy microphone port\n");
    }
    #endif

    // Get all available microphone ports
    if (mic_ports != NULL) {
        free(mic_ports);  // Free existing mic_ports if any
    }
    mic_ports = jack_get_ports(jack_client, NULL, NULL, JackPortIsPhysical|JackPortIsOutput);
    if (mic_ports == NULL) {
        fprintf(stderr, "[AudioManager] No microphone ports found\n");
        free(outputs);
        return;
    }
    
    // Count how many real microphones we have
    this->num_physical_input_channels = 0;
    while (mic_ports[this->num_physical_input_channels] != NULL) {
        this->num_physical_input_channels++;
    }
    
    #if USE_DUMMY_MIC
    // Count our dummy port as an additional input channel
    this->num_physical_input_channels++;
    #endif
    
    // Get physical output ports (speakers)
    if (spk_ports != NULL) {
        free(spk_ports);  // Free existing spk_ports if any
    }
    spk_ports = jack_get_ports(jack_client, NULL, NULL, JackPortIsPhysical|JackPortIsInput);
    if (spk_ports == NULL) {
        fprintf(stderr, "[AudioManager] No physical input ports found\n");
        free(outputs);
        return;
    }
    
    // Count physical output channels
    this->num_physical_output_channels = 0;
    while (spk_ports[this->num_physical_output_channels] != NULL) {
        this->num_physical_output_channels++;
    }
    
    // Free the outputs list if it's different from mic_ports
    if (outputs != NULL && outputs != mic_ports) {
        free(outputs);
    }
    
    // Print port information
    if (spk_ports != NULL && this->num_physical_output_channels >= 2) {
        printf("Speaker Ports: %s, %s\n", spk_ports[0], spk_ports[1]);
    } else {
        printf("Speaker Ports: (none or insufficient)\n");
    }
    
    printf("Num physical inputs: %d\n", this->num_physical_input_channels);
    printf("Num physical outputs: %d\n", this->num_physical_output_channels);
}

AudioManager::AudioManager(int num_input_channels, int num_output_channels): AudioManager(){
    printf("Number of input channels: %d\n", num_input_channels);
    printf("Number of output channels: %d\n", num_output_channels);
    this->setup_ports(num_input_channels, num_output_channels);

    // input_debug_v = std::vector<std::vector<float>>(std::max(this->num_physical_input_channels, num_input_channels));
    // output_debug_v = std::vector<std::vector<float>>(num_output_channels);
}

void AudioManager::allocate_monitoring_buffers(int max_samples, std::vector<int> input_channel_mask, std::vector<int> output_channel_mask){
    assert (input_channel_mask.size() == get_num_input_channels());
    assert (output_channel_mask.size() == get_num_output_channels());
    
    int num_inputs = get_num_input_channels();
    int num_outputs = get_num_input_channels();
    
    // Delete all current monitored data
    for(int i = 0; i < input_debug_v.size(); i++){
        rb_destroy(&input_debug_v[i]);
    }
    input_debug_v.clear();

    for(int i = 0; i < output_debug_v.size(); i++){
        rb_destroy(&output_debug_v[i]);
    }
    output_debug_v.clear();

    // Allocate memory again
    input_debug_v.resize(num_inputs);
    for(int i = 0; i < num_inputs; i++){
        if(input_channel_mask[i]){
            rb_create(&input_debug_v[i], max_samples * sizeof(float));
        }else{
	    rb_init(&input_debug_v[i], NULL, 0);
	}
    }
    
    output_debug_v.resize(num_outputs);
    for(int i = 0; i < num_outputs; i++){
        if(output_channel_mask[i]){
            rb_create(&output_debug_v[i], max_samples * sizeof(float));
        }else{
	    rb_init(&output_debug_v[i], NULL, 0);
	}
    }
}

void AudioManager::reset_channel_order(){
    std::vector<int> input_order;
    // Map the i-th virtual channel to the (i%num_physical_inputs)-th channel
    for(int i = 0; i < this->input_ports.size(); i++){
        input_order.push_back(i % this->num_physical_input_channels);
    }
    // Ensure order passes neccessary checks
    this->set_input_channel_order(input_order); 

    std::vector<int> output_order;
    // Map the i-th physical channel to the (i%num_virtual_outputs)-th channel
    for(int i = 0; i < this->num_physical_output_channels; i++){
        output_order.push_back(i % this->output_ports.size());
    }
    // Ensure order passes neccessary checks
    this->set_output_channel_order(output_order);
}

void AudioManager::setup_ports(int num_input_ports, int num_output_ports){
    // Register virtual input ports
    char input_port_name[32];
    jack_port_t *input_port;
    for(int i = 0; i < num_input_ports; i++){  // Only create the requested number of ports
        sprintf(input_port_name, "input_%02d", i);
        input_port = jack_port_register (jack_client, input_port_name,
                                         JACK_DEFAULT_AUDIO_TYPE,
                                         JackPortIsInput, 0);
        input_ports.push_back(input_port);
    }

    char output_port_name[32];
    jack_port_t *output_port;
    for(int i = 0; i < num_output_ports; i++){
        sprintf(output_port_name, "output_%02d", i);
        output_port = jack_port_register (jack_client, output_port_name,
													  JACK_DEFAULT_AUDIO_TYPE,
													  JackPortIsOutput, 0);
        output_ports.push_back(output_port);
    }
    
    // Initialize output buffers, one for each virtual output port
    output_buffers = new ringbuffer_t[num_output_ports];
    for(int i = 0; i < num_output_ports; i++){
        rb_create(&output_buffers[i], AUDIO_OUTPUT_BUFFER_SIZE);
    }

    // Initialize default port ordering: i-th virtual port -> (i % num_physical_ports)-th physical port
    reset_channel_order();

    // Activate the client so it's ready to process audio
    if (jack_activate(jack_client) != 0) {
        fprintf(stderr, "Error: Cannot activate JACK client\n");
        return;
    }
}

void AudioManager::start_io_monitoring(){
    is_io_monitoring = true; 
}

void AudioManager::stop_io_monitoring(){
    is_io_monitoring = false; 
}

void AudioManager::start(){
    // Connect client (virtual) ports to physical ports
    printf("Connecting input/output ports...\n");
    
    // Get available microphone ports (output ports in JACK terminology)
    const char **available_mic_ports = jack_get_ports(jack_client, NULL, NULL, JackPortIsOutput);
    
    // In case mic_ports wasn't properly initialized
    if (mic_ports == NULL) {
        mic_ports = available_mic_ports;
    } else if (available_mic_ports != NULL) {
        // If we already have mic_ports, free the new ones
        free(available_mic_ports);
    }
    
    for(int i = 0; i < (int) this->input_port_order.size(); i++){
        int physical_port_id = this->input_port_order[i];
        
        // Make sure we don't go out of bounds
        if (physical_port_id >= this->num_physical_input_channels) {
            fprintf(stderr, "Warning: Physical port ID %d is out of bounds (max %d)\n", 
                   physical_port_id, this->num_physical_input_channels - 1);
            continue;
        }
        
        #if USE_DUMMY_MIC
        // If using the dummy mic port and this is the last port (dummy one)
        if (physical_port_id == this->num_physical_input_channels - 1 && dummy_mic_port != NULL) {
            // Connect our dummy mic port to this input
            if (jack_connect(jack_client, jack_port_name(dummy_mic_port), jack_port_name(input_ports[i]))) {
                fprintf(stderr, "ERROR DURING DUMMY MIC CONNECTION\n");
                fprintf(stderr, "Cannot connect ports %s and %s\n",
                        jack_port_name(dummy_mic_port),
                        jack_port_name(input_ports[i]));
            } else {
                printf("Connected dummy mic to %s\n", jack_port_name(input_ports[i]));
            }
            continue; // Skip to next port
        }
        #endif
        
        // Connect to a real microphone port
        if (mic_ports != NULL && mic_ports[physical_port_id] != NULL) {
            if(jack_connect(jack_client, mic_ports[physical_port_id], jack_port_name(input_ports[i]))){
                fprintf(stderr, "ERROR DURING MIC CONNECTION\n");
                fprintf(stderr, "Cannot connect ports %s and %s\n",
                       mic_ports[physical_port_id],
                       jack_port_name(input_ports[i]));
            } else {
                printf("Connected %s to %s\n", mic_ports[physical_port_id], jack_port_name(input_ports[i]));
            }
        } else {
            fprintf(stderr, "No microphone available at index %d\n", physical_port_id);
        }
    }

    for(int i = 0; i < (int) this->output_port_order.size(); i++){
        int virtual_port_id = this->output_port_order[i];
        
        if (virtual_port_id >= (int)this->output_ports.size() || i >= this->num_physical_output_channels) {
            fprintf(stderr, "Invalid output port mapping: %d -> %d\n", virtual_port_id, i);
            continue;
        }

        if (spk_ports != NULL && spk_ports[i] != NULL) {
            if(jack_connect(jack_client, jack_port_name(output_ports[virtual_port_id]), spk_ports[i])){
                fprintf(stderr, "ERROR DURING SPK CONNECTION\n");
                fprintf(stderr, "Cannot connect ports %s and %s\n",
                    spk_ports[i],
                    jack_port_name(output_ports[virtual_port_id]));
            } else {
                printf("Connected %s to %s\n", jack_port_name(output_ports[virtual_port_id]), spk_ports[i]);
            }
        } else {
            fprintf(stderr, "No speaker available at index %d\n", i);
        }
    }

    // Finally, start the callback process
    is_running = true;
}

void AudioManager::stop(){
    // First stop the callback from processing
    is_running = false;
    
    // Now disconnect all ports
    printf("Disconnecting all ports...\n");
    
    // Disconnect input ports
    for(int i = 0; i < input_ports.size(); i++){
        if (input_ports[i] != NULL) {
            const char* port_name = jack_port_name(input_ports[i]);
            if (port_name != NULL) {
                printf("Disconnecting input port %s\n", port_name);
                // Get all connections for this port
                const char** connections = jack_port_get_all_connections(jack_client, input_ports[i]);
                if (connections != NULL) {
                    for (int j = 0; connections[j] != NULL; j++) {
                        jack_disconnect(jack_client, connections[j], port_name);
                    }
                    free(connections);
                }
            }
        }
    }
    
    // Disconnect output ports
    for(int i = 0; i < output_ports.size(); i++){
        if (output_ports[i] != NULL) {
            const char* port_name = jack_port_name(output_ports[i]);
            if (port_name != NULL) {
                printf("Disconnecting output port %s\n", port_name);
                // Get all connections for this port
                const char** connections = jack_port_get_all_connections(jack_client, output_ports[i]);
                if (connections != NULL) {
                    for (int j = 0; connections[j] != NULL; j++) {
                        jack_disconnect(jack_client, port_name, connections[j]);
                    }
                    free(connections);
                }
            }
        }
    }
    
    // Clear all buffers
    for(int i = 0; i < output_ports.size(); i++){
        rb_reset(&output_buffers[i]);
    }
}

void AudioManager::end() {
    static bool already_ended = false;
    if (already_ended) {
        printf("Audio Manager already ended, skipping cleanup\n");
        return;
    }
    already_ended = true;
    
    printf("Ending Audio Manager...\n");
    
    // First stop all processing and disconnect ports
    //stop();
    
    // Deactivate the client
    if (jack_client != NULL && jack_deactivate(jack_client) != 0) {
        fprintf(stderr, "Error: Cannot deactivate JACK client\n");
    }
    
    #if USE_DUMMY_MIC
    // Unregister our dummy mic port
    if (dummy_mic_port != NULL) {
        jack_port_unregister(jack_client, dummy_mic_port);
        dummy_mic_port = NULL;
    }
    #endif
    
    // Close the client
    if (jack_client != NULL) {
        printf("Closing JACK client...\n");
        jack_client_close(jack_client);
        jack_client = NULL;
    }
    
    // Free port arrays if they exist
    if (mic_ports != NULL) {
        free(mic_ports);
        mic_ports = NULL;
    }
    if (spk_ports != NULL) {
        free(spk_ports);
        spk_ports = NULL;
    }
    
    // Free output buffers
    if (output_buffers != NULL) {
        for(int i = 0; i < output_ports.size(); i++){
            rb_destroy(&output_buffers[i]);
        }
        delete[] output_buffers;
        output_buffers = NULL;
    }
    
    // Clear all vectors
    input_ports.clear();
    output_ports.clear();
    input_port_order.clear();
    output_port_order.clear();
    input_debug_v.clear();
    output_debug_v.clear();
    
    // Reset the static flag if we need to reuse this instance
    already_ended = false;
}

AudioManager::~AudioManager(){
    // Call end() to clean up all resources
    end();
}

int AudioManager::get_num_input_channels(){
    return this->input_port_order.size();
}

int AudioManager::get_num_output_channels(){
    return this->output_ports.size();
}

int AudioManager::get_block_size(){
    return block_size;
}

void AudioManager::set_output_channel_order(const std::vector<int> order){
    // Output channel order must have the same dimension as the physical output channels
    assert(int(order.size()) == this->num_physical_output_channels);

    this->output_port_order = order;
}

void AudioManager::set_input_channel_order(const std::vector<int> order){
    // Input channel order must have the same dimension as the virtual input channels
    // assert(order.size() == this->input_ports.size());
    
    this->input_port_order = order;
}

std::vector<int> AudioManager::get_input_channel_order(void){
    return this->input_port_order;
}

float* AudioManager::get_input_buffer_ptr(int double_buffer_idx){
    return this->input_buffers + (double_buffer_idx) * this->input_samples_per_buffer;
}

void AudioManager::configure_input_buffers(float* double_buffer, int samples_per_channel){
    this->current_input_buffer_step = 0;
    this->current_input_buffer_idx = 0;
    
    this->input_buffers = double_buffer;
    this->input_samples_per_channel = samples_per_channel;
    this->input_samples_per_buffer = samples_per_channel * get_num_input_channels();
}

void AudioManager::release_buffer(){
    // this->input_buffer_access_lock.lock();
    this->current_input_buffer_step = !this->current_input_buffer_step;
    this->current_input_buffer_idx = 0;
    // this->input_buffer_access_lock.unlock();

    this->input_buffer_ready.unlock();
}

void AudioManager::advance_idx(int num_samples){
    // this->input_buffer_access_lock.lock();
    this->current_input_buffer_idx = (this->current_input_buffer_idx + num_samples) % this->input_samples_per_channel;
    // this->input_buffer_access_lock.unlock();
}

bool detect_spike(const float* arr, int length){
    // Detects a whether there is some value that crosses a threshold
    for(int i = 0; i < length; i++){
        if(abs(arr[i]) >= 0.8){
            return true;
        }
    }
    return false;
}

std::vector<int> AudioManager::calibrate_channels(int num_channels){
    printf("CALIBRATING!\n");

    // Reset channel order (read from all physical channels)
    reset_channel_order();
    int num_inputs = this->get_num_input_channels();

    // Configure input buffer to be some multiple of block size
    int buf_size = this->block_size * 4;
    
    float *buf = new float[2 * num_inputs * buf_size];
    
    this->configure_input_buffers(buf, buf_size);

    // Create vectors for the new microphone order and bool array to check which channels
    // are successfully calibrated
    std::vector<int> new_order;
    std::vector<bool> finished_calibrating(num_inputs);

    // Start reading audio samples from JACK server
    
    this->start();
    int channel_index, num_frames=0;
    while(new_order.size() < num_channels){
        bool channel_detected = false;
        
        // Read audio until we detect a spike in some channel
        while(!channel_detected){
            // Get audio buffer
            const float *released_mic_buf = this->wait_for_input_buffer();

            // Skip the first few frames for warm-up
            if(num_frames < 4) {
                num_frames++;
                continue;
            }
            
            // Go over each channel in buffer
            for(channel_index = 0; channel_index < num_inputs; channel_index++){
                // If we've already seen a spike in this channel, skip it
                if(finished_calibrating[channel_index]) continue;

                // Monitor spikes along this channel, if we found one, than this channel is next in order
                if(detect_spike(released_mic_buf + channel_index * buf_size, buf_size)){
                    channel_detected = true;
                    num_frames = 0;
                    break;
                }
            }
        }
        
        // Mark this channel as calibrated and add it to the order
        finished_calibrating[channel_index] = 1;
        new_order.push_back(channel_index);

        printf("CHANNEL %d/%d FINISHED\n", new_order.size(), num_channels);
    }
    
    // Stop client
    this->stop();

    // Update order
    
    set_input_channel_order(new_order);
    

    // Cleanup
    delete buf;

    printf("CALIBRATION FINISHED!\nPort order: ");
    for(auto &idx : this->input_port_order){
        printf("%d ", idx);
    }printf("\n");

    return this->input_port_order;
}

void AudioManager::retrieve_monitored_input(std::vector<std::vector<float>> &dst){
    int num_inputs = input_debug_v.size();
    int max_samples = 0; 
    int num_samples;
    float sample;
    
    dst.resize(num_inputs);
    for(int i = 0; i < num_inputs; i++){
        num_samples = rb_size(&input_debug_v[i]) / sizeof(float);
	printf("NUM SAMPLES IN %d-th channel: %d\n", i, num_samples);
        for(int j = 0; j < num_samples; j++){
            // Read sample as float from ringbuffer
            rb_get_fast(&input_debug_v[i], (uint8_t*) &sample, sizeof(float));
            rb_advance(&input_debug_v[i], sizeof(float));
            
            // Add sample to channel
            dst[i].push_back(sample);
        }
        max_samples = std::max(max_samples, num_samples);
    }
	printf("PADDING!\n");
    // Fill the unmasked channels (and channels with remaining samples) with zeros
    for(int i = 0; i < num_inputs; i++){
        while(dst[i].size() < max_samples) dst[i].push_back(0);
    }
}

void AudioManager::retrieve_monitored_output(std::vector<std::vector<float>> &dst){
    int num_outputs = output_debug_v.size();
    int max_samples = 0; 
    int num_samples;
    float sample;
    
    dst.resize(num_outputs);
    for(int i = 0; i < num_outputs; i++){
        num_samples = rb_size(&output_debug_v[i]) / sizeof(float);
        for(int j = 0; j < num_samples; j++){
            // Read sample as float from ringbuffer
            rb_get_fast(&output_debug_v[i], (uint8_t*) &sample, sizeof(float));
            rb_advance(&output_debug_v[i], sizeof(float));
            
            // Add sample to channel
            dst[i].push_back(sample);
        }
        max_samples = std::max(max_samples, num_samples);
    }

    // Fill the unmasked channels (and channels with remaining samples) with zeros
    for(int i = 0; i < num_outputs; i++){
        while(dst[i].size() < max_samples) dst[i].push_back(0);
    }
}
