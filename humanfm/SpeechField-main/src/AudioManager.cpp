#include <jack/jack.h>
#include <sys/types.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>

#include "AudioManager.h"

// On some platforms, scaling is required
// ALSA input is different from output (32-bit vs 24-bit)
#define SCALE_REQUIRED false

using namespace std::chrono;
auto finish_time = high_resolution_clock::now();
auto start_time = high_resolution_clock::now();

void PCM32_to_PCM24(float *arr, int num_samples){
    #if SCALE_REQUIRED
        
        for(int i = 0; i < num_samples; i++){
            arr[i] /= (1 << 10);
            
        }
    #endif
}

float aux_arr[100000];

int jack_callback_process(jack_nframes_t nframes, void *arg){
    start_time = high_resolution_clock::now();
    
    jack_default_audio_sample_t *in, *out;    
    AudioManager *audio_manager = (AudioManager *) (arg);

    float *buf_start;
    // float *current_write_buffer = &audio_manager->input_buffers[audio_manager->current_input_buffer_step];
    float *current_write_buffer = audio_manager->get_input_buffer_ptr(audio_manager->current_input_buffer_step);
    
    // Number of samples to write to the buffer
    uint32_t samples_to_write = nframes;

    // Samples remaining to fill the current buffer
    uint32_t samples_remaining;

    // Samples written to buffer
    uint32_t input_samples_written;
    
    // Whether or not to release the current buffer
    bool release_buffer = false;

    // Go over each input channel
    for(int i = 0; i < audio_manager->get_num_input_channels(); i++){
        in = (jack_default_audio_sample_t *) jack_port_get_buffer(audio_manager->input_ports[i], nframes);

        // At each channel, we want to write as many samples as the Jack client receives
        // Writing starts at the i-th channel with an offset given by the current buffer index
        samples_to_write = nframes;
        buf_start = current_write_buffer + i * audio_manager->input_samples_per_channel + audio_manager->current_input_buffer_idx;
        
        // Calculate the number of samples remaining to fill the input buffer at the original sampling rate
        samples_remaining = audio_manager->input_resamplers[i]\
            ->calculate_input_frames(audio_manager->input_samples_per_channel - audio_manager->current_input_buffer_idx);
        
        // If the buffer has fewer samples required to fill it, only do a partial write and release the buffer
        // Then, start writing in the next buffer
        if(samples_remaining <= samples_to_write){
            // Write however many samples are left to fill the buffer
            
            // memcpy(buf_start, in, samples_remaining * sizeof(jack_default_audio_sample_t));
            input_samples_written = audio_manager->input_resamplers[i]->resample(buf_start, in, samples_remaining);
            // printf("%d %d\n", samples_remaining, input_samples_written);

            // Probe input audio for debugging
            if(audio_manager->is_io_monitoring){
                // If there is enough space to write audio, then we write
                // Note that for unmasked channels, the ringbuffer capacity is zero and so this
                // statement is always false.
                int bytes_to_store = sizeof(float) * input_samples_written;
                if(rb_enough_space(&audio_manager->input_debug_v[i], bytes_to_store)){
                    rb_put(&audio_manager->input_debug_v[i], (uint8_t*) buf_start, bytes_to_store);
                }
            }
            
            // Advance the read buffer
            in += samples_remaining;
            
            // Start writing at the beginning of the other buffer
            // buf_start = &audio_manager->input_buffers[!audio_manager->current_input_buffer_step] + i * audio_manager->input_buffer_size;
            buf_start = audio_manager->get_input_buffer_ptr(!audio_manager->current_input_buffer_step) + i * audio_manager->input_samples_per_channel;
            
            // We wrote some samples, now we need to write fewer samples
            samples_to_write = samples_to_write - samples_remaining;

            // Indicate that the buffer should be released
            release_buffer = true;
        }
        
        if(samples_to_write > 0){
            // memcpy(buf_start, in, samples_to_write * sizeof(jack_default_audio_sample_t));
            input_samples_written = audio_manager->input_resamplers[i]->resample(buf_start, in, samples_to_write);

            // Probe input audio for debugging
            // if(i == 0){
                // audio_manager->input_debug_v[i].insert(audio_manager->input_debug_v[i].end(), buf_start, buf_start + input_samples_written);
            // }
            if(audio_manager->is_io_monitoring){
                // If there is enough space to write audio, then we write
                // Note that for unmasked channels, the ringbuffer capacity is zero and so this
                // statement is always false.
                int bytes_to_store = sizeof(float) * input_samples_written;
                if(rb_enough_space(&audio_manager->input_debug_v[i], bytes_to_store)){
                    rb_put(&audio_manager->input_debug_v[i], (uint8_t*) buf_start, bytes_to_store);
                }else{
		        }
            }
        }else{
            input_samples_written = 0;
        }
    }

    // Release buffer if we have to
    if(release_buffer){
        audio_manager->release_buffer();
    }

    // Advance the current writing index by the amount of samples we've written
    audio_manager->advance_idx(input_samples_written);
    
    // Number of frames required to read from output ringbuffer
    uint32_t max_samples_to_read;
    int samples_to_read, samples_written;

    // FIXME: This may cause weird errors if one channel is written to but not the others
    for(int i = 0; i < audio_manager->get_num_output_channels(); i++){
        ringbuffer_t *output_rb = &audio_manager->output_buffers[i];
        uint32_t samples_remaining = rb_size(output_rb) / sizeof(jack_default_audio_sample_t);
        
        // Read as many samples as needed/available from buffer
        // This will depend on the output sampling rate vs. input sampling rate
        max_samples_to_read = audio_manager->output_resamplers[i]->calculate_input_frames(nframes);
        samples_to_read = std::min(samples_remaining, max_samples_to_read);

	//printf("Samples remaining: %d\t To read: %d\n", samples_remaining, samples_to_read);
        
        // Load data from ringbuffer into auxilliary array
        rb_get_fast(output_rb, (uint8_t*) aux_arr, sizeof(jack_default_audio_sample_t) * samples_to_read);
        rb_advance(output_rb, sizeof(jack_default_audio_sample_t) * samples_to_read);

        // Store output samples to output debug vector
	    // audio_manager->output_debug_v[i].insert(audio_manager->output_debug_v[i].end(), aux_arr, aux_arr + samples_to_read);
        
        if(audio_manager->is_io_monitoring){
            // If there is enough space to write audio, then we write
            // Note that for unmasked channels, the ringbuffer capacity is zero and so this
            // statement is always false.
            int bytes_to_store = sizeof(float) * samples_to_read;
            if(rb_enough_space(&audio_manager->output_debug_v[i], bytes_to_store)){
                rb_put(&audio_manager->output_debug_v[i], (uint8_t*) aux_arr, bytes_to_store);
            }
        }

        // Get output port buffer
        out = (jack_default_audio_sample_t *) jack_port_get_buffer(
            audio_manager->output_ports[i], nframes);

        // Resample and store result directly in output buffer
        // Also get the number of output samples written from this
        samples_written = audio_manager->output_resamplers[i]->resample(out, aux_arr, samples_to_read);

        // FIXME: This should really be a conversion to float
        // Scale elements in buffer
        PCM32_to_PCM24(out, nframes);

        // printf("%d %d\n", samples_to_read, samples_written);
        
        // Fill the rest with zeros if needed.
        if(samples_written < (int) nframes){
            memset(out, 0, sizeof (jack_default_audio_sample_t) * (nframes - samples_written));
        }
    }

    finish_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(finish_time - start_time);
    printf("Duration %dus\n", duration);
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
	    const char *server_name = "SpeechField";


    jack_options_t options = static_cast<jack_options_t>((int) JackNoStartServer \
			      | (int) JackUseExactName \
			     | (int) JackServerName);


    jack_status_t status;

    jack_client = jack_client_open(client_name, options, &status, server_name);

    block_size = jack_get_buffer_size(jack_client);
    sampling_rate = jack_get_sample_rate(jack_client);

    bool realtime = jack_is_realtime(jack_client);
    
    fprintf(stderr, "[AudioManager] JACK Sampling rate: %d\n", sampling_rate);
    fprintf(stderr, "[AudioManager] JACK Block size: %d\n", block_size);
    fprintf(stderr, "[AudioManager] Realtime? %d\n", realtime);

    // Check if JACK Client open failed
	if (jack_client == NULL) {
		fprintf(stderr, "[AudioManager] jack_client_open() failed, status = 0x%2.0x\n", status);
		if (status & JackServerFailed) {
			fprintf(stderr, "Unable to connect to JACK server\n");
		}
        assert(0);
	}
    
    // Check if JACK Client name was reassigned
    if (status & JackNameNotUnique) {
		client_name = jack_get_client_name(jack_client);
		fprintf (stderr, "[AudioManager] unique name `%s' assigned\n", client_name);
	}
    
    // Check if JACK Server had been started
    if (status & JackServerStarted) {
		fprintf(stderr, "[AudioManager] JACK server started\n");
	}

    // Discover physical ports
    query_physical_ports();

    // Register callbacks
    jack_set_process_callback(jack_client, jack_callback_process, (void*) this);

    is_io_monitoring = false;
}

void AudioManager::query_physical_ports(){
    // Get physical input ports (microphones)
    mic_ports = jack_get_ports(jack_client, NULL, NULL, JackPortIsPhysical|JackPortIsOutput);
    this->num_physical_input_channels = 0;
    while(mic_ports[this->num_physical_input_channels] != NULL){
        this->num_physical_input_channels++;
    }

    // Get physical output ports (speakers)
    spk_ports = jack_get_ports(jack_client, NULL, NULL, JackPortIsPhysical|JackPortIsInput);
    this->num_physical_output_channels = 0;
    while(mic_ports[this->num_physical_output_channels] != NULL){
        this->num_physical_output_channels++;
    }

    printf("Num physical inputs: %d\n", this->num_physical_input_channels);
    printf("Num physical outputs: %d\n", this->num_physical_output_channels);
}

AudioManager::AudioManager(int num_input_channels, int num_output_channels): AudioManager(){
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
    for(int i = 0; i < std::max(num_input_ports, this->num_physical_input_channels); i++){
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

    // Activate client before connecting
    jack_activate(jack_client);
}

void AudioManager::start_io_monitoring(){
    is_io_monitoring = true; 
}

void AudioManager::stop_io_monitoring(){
    is_io_monitoring = false; 
}

void AudioManager::start(){
    // TODO: Check if properly configured
    
    // Connect client (virtual) ports to physical ports
    for(int i = 0; i < (int) this->input_port_order.size(); i++){
        int physical_port_id = this->input_port_order[i];
        if(jack_connect(jack_client, mic_ports[physical_port_id], jack_port_name (input_ports[i]))){
            fprintf(stderr, "ERROR DURING MIC CONNECTION\n");
			fprintf (stderr, "Cannot connect input ports %s and %s\n",
                jack_port_name ((const jack_port_t*) mic_ports[physical_port_id]),
                jack_port_name (input_ports[i]));
		}
	}

    for(int i = 0; i < (int) this->output_port_order.size(); i++){
        int virtual_port_id = this->output_port_order[i];

		if(jack_connect(jack_client, jack_port_name (output_ports[virtual_port_id]), spk_ports[i])){
            fprintf(stderr, "ERROR DURING SPK CONNECTION\n");
			fprintf (stderr, "Cannot connect input ports %s and %s\n",
                jack_port_name ((const jack_port_t*) spk_ports[i]),
                jack_port_name (output_ports[virtual_port_id]));
		}
	}
}

void AudioManager::stop(){
    // Disconnect all ports
    for(int i = 0; i < (int) this->input_port_order.size(); i++){
        int physical_port_id = this->input_port_order[i];
        
        if(jack_disconnect(jack_client, mic_ports[physical_port_id], jack_port_name (input_ports[i]))){
            fprintf (stderr, "Cannot disconnect input ports %s and %s\n",
                jack_port_name ((const jack_port_t*) spk_ports[physical_port_id]),
                jack_port_name (output_ports[i]));
        }
    }
    for(int i = 0; i < (int) this->output_port_order.size(); i++){
        int virtual_port_id = this->output_port_order[i];
        
        if(jack_disconnect(jack_client, jack_port_name (output_ports[virtual_port_id]), spk_ports[i])){
            fprintf (stderr, "Cannot disconnect input ports %s and %s\n",
                jack_port_name ((const jack_port_t*) spk_ports[i]),
                jack_port_name (output_ports[virtual_port_id]));
        }
    }

    // Clear buffers
    for(int i = 0; i < this->get_num_output_channels(); i++){
        rb_reset(&output_buffers[i]);
    }

}

AudioManager::~AudioManager(){
    jack_deactivate(jack_client);
    jack_client_close(jack_client);
    free(mic_ports);
    free(spk_ports);
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

void AudioManager::set_target_sampling_rate(int target_sampling_rate){
    for (auto r : input_resamplers){
        delete r;
    }

    for (auto r : output_resamplers){
        delete r;
    }
    
    input_resamplers.clear();
    output_resamplers.clear();

    this->target_sampling_rate = target_sampling_rate;

    for(int i = 0; i < (int) this->input_ports.size(); i++){
        ResamplerWrapper* r = new ResamplerWrapper(this->sampling_rate, target_sampling_rate, 1);
        input_resamplers.push_back(r);
    }

    for(int i = 0; i < (int) this->output_ports.size(); i++){
        ResamplerWrapper* r = new ResamplerWrapper(target_sampling_rate, this->sampling_rate, 1);
        output_resamplers.push_back(r);
    }

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

		// printf("channel %d\n", channel_index);

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
