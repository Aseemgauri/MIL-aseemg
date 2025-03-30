#ifndef __AUDIO_MANAGER_H__
#define __AUDIO_MANAGER_H__

#include <jack/jack.h>
#include <vector>
#include <vector>
#include <mutex>

#include "ringbuffer.h"

#define JACK_CLIENT_NAME "SpeechField"
#define AUDIO_OUTPUT_BUFFER_SIZE 1000000


class AudioManager{
    public:
        AudioManager();
        AudioManager(int num_input_channels, int num_output_channels);
        ~AudioManager();

        void start();
        void stop();
        void end();  // New method for complete cleanup

        /*
        Specifies the point in memory where double-buffered data from the microphones
        will be written to. It is expected that there is enough memory for 2 * input_chunk size floats.
        */
        void configure_input_buffers(float* double_buffer, int samples_per_channel);
        
        int get_num_input_channels();
        int get_num_output_channels();
        
        int get_block_size();

        // Deprecated: Sampling rate is now fixed at 44100Hz
        void set_target_sampling_rate(int target_sampling_rate) {
            if (target_sampling_rate != 44100) {
                fprintf(stderr, "Warning: set_target_sampling_rate() is deprecated. Only 44100Hz is supported.\n");
            }
        }

        void setup_ports(int num_input_ports, int num_output_ports);

        const float* wait_for_input_buffer();
        
        void write_audio(const float *data, int num_samples, int channel_index);

        friend int jack_callback_process(jack_nframes_t nframes, void *arg);
        friend int jack_callback_shutdown(jack_nframes_t nframes, void *arg);
        
        // I/O MONITORING
        // Functions to enable/disable I/O monitoring for debugging purposes
        void allocate_monitoring_buffers(int max_samples, std::vector<int> input_channel_mask, std::vector<int> output_channel_mask);
        void start_io_monitoring();
        void stop_io_monitoring();
        // Functions to retrive monitored audio
        void retrieve_monitored_input(std::vector<std::vector<float>> &dst);
        void retrieve_monitored_output(std::vector<std::vector<float>> &dst);

        // Do channel calibration
	std::vector<int> calibrate_channels(int num_channels);
        void reset_channel_order();
        void set_output_channel_order(const std::vector<int> order);
        void set_input_channel_order(const std::vector<int> order);
	std::vector<int> get_input_channel_order(void);
    
        std::vector<ringbuffer_t> input_debug_v;
        std::vector<ringbuffer_t> output_debug_v;
        ringbuffer_t* output_buffers;
        bool is_running;  // Flag to control audio processing
        bool is_io_monitoring;  // Flag to control audio monitoring
    private:
        bool input_data_available();
        void release_buffer();
        void advance_idx(int num_samples);
        
        // Finds and enumerates server input and output ports
        void query_physical_ports();

        // Returns pointer to the start of the i-th buffer in the input double buffer
        float* get_input_buffer_ptr(int double_buffer_idx);

        jack_client_t *jack_client;
        
        int sampling_rate;
        int block_size;
        int num_physical_input_channels;
        int num_physical_output_channels;
        
        // For inputs:
        // input_port_order[i] is the physical channel that the i-th input channel is sourced from
        // i.e. One physical channel can pass data to multiple input channels
        // Caveat: Some physical channels may not source any input channels
        std::vector<int> input_port_order;
        
        // For outputs:
        // output_port_order[i] is the output channel that the i-th physical channel sources from
        // i.e. One physical channel only receives samples from a single virtual output channel
        // Caveat: All physical channels must source from some input channel
        std::vector<int> output_port_order;

        // Arrays of virtual ports
        std::vector<jack_port_t*> input_ports;
        std::vector<jack_port_t*> output_ports;

        // Arrays of physical ports
        const char **mic_ports;
        const char **spk_ports;

        std::mutex input_buffer_access_lock;
        std::mutex input_buffer_ready;
        int current_input_buffer_idx;
        int current_input_buffer_step;
        int input_samples_per_buffer;
        int input_samples_per_channel;
        float *input_buffers;
};

#endif
