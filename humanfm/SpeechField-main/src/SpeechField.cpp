#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <unistd.h>

#include "SpeechField.h"
#include "SFModel.h"
#include "AudioManager.h"
#include "utils.h"


using namespace std::chrono;

#define SR 24000

enum state {
    STATE_OFF,
    STATE_PASSTHROUGH,
    STATE_ACTIVATED,
    STATE_CALIBRATING
};

static volatile state current_state = STATE_OFF;

void update_state(state target){
    if(current_state != STATE_OFF){
        current_state = target;
    }
}

void* stdin_listen(void* args){
    (void) args;

    std::string command;
    while(true){
        std::cout << "Enter command [passthrough/activate/quit]: ";
        std::cin >> command;

        if(command == "passthrough"){
            update_state(STATE_PASSTHROUGH);
            std::cout << "Pass-through on" << std::endl;
        }else if(command == "activate"){
            update_state(STATE_ACTIVATED);
        }else if(command == "quit"){
            update_state(STATE_OFF);
            std::cout << "Goodbye." << std::endl;
            break;
        }else{
            std::cout << "Command " << command << " undefined. Please try again." << std::endl;
        }
    }
    return NULL;
}


void SpeechField::run(const std::string &model_path){
    
    SFModel nn = SFModel(model_path);
    printf("[main] Model Initialized!\n");
    
    nn.infer(); // Cold start
    printf("[main] Cold start finished!\n");
    
    // Read model params
    int chunk_length = nn.get_chunk_length();
    int pad_length = nn.get_pad_length();
    int num_input_channels = nn.get_num_input_channels();
    int num_output_channels = nn.get_num_output_channels();
    
    
    AudioManager audio_manager = AudioManager(num_input_channels, num_output_channels);
    printf("[main] Audio Manager Initialized!\n");
    
    // Receive and put 16kHz audio
    audio_manager.set_target_sampling_rate(SR);

    // FOR BEST RESULTS, ENSURE THAT input_mixture_samples IS A MULTIPLE OF JACK SAMPLES PER FRAME
    
    // out buf
    float *out_buf = new float[chunk_length * num_input_channels];

    current_state = STATE_PASSTHROUGH;
	
    // Calibrate channels
    auto input_channel_order = audio_manager.calibrate_channels(num_input_channels);

    // Calibrate channels
    //audio_manager.calibrate_channels(2);

    // return;

    // Output channel order = same channel to both channels
    std::vector<int> output_channel_order = {0, 0, 0, 0, 0, 0, 0, 0};
    audio_manager.set_output_channel_order(output_channel_order);

    //std::vector<int> input_channel_order = {0, 1, 2, 3, 4, 5};
    //audio_manager.set_input_channel_order(input_channel_order);
    
    // Initialize internal buffer
    int input_mixture_samples = chunk_length + pad_length;
    float *buf = new float[2 * chunk_length * num_input_channels];

    audio_manager.configure_input_buffers(buf, chunk_length);

    // Run thread to respond to user input
    pthread_t stdin_listen_thread_id;
    pthread_create(&stdin_listen_thread_id, NULL, &stdin_listen, NULL);
    
    printf("[main] Input ports: %d\t Output ports: %d\n", 
		    audio_manager.get_num_input_channels(),
		    audio_manager.get_num_output_channels());
    printf("[main] Num output channels: %d\n", num_output_channels);

    // Allocate memory for monitoring
    std::vector<int> input_monitor_mask = {1,0,0,0,0,0};
    std::vector<int> output_monitor_mask = {1};
    audio_manager.allocate_monitoring_buffers(2 * 1440000, input_monitor_mask, output_monitor_mask);
    
    audio_manager.start();
    audio_manager.start_io_monitoring();
    
    auto current_time = high_resolution_clock::now();
    auto start_time = high_resolution_clock::now();
    std::vector<float> times;

    int j;
    
    // while(0){
    while(current_state != STATE_OFF){
        const float *released_mic_buf = audio_manager.wait_for_input_buffer();
        current_time = high_resolution_clock::now();
        times.push_back(duration_cast<microseconds>(current_time - start_time).count());
        // std::cout << "RELEASED " << times.back() << " \n" << std::endl;
        switch(current_state){
            case STATE_PASSTHROUGH:
            {
                for(int c = 0; c < num_output_channels; c++){
                    audio_manager.write_audio(released_mic_buf + c * chunk_length,
                                              chunk_length,
                                              c);
                }
            }
            break;
            
            case STATE_ACTIVATED: 
            {
                // Potential (minor) speedup:
                // Read passthrough_buffer_size samples at a time
                // And model gradually
                
                // Audio processing start
                nn.feed_audio_chunk(released_mic_buf, num_input_channels, chunk_length);

                auto start = high_resolution_clock::now();
                
                // Run inference
                nn.infer();
		//usleep(7000);

                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start);
                if(duration.count() >= 8000){
                    printf("Potential chunk drop! Time taken: %d\n", duration);
                }

                // Get model filtered output
                // float* output_data = nn.current_frame;
		Ort::Value &fx_output = nn.get_output("filtered_output");
                float* output_data = (float*) fx_output.GetTensorMutableRawData();
                
		// Audio processing end
                
                for(int c = 0; c < num_output_channels; c++){
		    for(j = 0; j < chunk_length; j++) output_data[j] *= 20;
		    //audio_manager.write_audio(released_mic_buf + c * chunk_length,
		    audio_manager.write_audio(output_data + c * chunk_length,
                                              chunk_length,
                                              c);
                }
            }
            break;

            case STATE_OFF:
            break;
            
            default:
                std::cout << "STATE UNDEFINED" << std::endl;
            break;
        }
    }
    audio_manager.stop_io_monitoring();
    audio_manager.stop();
    
    float avg = 0;
    for(int i = 1; i < times.size(); i++){
         float diff = times[i]  - times[i-1];
    //     std::cout << diff << std::endl;
         avg += diff / (times.size() - 1);
    }
    std::cout << "Average loop time " << avg/1000 << "ms."  << std::endl;
    //printf("AUDIO SIZE (%d)", audio_manager.input_debug_v.size());
    std::vector<std::vector<float>> input_audio;// = std::vector<std::vector<float>>(input_channel_order.size());
    audio_manager.retrieve_monitored_input(input_audio);
    // for(int i = 0; i < input_channel_order.size(); i++){
	// for(int j = 0; j < audio_manager.input_debug_v[i].size(); j++){
  	//     input_audio[i].push_back(audio_manager.input_debug_v[i][j]);
	// }
    // }
    write_wavfile("input_stream.wav", input_audio, SR);

    std::vector<std::vector<float>> output_audio;// = audio_manager.output_debug_v;
    audio_manager.retrieve_monitored_output(output_audio);
	write_wavfile("output_stream.wav", output_audio, SR);

    free(buf);

    pthread_join(stdin_listen_thread_id, NULL);
  
    pthread_exit(NULL);
}
