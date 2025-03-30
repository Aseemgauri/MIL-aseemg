#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <filesystem>


#include "test.h"
#include "SFModel.h"
#include "cnpy.h"
#include "utils.h"
#include "Resampler.h"
#include "AudioManager.h"

namespace fs = std::filesystem;

using namespace std::chrono;


bool Test::test_runtime(const std::string &tse_model_path){

    SFModel tshear = SFModel(tse_model_path);
	printf("Initialized model\n");
	tshear.infer();

	tshear.reset_state();
	
	const int RUNS = 1000;
	float times[RUNS];
	std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
	for(int i = 0; i < RUNS; i++){
		start = high_resolution_clock::now();
		tshear.infer();
		stop = high_resolution_clock::now();
		times[i] = duration_cast<microseconds>(stop - start).count();
	}

	float mean_time = 0;
	for(int i = 0; i < RUNS; i++){
		mean_time += times[i];
	}
	mean_time /= RUNS;

	std::ofstream out("runtime_test_results.txt");
	for(int i = 0 ; i< RUNS; i++){
		out << std::setprecision(10) << std::fixed << times[i] << "\n";
	}

	printf("Ran inference\n");
	std::cout << "Time taken: " << mean_time << " us" << std::endl;
	printf("Test 1 over\n");
    return true;
}


/*
Check if streaming inference is implemented correctly.
*/
bool Test::test_streaming_correctness(const std::string &model_path,
									  const std::string &test_data_path){

    // Initialize model
	SFModel tshear = SFModel(model_path);

	std::map<std::string, cnpy::NpyArray> input_data;
	
	// Load numpy arrays
	fs::path path;
	path = fs::path(test_data_path) / fs::path("e2e_input_X.npy");
	cnpy::NpyArray e2e_X = cnpy::npy_load(path);
	
	path = fs::path(test_data_path) / fs::path("e2e_output_streaming.npy");
	cnpy::NpyArray e2e_output_streaming = cnpy::npy_load(path);
	
	path = fs::path(test_data_path) / fs::path("e2e_output_full.npy");
	cnpy::NpyArray e2e_output_full = cnpy::npy_load(path);

	// Get input shape
	int num_samples = e2e_X.shape[2];
	int num_input_channels = e2e_X.shape[1];
	int num_output_channels = e2e_output_full.shape[1];
    
	// Get input parameters
	int chunk_size = tshear.get_chunk_length();
	int pad_size = tshear.get_pad_length();
	
	// Initialize vector to store output audio
	std::vector<std::vector<float>> result(num_output_channels);
	
	// Initialize array to current chunk
	float *current_chunk = new float[num_samples * chunk_size];

	tshear.reset_state();
	
	// Pre-fill the current frame with a number of samples equal to stride size
	int channel_idx;
	for(channel_idx = 0; channel_idx < num_input_channels; channel_idx++){
		memcpy(current_chunk + pad_size * channel_idx,
				e2e_X.data<float>() + num_samples * channel_idx,
				sizeof(float) * pad_size);
	}
	tshear.feed_audio_chunk(current_chunk, num_input_channels, pad_size);

	// Start at the beginning of the array
	// Process chunk_size samples at a time from all channels
	// Break if there aren't enough samples anymore	
	int t = pad_size;
	while(t + chunk_size <= num_samples){
		// Copy chunk_size samples from all channels into current chunk
		for(channel_idx = 0; channel_idx < num_input_channels; channel_idx++){
			memcpy(current_chunk + chunk_size * channel_idx,
				   e2e_X.data<float>() + num_samples * channel_idx + t,
				   sizeof(float) * chunk_size);
		}
		tshear.feed_audio_chunk(current_chunk, num_input_channels, chunk_size);

		// Run inference
		tshear.infer();

		// Get model filtered output
		Ort::Value &fx_output = tshear.get_output("filtered_output");
		const float* output_data = (const float*) fx_output.GetTensorRawData();
		
		// Insert output into the results vector at the right channel
		for(int i = 0; i < num_output_channels; i++){
			const float *copy_start = output_data + chunk_size * i;
			result[i].insert(result[i].end(), copy_start, copy_start + chunk_size);
		}

		t += chunk_size;
	}

	// Deallocate memory used to copy chunks
	delete [] current_chunk;
	
	// Compare outputs
	const float* expected_output_streaming = e2e_output_streaming.data<float>();
	const float* expected_output_full = e2e_output_full.data<float>();
	int num_output_samples = e2e_output_streaming.shape[2];
	
	// Print sample by sample.
	bool success = true;
	float EPS = 1e-3;
	for(int i = 0; i < num_output_samples; i++){
		for(int c = 0; c < num_output_channels; c++){
			int index = c * num_output_samples + i;
			// printf("%.02f %.02f %.02f\n",
			// 		result[c][i],
			// 		expected_output_full[index],
			// 		expected_output_streaming[index]); // <--- ALL THREE MUST BE EQUAL
			if(fabs(result[c][i] - expected_output_streaming[index]) > EPS){
				success = false;
				break;
			}

			if(fabs(result[c][i] - expected_output_full[index]) > EPS){
				success = false;
				break;
			}
		}
	}

	return success;
}

/*
Check if inference is implemented correctly.
*/
bool Test::test_extraction_model(const std::string &model_path,
						 	 	 const std::string &test_data_path){
	printf("Second test\n");
    // Initialize model
	SFModel model = SFModel(model_path);

    fs::path input_names_path = fs::path(test_data_path) / fs::path("input_names.txt");
	std::ifstream input_names_in(input_names_path.c_str());

	// Load numpy arrays
	while (!input_names_in.eof()){
		std::string input_name;
		input_names_in >> input_name;

		fs::path array_path = fs::path(test_data_path) / fs::path(input_name + ".npy");
		cnpy::NpyArray input_array = cnpy::npy_load(array_path);

		model.set_input(input_name, input_array.data<float>());
	}

	model.infer();
	
	fs::path output_names_path = fs::path(test_data_path) / fs::path("output_names.txt");
	std::ifstream output_names_in(output_names_path.c_str());
	
	float EPS = 1e-3;
	bool success = true;
	// Load numpy arrays
	while (!output_names_in.eof()){
		std::string output_name;
		output_names_in >> output_name;

		Ort::Value &outval = model.get_output(output_name);
		const float* output_data = (const float*) outval.GetTensorRawData();

		fs::path array_path = fs::path(test_data_path) / fs::path(output_name + ".npy");
		cnpy::NpyArray input_array = cnpy::npy_load(array_path);

		auto shape = model.get_output_shape(output_name);
		for(int i = 0; i < shape[0]; i++){
			for(int j = 0; j < shape[1]; j++){
				for(int k = 0; k < shape[2]; k++){
					
					int index = k + j * shape[2] + i * (shape[1] * shape[2]);

					float exp = input_array.data<float>()[index];
					float actual = output_data[index];

					// if(j == 0 && output_name == "filtered_output"){
					// 	std::cout << exp << " " << actual << std::endl;
					// }

					if(fabs(exp - actual) > EPS){
						success = false;
						break;
					}
				}
			}
		}
		
	}

	return success;
	return true;
}

bool Test::test_resampling(const std::string &input_audio_path,
					 	   const std::string &downsampled_audio_path,
						   const std::string &upsampled_audio_path){
	bool success = true;

	// Read input wavefile
	std::vector<std::vector<float>> audio;
	int sr = read_wavfile(input_audio_path, audio);
	int target_sr = sr / 2;
	
	int src_rate = sr;
	int dst_rate = target_sr;
	int num_ch = audio.size();
	int num_frames = audio[0].size();

	// Setup downsampler
	ResamplerWrapper downsampler = ResamplerWrapper(src_rate, dst_rate, 1);
	int num_output_frames = downsampler.calculate_output_frames(num_frames);
	
	float *audio_data = new float[num_ch * num_frames];
	float *audio_data_resampled = new float[num_ch * num_output_frames];
	
	// Copy data to float array
	for(int i = 0; i < num_ch; i++){
		memcpy(audio_data + i * num_frames, audio[i].data(), audio[i].size() * sizeof(float));
	}

	// Downsample
	// downsampler.resample(audio_data_resampled, audio_data, num_frames);

	// Chunk-wise downsample
	int chunk_size = 256;
	
	// Pointers to original and downsampled data
	float* orig_ptr = audio_data;
	float* downsample_ptr = audio_data_resampled;
	int input_frames, output_frames, frames_written = 0;
	int num_chunks = 0;
	
	auto start = high_resolution_clock::now();
	while(frames_written < num_frames){
		// Get max number of frames to read from input
		input_frames = std::min(chunk_size, num_frames - frames_written);
		
		// Resample
		output_frames = downsampler.resample(downsample_ptr, orig_ptr, input_frames);
		
		orig_ptr += input_frames;
		downsample_ptr += output_frames;
		frames_written += input_frames; 
		num_chunks++;
	}
	auto end = high_resolution_clock::now();

	int downsampling_time = duration_cast<microseconds>(end - start).count();
	std::cout << "Time taken to resample " << num_chunks << " chunks: " << downsampling_time << " us" << std::endl;

	// Copy data to vector
	std::vector<std::vector<float> > out(num_ch);
	for(int i = 0; i < num_ch; i++){
		for(int j = 0; j < num_output_frames; j++){
			out[i].push_back(audio_data_resampled[i * num_output_frames + j]);
		}
	}
	// Save downsampled
	write_wavfile(downsampled_audio_path, out, target_sr);
	
	// Create upsampler
	ResamplerWrapper upsampler = ResamplerWrapper(dst_rate, src_rate, 1);
	upsampler.resample(audio_data, audio_data_resampled, num_output_frames);

	out.clear();
	out = std::vector<std::vector<float>>(num_ch);
	for(int i = 0; i < num_ch; i++){
		for(int j = 0; j < num_frames; j++){
			out[i].push_back(audio_data[i * num_frames + j]);
		}
	}
	// Save upsampled
	write_wavfile(upsampled_audio_path, out, sr);

	delete [] audio_data;
	delete [] audio_data_resampled;

	return success;
}

void update_channel_order(AudioManager* audio_manager){
    // For 2 inputs:
    // input_00 -> physical port 0 (real mic)
    // input_01 -> physical port 1 (dummy mic)
    std::vector<int> input_channel_order = {0, 1};
    audio_manager->set_input_channel_order(input_channel_order);
}

bool Test::test_audio_manager(){
	int chunk_length = 192;
	int num_input_channels = 2;

	printf("Initializing Audio Manager\n");
	AudioManager audio_manager = AudioManager(num_input_channels, 2);
	printf("Initialized Audio Manager\n");

	std::vector<int> output_channel_order = {0, 0};
	audio_manager.set_output_channel_order(output_channel_order);
	
	// Configure input buffers first
	float *buf = new float[2 * chunk_length * num_input_channels];
	printf("Configuring input buffers...\n");
	audio_manager.configure_input_buffers(buf, chunk_length);
	printf("Input buffers configured\n");
	
	printf("Starting Audio Manager...\n");
	audio_manager.start();
	printf("Audio Manager Started!\n");

	// Sleep for 5 seconds
	printf("Sleeping for 5 seconds...\n");
	usleep(1000 * 1000 * 5);

	printf("Stopping Audio Manager...\n");
	audio_manager.stop();
	printf("Audio Manager Stopped!\n");
	
	// Give JACK time to clean up
	printf("Waiting for JACK to clean up...\n");
	usleep(500 * 1000); // 500ms delay
	
	update_channel_order(&audio_manager);
	
	auto input_port_order = audio_manager.get_input_channel_order();
	printf("Input port order: ");
	for(auto &idx : input_port_order){
		printf("%d ", idx);
	}printf("\n");
	
	printf("Pause for a second\n");
	usleep(1000 * 1000 * 1);
	
	printf("Starting Audio Manager again...\n");
	audio_manager.start();
	printf("Audio Manager Started!\n");
	
	// Sleep again
	printf("Sleeping for 5 seconds...\n");
	usleep(1000 * 1000 * 5);

	printf("Stopping Audio Manager...\n");
	audio_manager.stop();
	printf("Audio Manager Stopped!\n");	

	// Give JACK time to clean up before deleting buffers
	printf("Waiting for JACK to clean up...\n");

	audio_manager.end();
	delete[] buf;
	return true;
}
