#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "utils.h"
#include "SpeechField.h"

#include <chrono>
using namespace std::chrono;


#include "test.h"

void test(std::string model_path){
	bool success;
	
	success = Test::test_runtime(model_path);
	if(success){
		printf("[SUCCESS] Runtime test\n");
	}else{
		printf("[FAILED] Runtime test\n");
	}

	success = Test::test_extraction_model(model_path, "test_data/replication_test");
	if(success){
		printf("[SUCCESS] Model correctness test\n");
	}else{
		printf("[FAILED] Model correctness test\n");
	}

	success = Test::test_streaming_correctness(model_path, "test_data/streaming_test");
	if(success){
		printf("[SUCCESS] Streaming correctness test\n");
	}else{
		printf("[FAILED] Streaming correctness test\n");
	}

	success = Test::test_resampling("test_data/resampling_test/speech2.wav",
	 								"test_data/resampling_test/downsampled.wav",
	 								"test_data/resampling_test/upsampled.wav");
	if(success){
		printf("[SUCCESS] Resampling test\n");
	}else{
		printf("[FAILED] Resampling test\n");
	}
	
	printf("RUNNING AUDIO MANAGER TEST\n");
	success = Test::test_audio_manager();
	if(success){
		printf("[SUCCESS] Audio Manager test\n");
	}else{
		printf("[FAILED] Audio Manager test\n");
	}
}

int main (int argc, char *argv[]){
	if(argc < 2){
		printf("\n\tUSAGE: ./SpeechField <model_path>\n\n");
		return -1;
	}

	std::string tse_model_path = argv[1];

	test(tse_model_path);

	// SpeechField::run(tse_model_path);
	
	return 0;
}
