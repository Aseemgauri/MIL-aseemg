#ifndef __TEST_H__
#define __TEST_H__

#include <string>

struct Test{
    Test();

    static bool test_runtime(const std::string &model_path);
    
    static bool test_streaming_correctness(const std::string &model_path,
                                           const std::string &test_data_path);

    static bool test_extraction_model(const std::string &model_path,
						              const std::string &test_data_path);

    static bool test_resampling(const std::string &input_audio_path,
                                const std::string &output_audio_path,
                                const std::string &upsampled_audio_path);

    static bool test_audio_manager();

};



#endif