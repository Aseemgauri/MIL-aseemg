#ifndef __UTILS_H__
#define __UTILS_H__

#include <string>
#include <vector>


// Return sampling rate
int read_wavfile(std::string path, std::vector<std::vector<float>> &audio);


void write_wavfile(std::string path, const std::vector<std::vector<float>> &audio, int sr);


#endif