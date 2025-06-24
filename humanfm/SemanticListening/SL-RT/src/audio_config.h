#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <cstdint>

// Audio processing parameters
#define CHUNK_SIZE 96     // Size of each audio chunk
#define LB_SIZE 96       // Lookback size
#define LF_SIZE 64        // Look forward size
#define NFFT (LB_SIZE + CHUNK_SIZE + LF_SIZE)  // Total window size
#define FFT_OUT_SIZE (NFFT/2 + 1)  // Size of FFT output (real + imaginary)
#define BUFFER_SIZE (NFFT - LB_SIZE)  // Size of synthesis buffers

// ISTFT parameters
#define ISTFT_OUTPUT_SIZE (CHUNK_SIZE + LF_SIZE)
#define ISTFT_LOOKBACK_BUFFERS (1 + ((ISTFT_OUTPUT_SIZE - 1) / CHUNK_SIZE))

// Memory alignment for SIMD operations
#define MEMORY_ALIGNMENT 32

// Window buffer declarations (only ones actually used in SL-RT)
extern double g_analysisWindow[NFFT];  // Analysis window
extern double g_synthesisWindow[ISTFT_OUTPUT_SIZE]; // Synthesis window

// Function declarations
void initializeWindows();

#endif // AUDIO_CONFIG_H 