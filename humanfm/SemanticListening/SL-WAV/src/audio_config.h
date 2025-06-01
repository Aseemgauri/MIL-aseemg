#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <cstdint>

// Audio processing parameters
#define CHUNK_SIZE 96     // Size of each audio chunk
#define LB_SIZE 96       // Lookback size
#define LF_SIZE 64        // Look forward size
#define WINDOW_SIZE (LB_SIZE + CHUNK_SIZE + LF_SIZE)  // Total window size
#define FFT_SIZE (WINDOW_SIZE/2 + 1)  // Size of FFT output (real + imaginary)
#define BUFFER_SIZE (WINDOW_SIZE - LB_SIZE)  // Size of synthesis buffers

// ISTFT parameters
#define ISTFT_OUTPUT_SIZE (CHUNK_SIZE + LF_SIZE)
#define ISTFT_LOOKBACK_BUFFERS (1 + ((ISTFT_OUTPUT_SIZE - 1) / CHUNK_SIZE))

// Memory alignment for SIMD operations
#define MEMORY_ALIGNMENT 32

// Static buffer declarations
extern double g_previousBuffer[BUFFER_SIZE];
extern double g_currentBuffer[BUFFER_SIZE];
extern double g_synthesisBuffer[WINDOW_SIZE];
extern double g_fftReal[FFT_SIZE];
extern double g_fftImag[FFT_SIZE];
extern double g_ifftResult[WINDOW_SIZE];
extern char g_windowBuffer[WINDOW_SIZE * sizeof(float)];
extern float g_chunkBuffer[CHUNK_SIZE];

// Precomputed window declarations
extern double g_analysisWindow[WINDOW_SIZE];  // Analysis window
extern double g_synthesisWindow[ISTFT_OUTPUT_SIZE]; // Synthesis window

// ISTFT context declarations
extern double g_istftContextBuffers[ISTFT_LOOKBACK_BUFFERS * ISTFT_OUTPUT_SIZE];
extern uint8_t g_lookbackBufIdx;

// Function declarations
void initializeWindows();
double* getIstftContextBuffer(int8_t lookbackIdx);

#endif // AUDIO_CONFIG_H 