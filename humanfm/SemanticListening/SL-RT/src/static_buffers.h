#ifndef STATIC_BUFFERS_H
#define STATIC_BUFFERS_H

#include "audio_config.h"
#include <stdint.h>

// Static buffer declarations
extern double g_previousBuffer[BUFFER_SIZE];
extern double g_currentBuffer[BUFFER_SIZE];
extern double g_synthesisBuffer[WINDOW_SIZE];
extern double g_fftReal[FFT_SIZE];
extern double g_fftImag[FFT_SIZE];
extern double g_ifftResult[WINDOW_SIZE];
extern char g_windowBuffer[WINDOW_SIZE * sizeof(float)];
extern char g_chunkBuffer[CHUNK_SIZE * sizeof(float)];

// Window buffer declarations
extern double g_analysisWindow[WINDOW_SIZE];
extern double g_synthesisWindow[ISTFT_OUTPUT_SIZE];

// ISTFT context buffers
extern double g_istftContextBuffers[ISTFT_LOOKBACK_BUFFERS * ISTFT_OUTPUT_SIZE];
extern uint8_t g_lookbackBufIdx;

// Function declarations
double* getIstftContextBuffer(int8_t lookbackIdx);
void initializeWindows();

#endif // STATIC_BUFFERS_H 