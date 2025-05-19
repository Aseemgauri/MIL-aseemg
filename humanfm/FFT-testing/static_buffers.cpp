#include "audio_config.h"
#include <cmath>
#include <stdexcept>

// Define static buffers
double g_previousBuffer[BUFFER_SIZE] = {0};
double g_currentBuffer[BUFFER_SIZE] = {0};
double g_synthesisBuffer[WINDOW_SIZE] = {0};
double g_fftReal[FFT_SIZE] = {0};
double g_fftImag[FFT_SIZE] = {0};
double g_ifftResult[WINDOW_SIZE] = {0};
char g_windowBuffer[WINDOW_SIZE * sizeof(float)] = {0};
char g_chunkBuffer[CHUNK_SIZE * sizeof(float)] = {0};

// Define window buffers
double g_analysisWindow[WINDOW_SIZE] = {0};
double g_synthesisWindow[ISTFT_OUTPUT_SIZE] = {0};

// ISTFT context buffers
double g_istftContextBuffers[ISTFT_LOOKBACK_BUFFERS * ISTFT_OUTPUT_SIZE] = {0};
uint8_t g_lookbackBufIdx = 0;

// Get ISTFT context buffer for a specific lookback index
double* getIstftContextBuffer(int8_t lookbackIdx) {
    lookbackIdx = (ISTFT_LOOKBACK_BUFFERS + lookbackIdx) % ISTFT_LOOKBACK_BUFFERS;
    return &g_istftContextBuffers[lookbackIdx * ISTFT_OUTPUT_SIZE];
}

// Initialize windows with perfect reconstruction
void initializeWindows() {
    const double PI = 3.14159265358979323846;
    
    // Compute Hanning window for analysis
    for (int i = 0; i < WINDOW_SIZE; i++) {
        g_analysisWindow[i] = 0.5 * (1.0 - cos(2.0 * PI * i / (WINDOW_SIZE - 1)));
    }
    
    // Compute perfect synthesis window
    int oWS = ISTFT_OUTPUT_SIZE;  // Output window size
    int A = oWS;                  // Synthesis window size
    int B = CHUNK_SIZE;           // Chunk size
    int N = WINDOW_SIZE;          // FFT size
    
    if ((A % B) == 0) {
        // Case 1: Output window size is a multiple of chunk size
        for (int i = 0; i < A; i++) {
            double num = g_analysisWindow[N - A + i];
            double denom = 0;
            
            for (int k = 0; k < A/B; k++) {
                denom += g_analysisWindow[N - A + (i % B) + k * B] * 
                        g_analysisWindow[N - A + (i % B) + k * B];
            }
            
            g_synthesisWindow[i] = num / denom;
        }
    }
    else if (LF_SIZE < CHUNK_SIZE) {
        // Case 2: Lookahead is less than chunk size
        for (int i = 0; i < A; i++) {
            if (i >= LF_SIZE && i < oWS - LF_SIZE) {
                // Non-overlapping region
                g_synthesisWindow[i] = 1.0 / g_analysisWindow[LB_SIZE + i];
            }
            else {
                // Overlapping region
                double num = g_analysisWindow[LB_SIZE + i];
                double denom = 0;
                
                for (int k = 0; k < A/B + 1; k++) {
                    denom += g_analysisWindow[LB_SIZE + (i % B) + k * B] * 
                            g_analysisWindow[LB_SIZE + (i % B) + k * B];
                }
                
                g_synthesisWindow[i] = num / denom;
            }
        }
    }
    else {
        throw std::runtime_error("Invalid window configuration: Either make front pad a multiple of chunk size, or make chunk size > front pad");
    }
} 