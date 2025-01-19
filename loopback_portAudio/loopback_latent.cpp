#include <iostream>
#include <list>
#include "portaudio.h"

#define DUR_MILLIS          (5000)
#define NUM_CHANNELS        (2)
#define SAMPLE_RATE         (44100)
#define FRAMES_PER_BUFFER   (128)
#define SAMPLE_FORMAT       paFloat32
#define SAMPLE_SIZE         sizeof(float)
#define SAMPLE_SILENCE      (0.0f)

using namespace std;

PaError err;
PaStream *stream = NULL;

list<float> q;
int maxQSize = FRAMES_PER_BUFFER * NUM_CHANNELS * 128;
int queueSize = 0;
static int wireCallback( const void *inputBuffer,
        void *outputBuffer,
        unsigned long framesPerBuffer,
        const PaStreamCallbackTimeInfo *timeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData
        )
{
    (void) timeInfo; // Prevent unused variable warnings.
    (void) statusFlags;
    (void) userData;

    // Cast data to floats:
    float *out = (float*)outputBuffer;
    float *in = (float*)inputBuffer;
    unsigned long i;
    
    // Copy (wire) input to output:
    for( i = 0; i < framesPerBuffer*NUM_CHANNELS; i++ ) {
        q.push_back(in[i]);
        queueSize++;
    }
        
    if(queueSize >= maxQSize) {
        for( i = 0; i < framesPerBuffer*NUM_CHANNELS; i++ ) {
            out[i] = q.front();
            q.pop_front();
            queueSize--;
        }
    } else {
        for( i = 0; i < framesPerBuffer*NUM_CHANNELS; i++ ) {
            out[i] = 0;
        }
    }

    return paContinue;
}

void startStream() {
    PaStreamParameters inParam;
    PaStreamParameters outParam;
    const PaDeviceInfo* inputInfo;
    const PaDeviceInfo* outputInfo;

    inParam.device = Pa_GetDefaultInputDevice(); /* default input device */
    inputInfo = Pa_GetDeviceInfo( inParam.device );
    inParam.channelCount = NUM_CHANNELS;
    inParam.sampleFormat = SAMPLE_FORMAT;
    inParam.suggestedLatency = inputInfo->defaultLowInputLatency ;
    inParam.hostApiSpecificStreamInfo = NULL;

    outParam.device = Pa_GetDefaultOutputDevice(); /* default input device */
    outputInfo = Pa_GetDeviceInfo( outParam.device );
    outParam.channelCount = NUM_CHANNELS;
    outParam.sampleFormat = SAMPLE_FORMAT;
    outParam.suggestedLatency = outputInfo->defaultLowOutputLatency ;
    outParam.hostApiSpecificStreamInfo = NULL;

    // Open and start stream using wireCallback:
    err = Pa_OpenStream(
            &stream,
            &inParam,
            &outParam,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
            paClipOff,
            wireCallback,
            NULL
            );
    err = Pa_StartStream( stream );
    //Pa_Sleep(DUR_MILLIS); // let stream run for 5 seconds.

    // Cleanup:
    if (cin.get())
    {
        err = Pa_StopStream( stream );
        Pa_AbortStream( stream );
        Pa_CloseStream( stream );
    }
    
}

int main( int argc, char *argv[] ){
    err = Pa_Initialize();

    cout << "Press enter to quit" << endl;
    startStream();

    Pa_Terminate();

    return 0;
}
