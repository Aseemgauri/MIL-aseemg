import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

input_device = 'Logi USB Headset'
output_device = 'Logi USB Headset'
samplerate = 44100
channels = 1
blocksize = 128
latency = 'low'
dtype = 'float32'
subtype = 'PCM_16'



def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata


try:
    with sd.Stream(device=(input_device, output_device),
                   samplerate=samplerate, blocksize=blocksize,
                   dtype=dtype, latency=latency,
                   channels=channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    exit('')
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))
