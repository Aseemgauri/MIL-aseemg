import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
import queue

input_device = 'Logi USB Headset'
output_device = 'Logi USB Headset'
samplerate = 44100
channels = 1
blocksize = 128
latency = 'low'
dtype = 'float32'
subtype = 'PCM_16'

q = queue.Queue()
queue_len = 128
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())
    if(q.qsize() >= queue_len):
        outdata[:] = q.get()
    else:
        outdata[:] = 0

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
