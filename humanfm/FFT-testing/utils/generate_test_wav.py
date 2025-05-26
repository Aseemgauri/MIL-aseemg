import numpy as np
import wave

# Parameters
sample_rate = 16000  # 16 kHz
duration = 5  # seconds
frequencies = [440, 880, 1760]  # Hz
amplitude = 0.3  # range [-1.0, 1.0]

# Time array
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Generate mixed sine wave
mono_signal = sum(amplitude * np.sin(2 * np.pi * f * t) for f in frequencies)

# Normalize to 16-bit PCM range
mono_signal_int16 = np.int16(mono_signal * 32767)

# Duplicate for stereo: interleave left and right channels (same signal)
stereo_signal = np.column_stack((mono_signal_int16, mono_signal_int16)).flatten()

# Write to WAV file
with wave.open("test_sine_mix_stereo.wav", "w") as wav_file:
    wav_file.setnchannels(2)  # stereo
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(stereo_signal.tobytes())

print("Stereo WAV file 'test_sine_mix_stereo.wav' generated.")
