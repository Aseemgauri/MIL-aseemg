import sys
import os
import librosa
import soundfile as sf

def downsample_to_16k(input_file):
    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Output filename
    base, _ = os.path.splitext(input_file)
    output_file = f"{base}_5s_16k_float32.wav"

    # Load only the first 5 seconds
    max_duration = 5.0  # seconds
    audio, sr = librosa.load(input_file, sr=None, mono=True, duration=max_duration)

    # Resample to 16kHz
    target_sr = 16000
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Save using WAV format and FLOAT subtype to ensure 'data' chunk
    sf.write(output_file, audio_16k, samplerate=target_sr, format='WAV', subtype='FLOAT')

    print(f"Saved: {output_file} (5s, 16kHz, 32-bit float, 'data' chunk)")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python downsample_to_16k.py <input.wav>")
        sys.exit(1)

    input_path = sys.argv[1]
    downsample_to_16k(input_path)
