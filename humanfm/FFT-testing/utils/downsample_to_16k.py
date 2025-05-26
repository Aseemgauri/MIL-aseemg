import sys
import os
import soundfile as sf
from scipy.signal import resample_poly

def main():
    if len(sys.argv) != 2:
        print("Usage: python resample_to_16k.py <input.wav>")
        sys.exit(1)

    input_filename = sys.argv[1]
    
    if not os.path.isfile(input_filename):
        print(f"Error: File '{input_filename}' does not exist.")
        sys.exit(1)

    # Target sample rate
    target_sr = 16000

    # Load input file
    data, original_sr = sf.read(input_filename)

    # Compute up/down ratio for resampling
    gcd = lambda a, b: a if b == 0 else gcd(b, a % b)
    factor = gcd(original_sr, target_sr)
    up = target_sr // factor
    down = original_sr // factor

    # Resample
    resampled_data = resample_poly(data, up, down)

    # Output filename
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_16k.wav"

    # Save resampled audio
    sf.write(output_filename, resampled_data, target_sr)
    print(f"Resampled '{input_filename}' from {original_sr} Hz to {target_sr} Hz → '{output_filename}'")

if __name__ == "__main__":
    main()
