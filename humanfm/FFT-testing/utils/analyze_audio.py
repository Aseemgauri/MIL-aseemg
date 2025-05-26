#!/usr/bin/env python3

import numpy as np
import wave
import os
import matplotlib.pyplot as plt

def read_wav_file(filepath):
    """Read a WAV file and return the audio data as numpy array"""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Read audio data
            audio_data = wav_file.readframes(frames)
            
            # Convert to numpy array
            if sample_width == 1:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                audio_array = audio_array.astype(np.float32) - 128
            elif sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32)
            elif sample_width == 4:
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                audio_array = audio_array.astype(np.float32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            return audio_array, sample_rate, channels, sample_width
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None, None

def analyze_audio_stats(audio_data, filename):
    """Analyze audio statistics"""
    if audio_data is None or len(audio_data) == 0:
        return None
    
    stats = {
        'filename': filename,
        'length': len(audio_data),
        'min': np.min(audio_data),
        'max': np.max(audio_data),
        'mean': np.mean(audio_data),
        'std': np.std(audio_data),
        'rms': np.sqrt(np.mean(audio_data**2)),
        'peak_amplitude': np.max(np.abs(audio_data)),
        'zero_samples': np.sum(audio_data == 0),
        'non_zero_samples': np.sum(audio_data != 0),
        'dynamic_range_db': None
    }
    
    # Calculate dynamic range in dB
    if stats['peak_amplitude'] > 0:
        # Find minimum non-zero amplitude
        non_zero_data = audio_data[audio_data != 0]
        if len(non_zero_data) > 0:
            min_non_zero = np.min(np.abs(non_zero_data))
            if min_non_zero > 0:
                stats['dynamic_range_db'] = 20 * np.log10(stats['peak_amplitude'] / min_non_zero)
    
    return stats

def compare_directories():
    """Compare audio files between groundtruth and output directories"""
    groundtruth_dir = "groundtruth"
    output_dir = "output"
    
    print("=== AUDIO FILE ANALYSIS ===\n")
    
    # Get file lists
    gt_files = sorted([f for f in os.listdir(groundtruth_dir) if f.endswith('.wav')])
    out_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    
    print(f"Groundtruth files: {gt_files}")
    print(f"Output files: {out_files}")
    print()
    
    # Analyze each class
    for i in range(5):
        gt_file = f"class_{i}.wav"
        out_file = f"output_class{i}.wav"
        
        print(f"=== CLASS {i} ANALYSIS ===")
        
        # Read groundtruth file
        gt_path = os.path.join(groundtruth_dir, gt_file)
        if os.path.exists(gt_path):
            gt_audio, gt_sr, gt_ch, gt_sw = read_wav_file(gt_path)
            gt_stats = analyze_audio_stats(gt_audio, gt_file)
        else:
            print(f"Groundtruth file not found: {gt_path}")
            continue
        
        # Read output file
        out_path = os.path.join(output_dir, out_file)
        if os.path.exists(out_path):
            out_audio, out_sr, out_ch, out_sw = read_wav_file(out_path)
            out_stats = analyze_audio_stats(out_audio, out_file)
        else:
            print(f"Output file not found: {out_path}")
            continue
        
        if gt_stats and out_stats:
            print(f"Groundtruth ({gt_file}):")
            print(f"  Sample rate: {gt_sr} Hz, Channels: {gt_ch}, Sample width: {gt_sw} bytes")
            print(f"  Length: {gt_stats['length']} samples")
            print(f"  Range: [{gt_stats['min']:.3f}, {gt_stats['max']:.3f}]")
            print(f"  Mean: {gt_stats['mean']:.6f}, RMS: {gt_stats['rms']:.6f}")
            print(f"  Peak amplitude: {gt_stats['peak_amplitude']:.3f}")
            print(f"  Non-zero samples: {gt_stats['non_zero_samples']}/{gt_stats['length']} ({100*gt_stats['non_zero_samples']/gt_stats['length']:.1f}%)")
            if gt_stats['dynamic_range_db']:
                print(f"  Dynamic range: {gt_stats['dynamic_range_db']:.1f} dB")
            
            print(f"\nOutput ({out_file}):")
            print(f"  Sample rate: {out_sr} Hz, Channels: {out_ch}, Sample width: {out_sw} bytes")
            print(f"  Length: {out_stats['length']} samples")
            print(f"  Range: [{out_stats['min']:.3f}, {out_stats['max']:.3f}]")
            print(f"  Mean: {out_stats['mean']:.6f}, RMS: {out_stats['rms']:.6f}")
            print(f"  Peak amplitude: {out_stats['peak_amplitude']:.3f}")
            print(f"  Non-zero samples: {out_stats['non_zero_samples']}/{out_stats['length']} ({100*out_stats['non_zero_samples']/out_stats['length']:.1f}%)")
            if out_stats['dynamic_range_db']:
                print(f"  Dynamic range: {out_stats['dynamic_range_db']:.1f} dB")
            
            # Calculate amplitude ratio
            if gt_stats['peak_amplitude'] > 0 and out_stats['peak_amplitude'] > 0:
                amplitude_ratio = gt_stats['peak_amplitude'] / out_stats['peak_amplitude']
                amplitude_ratio_db = 20 * np.log10(amplitude_ratio)
                print(f"\nAMPLITUDE COMPARISON:")
                print(f"  Groundtruth peak: {gt_stats['peak_amplitude']:.3f}")
                print(f"  Output peak: {out_stats['peak_amplitude']:.3f}")
                print(f"  Ratio (GT/Output): {amplitude_ratio:.1f}x")
                print(f"  Ratio in dB: {amplitude_ratio_db:.1f} dB")
                print(f"  Suggested amplification: {amplitude_ratio:.1f}x")
            elif gt_stats['peak_amplitude'] > 0:
                print(f"\nAMPLITUDE COMPARISON:")
                print(f"  Groundtruth peak: {gt_stats['peak_amplitude']:.3f}")
                print(f"  Output peak: {out_stats['peak_amplitude']:.3f} (essentially silent)")
                print(f"  Output needs significant amplification!")
            
            # RMS comparison
            if gt_stats['rms'] > 0 and out_stats['rms'] > 0:
                rms_ratio = gt_stats['rms'] / out_stats['rms']
                rms_ratio_db = 20 * np.log10(rms_ratio)
                print(f"\nRMS COMPARISON:")
                print(f"  Groundtruth RMS: {gt_stats['rms']:.6f}")
                print(f"  Output RMS: {out_stats['rms']:.6f}")
                print(f"  RMS Ratio: {rms_ratio:.1f}x ({rms_ratio_db:.1f} dB)")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    compare_directories() 