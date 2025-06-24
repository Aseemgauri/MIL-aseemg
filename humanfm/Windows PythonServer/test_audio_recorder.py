#!/usr/bin/env python3
"""
Test Audio Recorder Server
Receives GStreamer audio stream from Orange Pi and saves to WAV file.
Type 'stop' to finish recording and save the file.
"""

import sys
import os
import threading
import time
from datetime import datetime
import wave
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Configuration - matches python_server.py
USE_BROADCAST = True  # Set to True for broadcast, False for multicast
ORANGEPI_IP = "192.168.2.218"  # Orange Pi IP address (for reference)

# Audio format constants
SAMPLE_RATE = 16000  # 16 kHz
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
TARGET_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH  # 1 second of audio = 32,000 bytes

# Global variables
buffered_audio = bytearray()
recording = True
total_samples_received = 0
start_time = None

print("🎙️  Audio Recording Test Server")
print("=" * 50)
print(f"📡 Orange Pi IP: {ORANGEPI_IP}")
print(f"🔊 Audio Format: {SAMPLE_RATE}Hz, {CHANNELS} channel, {SAMPLE_WIDTH*8}-bit")
print(f"💾 Will save to: recorded_audio_[timestamp].wav")
print("⌨️  Type 'stop' and press Enter to finish recording")
print("=" * 50)

# Initialize GStreamer
Gst.init(None)

# GStreamer pipeline - same as python_server.py
if USE_BROADCAST:
    # Broadcast mode - listen on all interfaces for broadcast packets
    pipeline_str = (
        'udpsrc port=5000 caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=96" '
        '! rtpopusdepay '
        '! decodebin '
        '! audioconvert '
        '! audioresample '
        '! audio/x-raw,rate=16000,format=S16LE,channels=1 '
        '! appsink name=sink emit-signals=true sync=false'
    )
    print("📡 Using BROADCAST mode - listening on port 5000")
else:
    # Multicast mode - join multicast group
    pipeline_str = (
        'udpsrc uri=udp://239.1.1.1:5000 caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=96" '
        '! rtpopusdepay '
        '! decodebin '
        '! audioconvert '
        '! audioresample '
        '! audio/x-raw,rate=16000,format=S16LE,channels=1 '
        '! appsink name=sink emit-signals=true sync=false'
    )
    print("📡 Using MULTICAST mode - listening on 239.1.1.1:5000")

# Build the pipeline
try:
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("sink")
    print("✅ GStreamer pipeline created successfully")
except Exception as e:
    print(f"❌ Failed to create GStreamer pipeline: {e}")
    sys.exit(1)

def format_duration(seconds):
    """Format duration in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def format_size(bytes_count):
    """Format bytes to human readable format"""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    else:
        return f"{bytes_count / (1024 * 1024):.1f} MB"

def on_new_sample(sink):
    """Callback function triggered when new audio buffer arrives"""
    global buffered_audio, total_samples_received, start_time
    
    if not recording:
        return Gst.FlowReturn.OK
    
    # Set start time on first sample
    if start_time is None:
        start_time = time.time()
        print("🎵 Recording started...")
    
    # Pull the most recent sample from appsink
    sample = sink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        
        if success:
            raw_audio = map_info.data
            buffered_audio += raw_audio
            total_samples_received += len(raw_audio)
            
            # Calculate current duration
            current_duration = len(buffered_audio) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
            
            # Print progress every second of audio
            if len(buffered_audio) >= TARGET_BYTES and len(buffered_audio) % TARGET_BYTES < len(raw_audio):
                elapsed_time = time.time() - start_time if start_time else 0
                print(f"📊 Recorded: {format_duration(current_duration)} | "
                      f"Size: {format_size(len(buffered_audio))} | "
                      f"Real-time: {format_duration(elapsed_time)}")
            
            # Clean up
            buffer.unmap(map_info)
    
    return Gst.FlowReturn.OK

def save_audio_to_wav():
    """Save the buffered audio data to a WAV file"""
    if len(buffered_audio) == 0:
        print("⚠️  No audio data to save!")
        return None
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"
    
    try:
        # Convert bytearray to numpy array of 16-bit integers
        audio_data = np.frombuffer(buffered_audio, dtype=np.int16)
        
        # Create WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(buffered_audio)
        
        # Calculate file info
        duration = len(buffered_audio) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
        file_size = os.path.getsize(filename)
        
        print(f"💾 Audio saved successfully!")
        print(f"   📁 File: {filename}")
        print(f"   ⏱️  Duration: {format_duration(duration)}")
        print(f"   📊 Size: {format_size(file_size)}")
        print(f"   🔊 Format: {SAMPLE_RATE}Hz, {CHANNELS} channel, {SAMPLE_WIDTH*8}-bit")
        print(f"   📈 Samples: {len(audio_data):,}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving WAV file: {e}")
        return None

def input_thread():
    """Thread to handle user input for stopping recording"""
    global recording
    
    while recording:
        try:
            user_input = input().strip().lower()
            if user_input == 'stop':
                print("\n🛑 Stop command received. Finishing recording...")
                recording = False
                break
        except (EOFError, KeyboardInterrupt):
            print("\n🛑 Interrupt received. Finishing recording...")
            recording = False
            break

def main():
    global recording
    
    # Set up the callback for new audio samples
    appsink.connect("new-sample", on_new_sample)
    
    # Start the GStreamer pipeline
    try:
        pipeline.set_state(Gst.State.PLAYING)
        print("🎤 Pipeline started. Waiting for audio from Orange Pi...")
        print("   (Make sure Orange Pi is streaming audio to this network)")
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return
    
    # Start input thread for stop command
    input_thread_handle = threading.Thread(target=input_thread, daemon=True)
    input_thread_handle.start()
    
    # Main loop
    loop = GLib.MainLoop()
    
    try:
        # Run the main loop in a separate thread so we can handle keyboard interrupt
        loop_thread = threading.Thread(target=loop.run, daemon=True)
        loop_thread.start()
        
        # Wait for recording to finish
        while recording and loop_thread.is_alive():
            time.sleep(0.1)
        
        # Stop the main loop
        loop.quit()
        
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received. Finishing recording...")
        recording = False
        loop.quit()
    
    # Stop the pipeline
    print("🔄 Stopping pipeline...")
    pipeline.set_state(Gst.State.NULL)
    
    # Save the recorded audio
    if len(buffered_audio) > 0:
        print("💾 Saving recorded audio...")
        filename = save_audio_to_wav()
        if filename:
            print(f"✅ Recording complete! Saved as: {filename}")
        else:
            print("❌ Failed to save recording")
    else:
        print("⚠️  No audio was recorded")
    
    print("👋 Test server stopped.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 