import sys
import os

# Add GStreamer site-packages to Python path before importing gi
#gstreamer_site_packages = r"C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages"
#if gstreamer_site_packages not in sys.path:
#    sys.path.insert(0, gstreamer_site_packages)

import gi
import requests
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
from mdl_model import MdlModel

# Configuration
USE_BROADCAST = True  # Set to True for broadcast, False for multicast
DEMO_MODE = False  # Set to True for demo mode, False for real classification
DEMO_CLASS = "Baby cry"  # The class to constantly show as detected in demo mode
CLASS_THRESHOLD = 0.4  # Threshold for determining if a class is active (0.0 to 1.0)

# Orange Pi Node.js server configuration
ORANGEPI_IP = "192.168.2.218"  # Replace with your Orange Pi's actual IP address
NODE_SERVER_PORT = 8000

# Initialize the model (ONNX and label CSV in current directory)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'audio_mdl.onnx')
LABEL_CSV = os.path.join(os.path.dirname(__file__), 'class_labels_indices.csv')
model = MdlModel(MODEL_PATH, LABEL_CSV)
callback = model.callback

# Initializing the Gstreamer object to begin listening for audio.
Gst.init(None)

# Pipeline details in a string.
# Processing UDP packets that have been compressed via OPUS, decompressing,
# and processing in a format that allows Python to read buffered data.
# Modifications made to allow for one channel, 16-bit samples, and 16kHz.
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

# Builds the pipeline and attempt to connect.
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

# ----- Audio Classification -----
# Set-up a byte array to process binary data (in this case audio).
buffered_audio = bytearray()
# 1 second of mono 16-bit PCM audio @ 16kHz.
'''
16-bit (2 bytes)
Mono (1 channel)
16,000 samples/second (16 kHz)
'''
# 640 bytes ÷ 2 bytes/sample = 320 samples
# 320 samples ÷ 16,000 samples/sec = 0.02 sec = 20 ms
# 16,000 samples/sec * 2 bytes/sample = 32,000 bytes/sec
TARGET_BYTES = 32000

# Pseudo-classifier function that converts raw PCM bytes to NumPy array and
# classifies based on mean power.
def classify_audio(audio_bytes):
    # This function is now unused, but kept for reference
    # Convert raw bytes to 16-bit PCM, then to float32 [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize to float32 [-1, 1] range (optional).
    # Ensures values are easier to compare.
    # 32768.0 is the maximum magnitude of a signed 16-bit number (2¹⁵).
    audio_float = audio_np.astype(np.float32) / 32768.0

    # Compute power (RMS).
    # Square each sample, get the average, and return the square root.
    power = np.sqrt(np.mean(audio_float ** 2))

    # Formatted to five decimal places.
    print(f"📊 Signal Power: {power:.5f}")

    # Pseudo-classification based on power level.
    if power > 0.2:
        print("🔊 Loud sound detected")
        return "LOUD"
    elif power > 0.05:
        print("🟢 Medium sound detected")
        return "MEDIUM"
    else:
        print("🔇 Quiet or background noise")
        return "QUIET"

# ----- Sample Collection -----
# Function that is triggered every time a new audio buffer arrives.
def on_new_sample(sink):
    # Defined as global to ensure global variable can be modified in this function.
    global buffered_audio

    # We pull the most recent sample from app-sink.
    sample = sink.emit("pull-sample")
    # If we indeed pull the sample, we extract the raw buffer data.
    # Mapping into memory so it can be read.
    if sample:
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        # If we successfully map, we attempt to process the raw audio data itself.
        if success:
            raw_audio = map_info.data
            buffered_audio += raw_audio

            print(f"📥 Received chunk: {len(raw_audio)} bytes | Buffered: {len(buffered_audio)} bytes")

            if len(buffered_audio) >= TARGET_BYTES:
                print("🧠 1-second buffer ready → Running classification...")
                
                if DEMO_MODE:
                    print(f"🎭 DEMO MODE ACTIVE: Will send {DEMO_CLASS} as detected regardless of actual classification")
                    # Still run classification for logging purposes
                    target_results = callback(buffered_audio[:TARGET_BYTES])
                    print("Actual target class scores (ignored in demo mode):")
                    for i, (label, score) in enumerate(target_results, 1):
                        thresholded_score = 1.0 if score >= CLASS_THRESHOLD else 0.0
                        print(f"  {i}. {label}: {score:.4f} → {thresholded_score}")
                else:
                    # Real mode: process normally
                    target_results = callback(buffered_audio[:TARGET_BYTES])
                    print("Target class scores:")
                    for i, (label, score) in enumerate(target_results, 1):
                        thresholded_score = 1.0 if score >= CLASS_THRESHOLD else 0.0
                        print(f"  {i}. {label}: {score:.4f} → {thresholded_score}")
                
                send_class_to_node(target_results)
                buffered_audio = bytearray()  # reset for next second

            # Clean-up after we are done using the current map.
            buffer.unmap(map_info)

    # Returning a success command.
    return Gst.FlowReturn.OK

# ----- Sending Detection to Node Server -----
def send_class_to_node(top_5_results):
    url = f"http://{ORANGEPI_IP}:{NODE_SERVER_PORT}/HumanFM/classify"
    
    if DEMO_MODE:
        # Demo mode: Send fixed demo data
        print(f"🎭 DEMO MODE: Sending {DEMO_CLASS} as detected")
        payload = {
            "classifications": [
                {"label": "Baby cry", "score": 1.0},
                {"label": "Cat", "score": 0.0},
                {"label": "Rooster", "score": 0.0},
                {"label": "Crickets", "score": 0.0},
                {"label": DEMO_CLASS, "score": 0.0}
            ]
        }
    else:
        # Real mode: Convert list of tuples to list of dictionaries for JSON serialization
        # Apply thresholding: scores >= CLASS_THRESHOLD become 1, scores < CLASS_THRESHOLD become 0
        payload = {
            "classifications": [
                {"label": label, "score": 1.0 if score >= CLASS_THRESHOLD else 0.0} 
                for label, score in top_5_results
            ]
        }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            if DEMO_MODE:
                print("✅ Successfully sent demo data to Node.js")
            else:
                print("✅ Successfully sent top 5 classifications to Node.js")
        else:
            print(f"❌ Failed to send: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"🚨 Connection error: Cannot reach Orange Pi at {ORANGEPI_IP}:{NODE_SERVER_PORT}")
        print(f"   Check if Orange Pi Node.js server is running")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Request error: {e}")

# Setting-up a callback so the function above is run every time we receive new audio.
appsink.connect("new-sample", on_new_sample)

# Starts the pipeline for listening and processing audio data.
pipeline.set_state(Gst.State.PLAYING)
print("🎤 Python pipeline listening for audio...")

print(f"🎯 Target classes: {', '.join(model.get_target_classes())}")
print(f"📊 Classification threshold: {CLASS_THRESHOLD} (classes with scores >= {CLASS_THRESHOLD} will be marked as active)")

if DEMO_MODE:
    print(f"🎭 DEMO MODE ENABLED: Will constantly send '{DEMO_CLASS}' as detected")
    print("   Set DEMO_MODE = False to use real classifications")
else:
    print("🔬 REAL MODE: Using actual AI classifications")

print(f"📡 Will send classification data to: http://{ORANGEPI_IP}:{NODE_SERVER_PORT}")
print(f"   Make sure Orange Pi Node.js server is running at this address")

# Creating a while true loop to ensure the pipeline continues to listen for audio
# and doesn't immediately exit.
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass

# Shutting down the pipeline and cleaning up.
pipeline.set_state(Gst.State.NULL)