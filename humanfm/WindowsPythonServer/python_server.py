import os
import gi
import requests

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from mdl_model import MdlModel

# Note: If GStreamer path issues occur, uncomment and modify the following:
# import sys
# gstreamer_site_packages = r"C:\Program Files\GStreamer\1.0\msvc_x86_64\lib\site-packages"
# if gstreamer_site_packages not in sys.path:
#     sys.path.insert(0, gstreamer_site_packages)

# ----- Configuration -----
# Audio processing settings
USE_DIRECT_IP = True   # Set to True for direct IP, False for broadcast/multicast
USE_BROADCAST = False  # Set to True for broadcast, False for multicast
LISTEN_IP = "0.0.0.0"  # Listen on all interfaces to accept from any sender
UDP_PORT = 5000
MULTICAST_ADDRESS = "239.1.1.1"

# Classification settings
DEMO_MODE = False  # Set to True for demo mode, False for real classification
DEMO_CLASS = "Baby cry"  # The class to constantly show as detected in demo mode
CLASS_THRESHOLD = 0.1  # Threshold for determining if a class is active (0.0 to 1.0)

# Node.js server configuration
ORANGEPI_IP = "100.82.70.55"  # Replace with your Orange Pi's actual IP address
NODE_SERVER_PORT = 8000

# Model initialization
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'audio_mdl.onnx')
LABEL_CSV = os.path.join(os.path.dirname(__file__), 'class_labels_indices.csv')
DEBUG_MODE = False  # Set to True to enable detailed model debug output

model = MdlModel(MODEL_PATH, LABEL_CSV, debug=DEBUG_MODE)
callback = model.callback

# Initializing the Gstreamer object to begin listening for audio.
Gst.init(None)

# Pipeline details in a string.
# Processing UDP packets that have been compressed via OPUS, decompressing,
# and processing in a format that allows Python to read buffered data.
# Modifications made to allow for one channel, 16-bit samples, and 16kHz.
# ----- GStreamer Pipeline Setup -----
if USE_DIRECT_IP:
    # Direct IP mode - listen on specific interface
    pipeline_str = (
        f'udpsrc address={LISTEN_IP} port={UDP_PORT} buffer-size=8192 '
        'caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=96" '
        '! rtpopusdepay '
        '! decodebin '
        '! audioconvert '
        '! audioresample '
        '! audio/x-raw,rate=16000,format=S16LE,channels=1 '
        '! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
    )
    print(f"📡 DIRECT IP mode (LOW-LATENCY) - listening on {LISTEN_IP}:{UDP_PORT}")
elif USE_BROADCAST:
    # Broadcast mode - minimal buffering for real-time processing
    pipeline_str = (
        f'udpsrc port={UDP_PORT} buffer-size=8192 '
        'caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=96" '
        '! rtpopusdepay '
        '! decodebin '
        '! audioconvert '
        '! audioresample '
        '! audio/x-raw,rate=16000,format=S16LE,channels=1 '
        '! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
    )
    print(f"📡 BROADCAST mode (LOW-LATENCY) - listening on port {UDP_PORT}")
else:
    # Multicast mode - minimal buffering for real-time processing
    pipeline_str = (
        f'udpsrc uri=udp://{MULTICAST_ADDRESS}:{UDP_PORT} buffer-size=8192 '
        'caps="application/x-rtp,media=audio,encoding-name=OPUS,payload=96" '
        '! rtpopusdepay '
        '! decodebin '
        '! audioconvert '
        '! audioresample '
        '! audio/x-raw,rate=16000,format=S16LE,channels=1 '
        '! appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true'
    )
    print(f"📡 MULTICAST mode (LOW-LATENCY) - listening on {MULTICAST_ADDRESS}:{UDP_PORT}")

# Builds the pipeline and attempt to connect.
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

# ----- Audio Configuration -----
# Audio format: 16-bit mono PCM @ 16kHz
# 1 second = 16,000 samples × 2 bytes = 32,000 bytes
TARGET_BYTES = 32000
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# Fixed-size ring buffer that's always exactly 1 second of audio
audio_ring_buffer = bytearray(TARGET_BYTES)  # FIXED size - always 32,000 bytes
ring_write_pos = 0  # Current write position in the ring buffer
ring_is_full = False  # Track if we've written at least 1 full second
processing_in_progress = False  # Flag to track if model is currently processing


# ----- Ring Buffer Operations (FIXED SIZE, OVERWRITE, EMPTY) -----
def add_audio_to_ring_buffer(new_audio_data):
    """
    Add new audio to FIXED-SIZE ring buffer. 
    Buffer is ALWAYS exactly TARGET_BYTES (32,000 bytes).
    New samples OVERWRITE oldest samples at the write position.
    Always contains the most recent 1 second of audio in chronological order.
    """
    global ring_write_pos, ring_is_full
    
    for byte_val in new_audio_data:
        # Write byte at current position, overwriting old data
        audio_ring_buffer[ring_write_pos] = byte_val
        
        # Advance write position, wrapping around when we reach the end
        ring_write_pos = (ring_write_pos + 1) % TARGET_BYTES
        
        # Mark as full once we've written a complete cycle
        if ring_write_pos == 0 and not ring_is_full:
            ring_is_full = True

def consume_audio_from_ring_buffer():
    """
    Extract the FULL 1 second of audio from ring buffer and COMPLETELY EMPTY it.
    Returns None if buffer is not full yet.
    After this call, buffer is reset and must fill up again from scratch.
    """
    global ring_write_pos, ring_is_full
    
    if not ring_is_full:
        return None  # Don't process until we have a complete 1 second
    
    # Extract audio in chronological order (oldest to newest)
    audio_to_process = bytearray(TARGET_BYTES)
    
    # Start reading from the current write position (oldest data)
    # and read in circular fashion to get chronological order
    for i in range(TARGET_BYTES):
        read_pos = (ring_write_pos + i) % TARGET_BYTES
        audio_to_process[i] = audio_ring_buffer[read_pos]
    
    # COMPLETELY EMPTY: Reset the ring buffer state
    ring_write_pos = 0
    ring_is_full = False
    # Note: We don't need to clear the actual buffer data, just reset the state
    
    print(f"🍽️ Consumed 1 second of audio | Ring buffer is now EMPTY and ready to fill")
    
    return bytes(audio_to_process)

def can_start_processing():
    """Check if we can start processing (ring buffer is full and not already processing)"""
    return ring_is_full and not processing_in_progress

def get_ring_buffer_status():
    """Get current ring buffer status for debugging"""
    # Calculate how much has been written since last reset
    bytes_written = ring_write_pos if not ring_is_full else TARGET_BYTES
    
    return {
        'bytes_written': bytes_written,
        'target_bytes': TARGET_BYTES,
        'is_full': ring_is_full,
        'is_ready': ring_is_full,
        'write_pos': ring_write_pos
    }

# ----- Sample Collection -----
# Function that is triggered every time a new audio buffer arrives.
def on_new_sample(sink):
    global processing_in_progress

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
            
            # ALWAYS add new audio to FIXED-SIZE ring buffer (overwrites oldest)
            add_audio_to_ring_buffer(raw_audio)
            
            # Show current ring buffer status
            status = get_ring_buffer_status()
            print(f"📥 Received chunk: {len(raw_audio)} bytes | Ring buffer: {status['bytes_written']}/{TARGET_BYTES} bytes | Full: {status['is_full']} | Processing: {processing_in_progress}")

            # Only start processing if ring buffer is FULL AND not already processing
            if can_start_processing():
                print("🧠 Ring buffer FULL → Starting classification...")
                processing_in_progress = True
                
                # CONSUME: Extract 1 second and COMPLETELY EMPTY the ring buffer
                audio_to_process = consume_audio_from_ring_buffer()
                
                if audio_to_process:
                    if DEMO_MODE:
                        print(f"🎭 DEMO MODE ACTIVE: Will send {DEMO_CLASS} as detected regardless of actual classification")
                        # Still run classification for logging purposes
                        target_results = callback(audio_to_process)
                        print("Actual target class scores (ignored in demo mode):")
                        for i, (label, score) in enumerate(target_results, 1):
                            thresholded_score = 1.0 if score >= CLASS_THRESHOLD else 0.0
                            print(f"  {i}. {label}: {score:.4f} → {thresholded_score}")
                    else:
                        # Real mode: process normally
                        target_results = callback(audio_to_process)
                        print("Target class scores:")
                        for i, (label, score) in enumerate(target_results, 1):
                            thresholded_score = 1.0 if score >= CLASS_THRESHOLD else 0.0
                            print(f"  {i}. {label}: {score:.4f} → {thresholded_score}")
                    
                    send_class_to_node(target_results)
                
                processing_in_progress = False
                print("✅ Processing complete - ring buffer is EMPTY and filling with NEW audio")

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

# ----- Main Execution -----
def main():
    """Main execution function"""
    # Setup callback and start pipeline
    appsink.connect("new-sample", on_new_sample)
    pipeline.set_state(Gst.State.PLAYING)
    
    # Display configuration
    print("🎤 Python pipeline listening for audio...")
    print(f"🎯 Target classes: {', '.join(model.get_target_classes())}")
    print(f"📊 Classification threshold: {CLASS_THRESHOLD}")
    
    if DEMO_MODE:
        print(f"🎭 DEMO MODE: Will constantly send '{DEMO_CLASS}' as detected")
    else:
        print("🔬 REAL MODE: Using actual AI classifications")
    
    print(f"📡 Sending data to: http://{ORANGEPI_IP}:{NODE_SERVER_PORT}")
    
    # Run main loop
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        pipeline.set_state(Gst.State.NULL)
        print("🏁 Pipeline stopped")

if __name__ == "__main__":
    main()