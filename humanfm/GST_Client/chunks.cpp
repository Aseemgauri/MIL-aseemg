#include "AudioFile.h"
#include <gst/gst.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <jack/jack.h>

/*----- Configuration Macros ----- */
#define USE_BROADCAST 1  // Set to 1 for broadcast, 0 for multicast
#define USE_FILE_AUDIO 0  // Set to 1 for WAV file, 0 for live microphone

// Audio backend selection (choose one)
#define AUDIO_BACKEND_AUTO 0      // autoaudiosrc (automatic selection)
#define AUDIO_BACKEND_JACK 1      // jackaudiosrc (for JACK server)
#define AUDIO_BACKEND_PULSE 0     // pulsesrc (for PulseAudio)
#define AUDIO_BACKEND_ALSA 0      // alsasrc (direct ALSA)

// ALSA device specification (only used if AUDIO_BACKEND_ALSA is 1)
#define ALSA_DEVICE "hw:0,0"      // Change to your specific device

/*----- Global Variables ----- */

// Audio file object for file mode
AudioFile<float> audioFile;
bool isFileLoaded = false;

/*----- Helper Functions ----- */

std::string getAudioSourceString() {
    std::string audioSrc;
    
#if AUDIO_BACKEND_JACK
    // Configure jackaudiosrc with proper connection settings
    audioSrc = "jackaudiosrc connect=auto-forced client-name=gst_audio_client";
    std::cout << "🔊 Using JACK audio source (jackaudiosrc)" << std::endl;
    std::cout << "   Connect mode: auto-forced (connects to available physical ports)" << std::endl;
    std::cout << "   Make sure JACK server is running!" << std::endl;
#elif AUDIO_BACKEND_PULSE
    audioSrc = "pulsesrc";
    std::cout << "🔊 Using PulseAudio source (pulsesrc)" << std::endl;
#elif AUDIO_BACKEND_ALSA
    audioSrc = "alsasrc device=" ALSA_DEVICE;
    std::cout << "🔊 Using ALSA source (alsasrc) device: " << ALSA_DEVICE << std::endl;
#else
    audioSrc = "autoaudiosrc";
    std::cout << "🔊 Using automatic audio source (autoaudiosrc)" << std::endl;
    std::cout << "   ⚠️  This may conflict with JACK if it's running!" << std::endl;
#endif

    return audioSrc;
}



// Function to automatically connect JACK ports for the GST client
bool connectJackPorts(const std::string& clientName) {
    std::cout << "🔌 Attempting to connect JACK ports for client: " << clientName << std::endl;
    
    // Wait a bit for the GStreamer JACK client to fully initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    jack_client_t* jack_client = jack_client_open("jack_connector", JackNoStartServer, nullptr);
    if (!jack_client) {
        std::cerr << "❌ Failed to connect to JACK server for port connection" << std::endl;
        return false;
    }
    
    // Try to find the GST client ports (try multiple times in case they're not ready yet)
    std::string gst_port1 = clientName + ":in_jackaudiosrc0_1";
    std::string gst_port2 = clientName + ":in_jackaudiosrc0_2";
    
    bool found_ports = false;
    for (int attempt = 0; attempt < 10; attempt++) {
        jack_port_t* port1 = jack_port_by_name(jack_client, gst_port1.c_str());
        jack_port_t* port2 = jack_port_by_name(jack_client, gst_port2.c_str());
        
        if (port1 && port2) {
            found_ports = true;
            break;
        }
        
        std::cout << "⏳ Waiting for GST JACK client ports... (attempt " << (attempt + 1) << "/10)" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    if (!found_ports) {
        std::cerr << "❌ Could not find GST JACK client ports: " << gst_port1 << ", " << gst_port2 << std::endl;
        jack_client_close(jack_client);
        return false;
    }
    
    // Connect system:capture_1 -> gst_client:in_jackaudiosrc0_1
    int result1 = jack_connect(jack_client, "system:capture_1", gst_port1.c_str());
    if (result1 == 0) {
        std::cout << "✅ Connected system:capture_1 -> " << gst_port1 << std::endl;
    } else if (result1 == EEXIST) {
        std::cout << "ℹ️  Connection already exists: system:capture_1 -> " << gst_port1 << std::endl;
    } else {
        std::cerr << "❌ Failed to connect system:capture_1 -> " << gst_port1 << " (error: " << result1 << ")" << std::endl;
    }
    
    // Connect system:capture_2 -> gst_client:in_jackaudiosrc0_2
    int result2 = jack_connect(jack_client, "system:capture_2", gst_port2.c_str());
    if (result2 == 0) {
        std::cout << "✅ Connected system:capture_2 -> " << gst_port2 << std::endl;
    } else if (result2 == EEXIST) {
        std::cout << "ℹ️  Connection already exists: system:capture_2 -> " << gst_port2 << std::endl;
    } else {
        std::cerr << "❌ Failed to connect system:capture_2 -> " << gst_port2 << " (error: " << result2 << ")" << std::endl;
    }
    
    jack_client_close(jack_client);
    
    bool success = (result1 == 0 || result1 == EEXIST) && (result2 == 0 || result2 == EEXIST);
    if (success) {
        std::cout << "🎵 JACK audio routing established successfully!" << std::endl;
    }
    
    return success;
}

/*----- Main Function ----- */

// Puts together the different program pieces above.
int main(int argc, char *argv[]) {
  // Check command line arguments
  if (USE_FILE_AUDIO && argc < 2) {
    std::cout << "Usage: " << argv[0] << " <wav_file_path>" << std::endl;
    std::cout << "Example: " << argv[0] << " audio/sample.wav" << std::endl;
    return 1;
  }

  // Load WAV file if in file mode
  if (USE_FILE_AUDIO) {
    std::string file_path = argv[1];
    isFileLoaded = audioFile.load(file_path);
    if (!isFileLoaded) {
      std::cout << "Error: Could not load WAV file: " << file_path << std::endl;
      return 1;
    }
    std::cout << "✅ Loaded WAV file: " << file_path << std::endl;
    audioFile.printSummary();
  }

  // Initializing Gstreamer and ensuring that we can process any command line arguments.
  gst_init(&argc, &argv);

  // Creating and parsing a Gstreamer pipeline string.
  // Capturing audio from live microphone or file, converting to the proper format, preparing
  // and creating packets to be sent to multiple receivers.
  
  std::string pipeline_str;

#if USE_FILE_AUDIO
  // File mode - stream WAV file at real-time rate (16kHz)
  pipeline_str = "filesrc location=" + std::string(argv[1]) + 
                 " ! wavparse ! audioconvert ! audioresample "
                 "! audio/x-raw,rate=16000,format=S16LE,channels=1 "
                 "! identity sync=true ! opusenc ! rtpopuspay ! udpsink ";
  
#if USE_BROADCAST
  pipeline_str += "host=255.255.255.255 port=5000";
  std::cout << "📡 Using FILE + BROADCAST mode (255.255.255.255:5000)" << std::endl;
#else
  pipeline_str += "host=239.1.1.1 auto-multicast=true port=5000";
  std::cout << "📡 Using FILE + MULTICAST mode (239.1.1.1:5000)" << std::endl;
#endif

#else
  // Live microphone mode
  std::string audioSrc = getAudioSourceString();
  
  pipeline_str = audioSrc + 
                 " ! audioconvert "
                 "! audioresample "
                 "! audio/x-raw,rate=16000,format=S16LE,channels=2 "
                 "! audioconvert "
                 "! audio/x-raw,rate=16000,format=S16LE,channels=1 "
                 "! opusenc "
                 "! rtpopuspay "
                 "! udpsink ";

#if USE_BROADCAST
  pipeline_str += "host=255.255.255.255 port=5000";
  std::cout << "📡 Using LIVE + BROADCAST mode (255.255.255.255:5000)" << std::endl;
#else
  pipeline_str += "host=239.1.1.1 auto-multicast=true port=5000";
  std::cout << "📡 Using LIVE + MULTICAST mode (239.1.1.1:5000)" << std::endl;
#endif

#endif

  std::cout << "🔧 GStreamer pipeline: " << pipeline_str << std::endl;

  // Parse and create the pipeline
  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
  
  if (!pipeline) {
    std::cout << "❌ Failed to create GStreamer pipeline!" << std::endl;
    if (error) {
      std::cout << "Error: " << error->message << std::endl;
      g_error_free(error);
    }
    return 1;
  }

  std::cout << "✅ GStreamer pipeline created successfully" << std::endl;

  // Start the pipeline and began recording from microphone or streaming file.
  GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  
  if (ret == GST_STATE_CHANGE_FAILURE) {
    std::cout << "❌ Failed to start GStreamer pipeline!" << std::endl;
    std::cout << "💡 Try running the audio diagnostic script: ./audio_device_check.sh" << std::endl;
    
    // Check for JACK-specific errors
    std::cout << "🔍 JACK Debugging Steps:" << std::endl;
    std::cout << "   1. Check if JACK server is running: jack_lsp" << std::endl;
    std::cout << "   2. Check JACK connections: jack_lsp -c" << std::endl;
    std::cout << "   3. Check for audio input sources connected to system:capture_*" << std::endl;
    
    gst_object_unref(pipeline);
    return 1;
  }

  std::cout << "✅ GStreamer pipeline started successfully" << std::endl;

#if USE_FILE_AUDIO
  std::cout << "📁 Streaming WAV file... Press Ctrl+C to stop.\n";
#else
  std::cout << "🎤 Streaming live microphone audio... Press Ctrl+C to stop.\n";
  
  // Automatically connect JACK ports for live audio mode
  std::thread connectionThread([&]() {
    if (connectJackPorts("gst_audio_client")) {
      std::cout << "🔊 Audio input successfully connected!" << std::endl;
    } else {
      std::cout << "⚠️  Failed to auto-connect JACK ports. You may need to connect manually." << std::endl;
    }
  });
  connectionThread.detach(); // Let it run in background
#endif

  // Similar to a while true loop which ensures audio continues to be recorded
  // until we manually sever the connection.
  GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
  g_main_loop_run(loop);

  // Clean-up and freeing of memory.
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  return 0;
}
