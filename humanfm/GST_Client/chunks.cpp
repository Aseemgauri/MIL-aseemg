#include "AudioFile.h"
#include <gst/gst.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <jack/jack.h>
#include "gst_config.h"  // Include the static configuration (auto-generated)

/*----- Configuration System ----- */
// All configuration values are now loaded from system.conf
// No more hardcoded values!

/*----- Global Variables ----- */

// Audio file object for file mode
AudioFile<float> audioFile;
bool isFileLoaded = false;

/*----- Helper Functions ----- */

std::string getTargetIP() {
    std::string mode = SYSTEM_STREAMING_MODE;
    if (mode == "multicast") {
        return SYSTEM_MULTICAST_ADDRESS;
    } else if (mode == "unicast") {
        return SYSTEM_LAPTOP_IP;
    } else {
        return "255.255.255.255";  // Default to broadcast
    }
}

std::string getFullAudioPath() {
    return std::string(GST_AUDIO_FILE_DIRECTORY) + "/" + GST_DEFAULT_AUDIO_FILE;
}

std::string getAudioSourceString() {
    std::string audioSrc;
    std::string backend = GST_AUDIO_BACKEND;
    
    if (backend == "jack") {
        // Configure jackaudiosrc with proper connection settings
        audioSrc = std::string("jackaudiosrc connect=auto-forced client-name=") + GST_JACK_CLIENT_NAME;
        std::cout << "🔊 Using JACK audio source (jackaudiosrc)" << std::endl;
        std::cout << "   Connect mode: auto-forced (connects to available physical ports)" << std::endl;
        std::cout << "   Audio amplification: " << GST_AMPLIFY_COMPENSATION << "x + Volume: " << GST_VOLUME_COMPENSATION << "x" << std::endl;
        std::cout << "   Total gain boost for ML detection" << std::endl;
        std::cout << "   Make sure JACK server is running!" << std::endl;
    } else if (backend == "pulse") {
        audioSrc = "pulsesrc";
        std::cout << "🔊 Using PulseAudio source (pulsesrc)" << std::endl;
    } else if (backend == "alsa") {
        audioSrc = std::string("alsasrc device=") + GST_ALSA_DEVICE;
        std::cout << "🔊 Using ALSA source (alsasrc) device: " << GST_ALSA_DEVICE << std::endl;
    } else {
        audioSrc = "autoaudiosrc";
        std::cout << "🔊 Using automatic audio source (autoaudiosrc)" << std::endl;
        std::cout << "   ⚠️  This may conflict with JACK if it's running!" << std::endl;
    }

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
    // Print configuration information
    std::cout << "=== GST Client Configuration (from system.conf) ===" << std::endl;
    std::cout << "Audio Mode: " << (GST_USE_FILE_AUDIO ? "file" : "live") << std::endl;
    std::cout << "Audio Backend: " << GST_AUDIO_BACKEND << std::endl;
    std::cout << "Sample Rate: " << SYSTEM_SAMPLE_RATE << " Hz" << std::endl;
    std::cout << "Streaming Mode: " << SYSTEM_STREAMING_MODE << std::endl;
    std::cout << "Target IP: " << getTargetIP() << std::endl;
    std::cout << "UDP Port: " << SYSTEM_UDP_STREAMING_PORT << std::endl;
    std::cout << "Volume Compensation: " << GST_VOLUME_COMPENSATION << "x" << std::endl;
    std::cout << "Amplify Compensation: " << GST_AMPLIFY_COMPENSATION << "x" << std::endl;
    std::cout << "========================================" << std::endl;

    bool useFileAudio = GST_USE_FILE_AUDIO;
    
    // Load WAV file if in file mode (always use config file path)
    if (useFileAudio) {
        std::string file_path = getFullAudioPath();
        
        isFileLoaded = audioFile.load(file_path);
        if (!isFileLoaded) {
            std::cout << "Error: Could not load WAV file: " << file_path << std::endl;
            std::cout << "💡 Check GST_AUDIO_FILE_DIRECTORY and GST_DEFAULT_AUDIO_FILE in system.conf" << std::endl;
            return 1;
        }
        std::cout << "✅ Loaded WAV file: " << file_path << std::endl;
        audioFile.printSummary();
    }

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Build pipeline based on configuration
    std::string pipeline_str;
    std::string target_ip = getTargetIP();
    int udp_port = SYSTEM_UDP_STREAMING_PORT;
    int sample_rate = SYSTEM_SAMPLE_RATE;
    std::string streaming_mode = SYSTEM_STREAMING_MODE;

    if (useFileAudio) {
        // File mode - stream WAV file at real-time rate
        std::string file_path = getFullAudioPath();
        
        pipeline_str = "filesrc location=" + file_path + 
                       " ! wavparse ! audioconvert ! audioresample "
                       "! audio/x-raw,rate=" + std::to_string(sample_rate) + ",format=S16LE,channels=1 "
                       "! identity sync=true ! opusenc ! rtpopuspay ! udpsink ";
        
        if (streaming_mode == "unicast") {
            pipeline_str += "host=" + target_ip + " port=" + std::to_string(udp_port);
            std::cout << "📡 Using FILE + UNICAST mode (" << target_ip << ":" << udp_port << ")" << std::endl;
        } else if (streaming_mode == "multicast") {
            pipeline_str += std::string("host=") + SYSTEM_MULTICAST_ADDRESS + " auto-multicast=true port=" + std::to_string(udp_port);
            std::cout << "📡 Using FILE + MULTICAST mode (" << SYSTEM_MULTICAST_ADDRESS << ":" << udp_port << ")" << std::endl;
        } else {
            pipeline_str += std::string("host=") + GST_BROADCAST_ADDRESS + " port=" + std::to_string(udp_port);
            std::cout << "📡 Using FILE + BROADCAST mode (" << GST_BROADCAST_ADDRESS << ":" << udp_port << ")" << std::endl;
        }
    } else {
        // Live microphone mode
        std::string audioSrc = getAudioSourceString();
        
        // Pipeline with both amplification and volume for better ML detection
        pipeline_str = audioSrc + 
                       " ! audioconvert "
                       "! audioamplify amplification=" + std::to_string(GST_AMPLIFY_COMPENSATION) + " "
                       "! audioresample "
                       "! audio/x-raw,rate=" + std::to_string(sample_rate) + ",format=S16LE,channels=2 "
                       "! audioconvert "
                       "! audio/x-raw,rate=" + std::to_string(sample_rate) + ",format=S16LE,channels=1 "
                       "! volume volume=" + std::to_string(GST_VOLUME_COMPENSATION) + " "
                       "! opusenc "
                       "! rtpopuspay "
                       "! udpsink ";

        if (streaming_mode == "unicast") {
            pipeline_str += "host=" + target_ip + " port=" + std::to_string(udp_port);
            std::cout << "📡 Using LIVE + UNICAST mode (" << target_ip << ":" << udp_port << ")" << std::endl;
        } else if (streaming_mode == "multicast") {
            pipeline_str += std::string("host=") + SYSTEM_MULTICAST_ADDRESS + " auto-multicast=true port=" + std::to_string(udp_port);
            std::cout << "📡 Using LIVE + MULTICAST mode (" << SYSTEM_MULTICAST_ADDRESS << ":" << udp_port << ")" << std::endl;
        } else {
            pipeline_str += std::string("host=") + GST_BROADCAST_ADDRESS + " port=" + std::to_string(udp_port);
            std::cout << "📡 Using LIVE + BROADCAST mode (" << GST_BROADCAST_ADDRESS << ":" << udp_port << ")" << std::endl;
        }
    }

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

    // Start the pipeline
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

    if (useFileAudio) {
        std::cout << "📁 Streaming WAV file... Press Ctrl+C to stop.\n";
    } else {
        std::cout << "🎤 Streaming live microphone audio... Press Ctrl+C to stop.\n";
        
        // Automatically connect JACK ports for live audio mode
        std::thread connectionThread([&]() {
            if (connectJackPorts(GST_JACK_CLIENT_NAME)) {
                std::cout << "🔊 Audio input successfully connected!" << std::endl;
            } else {
                std::cout << "⚠️  Failed to auto-connect JACK ports. You may need to connect manually." << std::endl;
            }
        });
        connectionThread.detach(); // Let it run in background
    }

    // Keep the pipeline running
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop);

    // Clean-up
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}
