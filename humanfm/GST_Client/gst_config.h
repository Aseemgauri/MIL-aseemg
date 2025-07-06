#ifndef GST_CONFIG_H
#define GST_CONFIG_H

/*
 * GST Client Configuration
 * 
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY!
 * Generated from system.conf by generate_configs.py
 * 
 * To change configuration values, edit system.conf and run:
 * python3 generate_configs.py
 */

#define SYSTEM_SAMPLE_RATE 16000
#define SYSTEM_NUM_CLASSES 5
#define SYSTEM_STREAMING_MODE "unicast"
#define SYSTEM_ORANGEPI_IP "100.82.70.55"
#define SYSTEM_LAPTOP_IP "100.121.151.92"
#define SYSTEM_UDP_STREAMING_PORT 5000
#define SYSTEM_MULTICAST_ADDRESS "239.1.1.1"
#define GST_AUDIO_MODE "live"
#define GST_AUDIO_FILE_DIRECTORY "/home/orangepi/Documents/MIL-aseemg/humanfm/audio_samples/"
#define GST_DEFAULT_AUDIO_FILE "test_sample.wav"
#define GST_VOLUME_COMPENSATION 4.0
#define GST_AMPLIFY_COMPENSATION 16.0
#define GST_AUDIO_BACKEND "jack"
#define GST_ALSA_DEVICE "hw:0,0"
#define GST_JACK_CLIENT_NAME "gst_audio_client"
#define GST_BROADCAST_ADDRESS "255.255.255.255"
#define GST_USE_FILE_AUDIO 0

#endif // GST_CONFIG_H
