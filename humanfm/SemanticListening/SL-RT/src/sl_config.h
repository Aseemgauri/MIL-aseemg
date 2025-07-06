#ifndef SL_CONFIG_H
#define SL_CONFIG_H

/*
 * Semantic Listening Client Configuration
 * 
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY!
 * Generated from system.conf by generate_configs.py
 * 
 * To change configuration values, edit system.conf and run:
 * python3 generate_configs.py
 */

#define SYSTEM_SAMPLE_RATE 16000
#define SYSTEM_NUM_CLASSES 5
#define SL_JACK_CLIENT_NAME "SH-audio-RT"
#define SL_GAIN_COMPENSATION_FACTOR 8.0
#define SL_MAX_JACK_BUFFER_SIZE 4096
#define SL_INPUT_PORTS 2
#define SL_OUTPUT_PORTS 2
#define SL_SOCKET_PATH "/tmp/humanfm.sock"
#define SL_ENABLE_PASSTHROUGH 0
#define SL_ENABLE_INFERENCE 1
#define SL_FILL_EMBEDDING 0

#endif // SL_CONFIG_H
