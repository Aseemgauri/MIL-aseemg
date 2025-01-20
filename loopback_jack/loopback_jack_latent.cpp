#include <iostream>
#include <jack/jack.h>
#include <jack/ringbuffer.h>
#include <cstring>

using namespace std;

// Buffer size for the delay: 16 * 1024 samples
constexpr size_t BUFFER_SIZE = 16 * 1024;

int processCallback(jack_nframes_t nframes, void* arg) {
    struct PortsAndBuffer {
        jack_port_t* input_port;
        jack_port_t* output_port;
        jack_ringbuffer_t* ring_buffer;
    };

    auto* data = static_cast<PortsAndBuffer*>(arg);

    // Get input and output buffers
    float* input_buffer = static_cast<float*>(jack_port_get_buffer(data->input_port, nframes));
    float* output_buffer = static_cast<float*>(jack_port_get_buffer(data->output_port, nframes));
    
    // Write input samples to the ring buffer
    size_t write_size = nframes * sizeof(float);
    if (jack_ringbuffer_write_space(data->ring_buffer) >= write_size) {
        jack_ringbuffer_write(data->ring_buffer, reinterpret_cast<char*>(input_buffer), write_size);
    } else {
        cerr << "Ring buffer overflow! Skipping write.\n";
    }

    // Read samples from the ring buffer if sufficiently full
    size_t read_size = (BUFFER_SIZE - nframes) * sizeof(float);
    if (jack_ringbuffer_read_space(data->ring_buffer) >= read_size) {
        jack_ringbuffer_read(data->ring_buffer, reinterpret_cast<char*>(output_buffer), write_size);
    } else {
        // If there isn't enough data, output silence
        memset(output_buffer, 0, read_size);
    }


    return 0;
}

int main() {
    const char* client_name = "audio_delay";
    jack_client_t* client;
    jack_options_t options = JackNullOption;
    jack_status_t status;

    client = jack_client_open(client_name, options, &status);
    if (client == nullptr) {
        cerr << "Failed to connect to JACK server: ";
        if (status & JackServerFailed) cerr << "Server failed\n";
        return 1;
    }

    cout << "Connected to JACK server as client: " << client_name << "\n";

    // Create input and output ports
    jack_port_t* input_port = jack_port_register(client, "input", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    jack_port_t* output_port = jack_port_register(client, "output", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);

    if (!input_port || !output_port) {
        cerr << "Failed to create JACK ports\n";
        jack_client_close(client);
        return 1;
    }

    // Create a ring buffer for delay
    jack_ringbuffer_t* ring_buffer = jack_ringbuffer_create(BUFFER_SIZE * sizeof(float));
    if (!ring_buffer) {
        cerr << "Failed to create ring buffer\n";
        jack_client_close(client);
        return 1;
    }

    // Zero the ring buffer
    jack_ringbuffer_reset(ring_buffer);

    // Pass ports and ring buffer to the callback
    struct PortsAndBuffer {
        jack_port_t* input_port;
        jack_port_t* output_port;
        jack_ringbuffer_t* ring_buffer;
    } data = {input_port, output_port, ring_buffer};

    jack_set_process_callback(client, processCallback, &data);

    // Activate the JACK client
    if (jack_activate(client)) {
        cerr << "Failed to activate JACK client\n";
        jack_client_close(client);
        return 1;
    }
    cout << "JACK client activated\n";

    // Automatically connect input and output ports
    const char** input_ports = jack_get_ports(client, nullptr, nullptr, JackPortIsPhysical | JackPortIsOutput);
    const char** output_ports = jack_get_ports(client, nullptr, nullptr, JackPortIsPhysical | JackPortIsInput);

    if (!input_ports || !output_ports) {
        cerr << "No physical ports available\n";
        jack_client_close(client);
        return 1;
    }

    // Connect input to the input port
    cout << "Connecting " << input_ports[0] << " to " << jack_port_name(input_port) << "\n";
    if (jack_connect(client, input_ports[0], jack_port_name(input_port))) {
        cerr << "Failed to connect input port\n";
    }

    // Connect the single output port to both physical playback ports (LR channels)
    cout << "Connecting " << jack_port_name(output_port) << " to " << output_ports[0] << "\n";
    if (jack_connect(client, jack_port_name(output_port), output_ports[0])) {
        cerr << "Failed to connect output port 1\n";
    }

    cout << "Connecting " << jack_port_name(output_port) << " to " << output_ports[1] << "\n";
    if (jack_connect(client, jack_port_name(output_port), output_ports[1])) {
        cerr << "Failed to connect output port 2\n";
    }

    jack_free(input_ports);
    jack_free(output_ports);

    unsigned int sample_rate = jack_get_sample_rate(client);
    cout << "Sample rate: " << sample_rate << " Hz\n";

    cout << "Running... Press Ctrl+C to stop.\n";
    while (true) {}

    // Clean up
    jack_ringbuffer_free(ring_buffer);
    jack_client_close(client);
    cout << "JACK client closed\n";

    return 0;
}
