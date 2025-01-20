#include <iostream>
#include <jack/jack.h>
using namespace std;

float y01 = 0;
float x01 = 0;
// JACK callback function for audio processing
int processCallback(jack_nframes_t nframes, void* arg) {
    jack_port_t** ports = reinterpret_cast<jack_port_t**>(arg);
    float* input_buffer = static_cast<float*>(jack_port_get_buffer(ports[0], nframes));
    float* output_buffer = static_cast<float*>(jack_port_get_buffer(ports[1], nframes));
    // Copy input buffer to output buffer
    for (jack_nframes_t i = 0; i < nframes; ++i) {
        float x = input_buffer[i];
        float y = (0.93662412 * y01) + (0.03168794 * x) + (0.03168794 * x01);
        x01 = x;
        y01 = y;
        output_buffer[i] = y;
    }

    return 0;
}

int main() {
    const char* client_name = "audio_passthrough";
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
    jack_port_t* output_port_1 = jack_port_register(client, "output_1", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    jack_port_t* output_port_2 = jack_port_register(client, "output_2", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);

    if (!input_port || !output_port_1 || !output_port_2) {
        cerr << "Failed to create JACK ports\n";
        jack_client_close(client);
        return 1;
    }

    // Pass input and one output port to the callback
    jack_port_t* ports[] = {input_port, output_port_1};
    jack_set_process_callback(client, processCallback, ports);

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

    // Connect output ports to both physical outputs
    cout << "Connecting " << jack_port_name(output_port_1) << " to " << output_ports[0] << "\n";
    if (jack_connect(client, jack_port_name(output_port_1), output_ports[0])) {
        cerr << "Failed to connect output port 1\n";
    }

    cout << "Connecting " << jack_port_name(output_port_1) << " to " << output_ports[1] << "\n";
    if (jack_connect(client, jack_port_name(output_port_1), output_ports[1])) {
        cerr << "Failed to connect output port 2\n";
    }

    // Free port lists
    jack_free(input_ports);
    jack_free(output_ports);

    cout << "Running... Press Ctrl+C to stop.\n";
    while (true) {}

    jack_client_close(client);
    cout << "JACK client closed\n";
    return 0;
}
