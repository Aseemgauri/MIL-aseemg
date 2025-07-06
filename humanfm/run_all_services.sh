#!/bin/bash

# Script to run all services concurrently in the specified order
# Order: generate_configs.py -> jack_server_start.sh -> jack_add_sc.sh -> node_server.sh -> sl_client.sh -> gst_client.sh

echo "Starting all services in sequence..."

# First, generate all configuration files (SYNCHRONOUS - must complete before services start)
echo "0. Generating configuration files..."
./set_configs.sh
if [ $? -ne 0 ]; then
    echo "❌ Config generation failed! Aborting service startup."
    exit 1
fi
echo "✅ Configuration files generated successfully"
echo ""

# Start jack_server_start.sh first
echo "1. Starting jack_server_start.sh..."
./jack_server_start.sh &
JACK_SERVER_PID=$!

# Wait 500ms before starting the next service
sleep 0.5

# Start jack_add_sc.sh
echo "2. Starting jack_add_sc.sh..."
./jack_add_sc.sh &
JACK_ADD_PID=$!

# Wait 500ms before starting the next service
sleep 0.5

# Start node_server.sh
echo "3. Starting node_server.sh..."
./node_server.sh &
NODE_SERVER_PID=$!

# Wait 500ms before starting the next service
sleep 0.5

# Start sl_client.sh
echo "4. Starting sl_client.sh..."
./sl_client.sh &
SL_CLIENT_PID=$!

# Wait 500ms before starting the next service
sleep 0.5

# Start gst_client.sh
echo "5. Starting gst_client.sh..."
./gst_client.sh &
GST_CLIENT_PID=$!

echo ""
echo "All services started successfully!"
echo "Configuration: ✅ Generated fresh configs"
echo "Process IDs:"
echo "  jack_server_start.sh: $JACK_SERVER_PID"
echo "  jack_add_sc.sh: $JACK_ADD_PID"
echo "  node_server.sh: $NODE_SERVER_PID"
echo "  sl_client.sh: $SL_CLIENT_PID"
echo "  gst_client.sh: $GST_CLIENT_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to handle cleanup when script is interrupted
cleanup() {
    echo ""
    echo "Stopping all services..."
    kill $JACK_SERVER_PID $JACK_ADD_PID $NODE_SERVER_PID $SL_CLIENT_PID $GST_CLIENT_PID 2>/dev/null
    killall jackd 2>/dev/null
    echo "All services stopped."
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGINT SIGTERM

# Wait for all background processes to complete
wait 