#!/bin/bash

# Runtime test script for SL Model
# This script builds and runs the runtime test

echo "Building SL Model runtime test..."

# Build the project
make clean
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Check if model file exists
MODEL_PATH="./pretrained/tfgridnet.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please ensure the model file exists in the pretrained directory"
    exit 1
fi

echo "Running runtime test with model: $MODEL_PATH"
echo "This will run 1000 inference iterations and measure average time..."
echo ""

# Run the runtime test
./runtime_test "$MODEL_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "Runtime test completed successfully!"
    echo "Results are saved in runtime_test_results.txt"
else
    echo ""
    echo "Runtime test failed!"
    exit 1
fi 