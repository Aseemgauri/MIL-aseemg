#!/bin/bash

echo "🧹 Cleaning and building all components..."

echo "🔧 Generating configuration files..."
python3 common/generate_configs.py
if [ $? -eq 0 ]; then
    echo "✅ Configuration files generated successfully"
else
    echo "❌ Configuration generation failed"
    exit 1
fi

echo "📦 Building GST Client..."
cd ./GST_Client
make clean
make
if [ $? -eq 0 ]; then
    echo "✅ GST Client built successfully"
else
    echo "❌ GST Client build failed"
    exit 1
fi

echo "📦 Building Semantic Listening Client..."
cd ../SemanticListening/SL-RT
make clean
make
if [ $? -eq 0 ]; then
    echo "✅ SL Client built successfully"
else
    echo "❌ SL Client build failed"
    exit 1
fi

echo "🎯 All components built successfully!"
