#!/bin/bash
# Fix llama.cpp build issues and complete the quantization

echo "🔧 Fixing llama.cpp build for quantization..."
echo "=============================================="

# Go to llama.cpp directory
cd llama.cpp

# Clean previous build attempt
if [ -d "build" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build
fi

# Try with CURL disabled (simpler)
echo "🔨 Building llama.cpp without CURL (recommended for quantization)..."
mkdir build && cd build
cmake .. -DLLAMA_CURL=OFF  # Disable CURL to avoid dependency issues
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ llama.cpp built successfully!"
    echo "📍 quantize tool location: $(pwd)/bin/quantize"

    # Go back to llmtrain directory
    cd ../..

    # Now quantize the GGUF model
    echo
    echo "🔄 Quantizing GGUF model..."
    python quantize_gguf_only.py

else
    echo "❌ Build failed. Trying alternative approach..."

    # Alternative: Try installing curl-dev
    echo "📦 Installing curl development libraries..."
    apt-get update
    apt-get install -y libcurl4-openssl-dev

    # Try build again with CURL
    cd ..  # Go back to llama.cpp root
    rm -rf build
    mkdir build && cd build
    cmake .. -DLLAMA_CURL=ON
    make -j$(nproc)

    if [ $? -eq 0 ]; then
        echo "✅ llama.cpp built successfully with CURL!"
        cd ../..
        python quantize_gguf_only.py
    else
        echo "❌ Build still failed."
        echo
        echo "💡 Manual quantization:"
        echo "  1. Install dependencies: apt-get install libcurl4-openssl-dev cmake build-essential"
        echo "  2. cd llama.cpp && mkdir build && cd build"
        echo "  3. cmake .. -DLLAMA_CURL=ON && make -j\$(nproc)"
        echo "  4. cd ../.. && python quantize_gguf_only.py"
    fi
fi
