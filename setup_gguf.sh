#!/bin/bash
# Setup script for GGUF conversion dependencies

echo "🦙 Setting up GGUF conversion tools..."
echo "====================================="

# Check if we're on the server
if [ "$PWD" != "/workspace/llmtrain" ]; then
    echo "⚠️  Warning: Not running in expected directory"
    echo "Expected: /workspace/llmtrain"
    echo "Current: $PWD"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install llama-cpp-python (convert functionality is built-in now)
echo "📦 Installing llama-cpp-python..."
pip install llama-cpp-python

if [ $? -eq 0 ]; then
    echo "✅ llama-cpp-python installed successfully"
else
    echo "❌ Failed to install llama-cpp-python"
    echo "💡 Try: pip install --upgrade pip setuptools wheel"
    exit 1
fi

# Install additional conversion tools
echo "📦 Installing huggingface_hub for model conversion..."
pip install huggingface_hub

# Optional: Install llama.cpp from source using CMake
echo
echo "🔧 Installing llama.cpp from source (recommended)..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp

    # Use CMake instead of make (new build system)
    mkdir build && cd build
    cmake .. -DLLAMA_CURL=ON
    make -j$(nproc)
    cd ../..

else
    echo "llama.cpp directory already exists, skipping clone"
fi

# Verify installation
echo
echo "🔍 Verifying installation..."
python3 -c "
try:
    from llama_cpp import convert_hf_to_gguf
    print('✅ llama-cpp-python convert module working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)

import os
convert_script = 'llama.cpp/convert_hf_to_gguf.py'
if os.path.exists(convert_script):
    print('✅ llama.cpp convert script found')
else:
    print('⚠️  llama.cpp convert script not found (optional)')
"

echo
echo "🎉 Setup complete!"
echo
echo "🚀 You can now run:"
echo "   python convert_to_gguf.py"
echo "   python infer_gguf_cpu.py"
