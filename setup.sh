#!/bin/bash
# Complete setup script for LLM training environment
# Run this after cloning the repository

set -e  # Exit on any error

echo "🤖 LLM Training Setup Script"
echo "============================"

# Check if we're in the right directory
if [ ! -f "training_data.jsonl" ] || [ ! -f "secondary_data.jsonl" ]; then
    echo "❌ Error: Please run this script from the llmtrain directory containing training_data.jsonl"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; then
    echo "✅ PyTorch installed successfully"
else
    echo "❌ PyTorch installation failed"
    echo "💡 Try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Install core dependencies in stages to handle dependencies properly
echo "📚 Installing core dependencies..."

# Stage 1: Basic packages
pip install python-dotenv tqdm psutil GPUtil jupyter ipykernel

# Stage 2: ML core packages
echo "🤖 Installing ML core packages..."
if pip install transformers datasets accelerate peft huggingface_hub safetensors; then
    echo "✅ ML core packages installed"
else
    echo "❌ ML core packages installation failed"
    exit 1
fi

# Stage 3: Training and quantization (torch-dependent)
echo "🎯 Installing training packages..."
if pip install unsloth[colab-new] unsloth_zoo bitsandbytes auto-gptq; then
    echo "✅ Training packages installed"
else
    echo "❌ Training packages installation failed"
    exit 1
fi

# Stage 4: Data processing
echo "📊 Installing data processing packages..."
pip install pandas numpy || echo "⚠️  Data processing packages installation had issues, continuing..."

# Stage 5: Experiment tracking
echo "📈 Installing experiment tracking..."
pip install wandb || echo "⚠️  WandB installation failed, continuing..."

# Stage 6: Advanced features (optional)
echo "🔧 Installing advanced features..."
pip install deepspeed || echo "⚠️  DeepSpeed installation failed, continuing..."

# Stage 7: Flash Attention (may fail on some systems)
echo "⚡ Installing flash attention..."
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed, continuing without it..."

# Setup environment variables
echo "🔑 Setting up environment variables..."
python3 setup_env.py

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
print('🔍 Testing imports...')
try:
    import torch
    print('✅ PyTorch imported successfully')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA device: {torch.cuda.get_device_name()}')
        print(f'   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        print('   ⚠️  CUDA not available - will use CPU (much slower)')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    exit(1)

try:
    import transformers
    print('✅ Transformers imported successfully')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')

try:
    import unsloth
    print('✅ Unsloth imported successfully')
except ImportError as e:
    print(f'⚠️  Unsloth import failed: {e}')

try:
    import wandb
    print('✅ WandB imported successfully')
except ImportError as e:
    print(f'⚠️  WandB import failed: {e}')

try:
    import huggingface_hub
    print('✅ HuggingFace Hub imported successfully')
except ImportError as e:
    print(f'⚠️  HuggingFace Hub import failed: {e}')

print('🎉 Core components verified!')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start training, run:"
echo "   source venv/bin/activate"
echo "   python3 train_qwen3.py"
echo ""
echo "📚 Available commands:"
echo "   • Train model: python3 train_qwen3.py"
echo "   • Quantize model: python3 quantize_model.py"
echo "   • Setup environment: python3 setup_env.py"
echo ""
echo "💡 Tips:"
echo "   • Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "   • Check training progress on WandB"
echo "   • Model outputs will be saved to outputs/ directory"
