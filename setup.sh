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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Setup environment variables
echo "🔑 Setting up environment variables..."
python3 setup_env.py

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import transformers
import unsloth
import wandb
import huggingface_hub
print('✅ All imports successful!')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
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
