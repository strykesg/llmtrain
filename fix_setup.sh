#!/bin/bash
# Fix setup script for common installation issues

echo "🔧 LLM Training Setup Fix Script"
echo "================================="

# Check if we're in the right directory
if [ ! -f "training_data.jsonl" ]; then
    echo "❌ Error: Please run this script from the llmtrain directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

source venv/bin/activate
echo "✅ Activated virtual environment"

# Try alternative PyTorch installation if needed
echo "🔥 Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing PyTorch with alternative CUDA version..."
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Fix flash-attn installation
echo "⚡ Fixing flash-attn installation..."
pip uninstall flash-attn -y 2>/dev/null

# Try different installation methods
echo "Method 1: Installing with build isolation disabled..."
if pip install flash-attn --no-build-isolation; then
    echo "✅ Flash-attn installed successfully"
else
    echo "Method 2: Installing from source with CUDA flags..."
    pip install flash-attn --no-build-isolation --global-option="--cuda_ext"
fi

# Fix any remaining dependency issues
echo "🔧 Checking for missing dependencies..."
pip check || echo "⚠️  Some dependency conflicts detected, but continuing..."

# Re-run verification
echo "✅ Re-verifying installation..."
python3 -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('   GPU:', torch.cuda.get_device_name())
except ImportError as e:
    print('❌ PyTorch issue:', e)

try:
    import unsloth
    print('✅ Unsloth available')
except ImportError as e:
    print('❌ Unsloth issue:', e)

try:
    import flash_attn
    print('✅ Flash Attention available')
except ImportError:
    print('⚠️  Flash Attention not available (optional)')
"

echo ""
echo "🎉 Fix script completed!"
echo "🚀 Try running training now: python3 train_qwen3.py"
