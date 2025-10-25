# 🤖 Qwen3-14B Financial Analyst Fine-tuning

Complete plug-and-play training pipeline for fine-tuning OpenPipe/Qwen3-14B-Instruct on financial data using Unsloth for maximum efficiency on H100 GPUs.

## 📋 Prerequisites

- **GPU**: H100 or A100 with at least 80GB VRAM
- **CUDA**: 12.1+ compatible drivers
- **Python**: 3.10+
- **RAM**: 128GB+ recommended
- **Storage**: 200GB+ free space

## 🚀 Quick Start (5 minutes setup)

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd llmtrain

# Run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅ Create Python virtual environment
- ✅ Install all dependencies with CUDA support
- ✅ Prompt for HuggingFace and WandB API keys
- ✅ Verify GPU compatibility
- ✅ Test API connections

### 2. Start Training
```bash
# Activate environment
source venv/bin/activate

# Start training (will use all available GPU power)
python3 train_qwen3.py
```

## 📊 Training Configuration

### Datasets
- **Primary**: `training_data.jsonl` (4,962 expert financial conversations)
- **Secondary**: `secondary_data.jsonl` (50,000 market news Q&A pairs)
- **Combined**: 54,962 training examples

### Model Configuration
- **Base Model**: OpenPipe/Qwen3-14B-Instruct
- **Fine-tuning**: LoRA adapters (r=16, alpha=16)
- **Quantization**: 4-bit during training, Q4_K_M final
- **Sequence Length**: 2048 tokens
- **Batch Size**: 2 (gradient accumulation: 4, effective: 8)

### Performance Optimizations
- ✅ **Unsloth**: 2x faster training, 60% less memory
- ✅ **Flash Attention**: Optimized attention for H100
- ✅ **Gradient Checkpointing**: Maximum memory efficiency
- ✅ **BF16/FP16**: Automatic precision selection
- ✅ **DeepSpeed**: Optional distributed training

## 🎯 Training Commands

### Full Pipeline (Training + Quantization)
```bash
python3 train_qwen3.py  # Trains and saves LoRA adapters
python3 quantize_model.py  # Quantizes to Q4_K_M
```

### Standalone Commands
```bash
# Setup environment only
python3 setup_env.py

# Training only
python3 train_qwen3.py

# Quantization only
python3 quantize_model.py --model_path qwen3-lora --output_path qwen3-q4km
```

## 📈 Monitoring & Logging

### Weights & Biases (WandB)
- **Project**: `qwen3-finetune` (configurable in .env)
- **Metrics**: Loss, learning rate, GPU memory, temperature
- **Artifacts**: Model checkpoints, training logs

### GPU Monitoring
```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f outputs/training.log
```

## 📁 Output Structure

```
outputs/
├── qwen3-lora/           # LoRA adapters
├── qwen3-q4km/           # Quantized model (Q4_K_M)
├── checkpoints/          # Training checkpoints
└── wandb/               # WandB logs
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key

# Optional
WANDB_PROJECT=qwen3-finetune
WANDB_ENTITY=your_username
```

### Training Parameters
Edit `train_qwen3.py` to modify:
- `max_steps`: Number of training steps
- `learning_rate`: Learning rate (default: 2e-4)
- `batch_size`: Per-device batch size
- `gradient_accumulation_steps`: Gradient accumulation

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in train_qwen3.py
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

**WandB Login Issues**
```bash
# Re-run setup
python3 setup_env.py
```

**HuggingFace Download Limits**
```bash
# Check token permissions at https://huggingface.co/settings/tokens
# Ensure model access is granted
```

### Performance Tuning

**For H100 (80GB)**
- Batch size: 2-4
- Gradient accumulation: 4-8
- Max sequence length: 2048-4096

**For A100 (40GB)**
- Batch size: 1-2
- Gradient accumulation: 8-16
- Max sequence length: 1024-2048

## 📋 Data Format

Training data uses OpenAI Chat Completion format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an elite Financial Analyst..."
    },
    {
      "role": "user",
      "content": "What is your market analysis?"
    },
    {
      "role": "assistant",
      "content": "Based on current data..."
    }
  ]
}
```

## 🔄 Reproducing Training

1. **Data Preparation**: Training data is pre-cleaned and ready
2. **Environment**: Setup script handles all dependencies
3. **Training**: Single command starts optimized training
4. **Quantization**: Automatic conversion to Q4_K_M format

## 📊 Expected Performance

### Training Time (H100)
- **50K examples**: ~4-6 hours
- **Throughput**: ~100-150 tokens/second
- **Memory usage**: 60-70GB peak

### Model Performance
- **Perplexity**: Expected improvement from 8-12 to 4-6
- **Response quality**: Enhanced financial analysis capabilities
- **Inference speed**: 50-100 tokens/second (quantized)

## 🤝 Contributing

1. Test on your H100 setup
2. Report any issues with GPU compatibility
3. Suggest performance optimizations
4. Add additional financial datasets

## 📞 Support

- Check WandB logs for training metrics
- Monitor GPU usage with `nvidia-smi`
- Review training logs in `outputs/`
- Test quantized model inference locally

---

**🎯 Ready to train? Just run `./setup.sh` and `python3 train_qwen3.py`!**