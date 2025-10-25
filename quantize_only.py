#!/usr/bin/env python3
"""
Standalone quantization script for Qwen3 model.
Run this after training to quantize the LoRA model.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def quantize_model():
    """Quantize the model to Q4_K_M using Unsloth."""
    print("⚡ Quantizing model to Q4_K_M using Unsloth...")

    try:
        from unsloth import FastLanguageModel
        import torch

        # Check if LoRA model exists
        if not os.path.exists("qwen3-lora"):
            print("❌ LoRA model not found! Run training first.")
            return False

        # Load the trained model with LoRA adapters
        print("🔄 Loading trained model with LoRA adapters...")
        hf_token = os.getenv('HF_TOKEN')

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-lora",  # Load from our saved LoRA model
            token=hf_token
        )

        # Merge LoRA adapters to base model for quantization
        print("🔀 Merging LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Same as training
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Save merged model temporarily
        print("💾 Saving merged model...")
        model.save_pretrained_merged("qwen3-merged", tokenizer, save_method="merged_16bit")

        # Now quantize the merged model
        print("🔢 Loading merged model for quantization...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-merged",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=hf_token
        )

        # Save quantized model
        model.save_pretrained("qwen3-quantized-q4km")
        tokenizer.save_pretrained("qwen3-quantized-q4km")
        print("✅ Quantized model saved to 'qwen3-quantized-q4km'")
        return True

    except Exception as e:
        print(f"❌ Unsloth quantization failed: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure your LoRA model exists in 'qwen3-lora/'")
        print("  2. Check your HF_TOKEN is set correctly")
        print("  3. Try using GGUF quantization instead:")
        print("     pip install llama.cpp")
        print("     python -m llama.cpp.convert --model qwen3-lora --outtype f16")
        return False

if __name__ == "__main__":
    success = quantize_model()
    if success:
        print("\n🎉 Quantization complete! Model ready for deployment.")
    else:
        print("\n⚠️  Quantization failed, but your LoRA model is still usable!")
