#!/usr/bin/env python3
"""
Standalone quantization script for Qwen3 models.
Converts trained model to Q4_K_M quantization format.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

load_dotenv()

def quantize_to_q4km(model_path="qwen3-lora", output_path="qwen3-q4km"):
    """Quantize model to Q4_K_M format."""
    print("⚡ Starting Q4_K_M quantization...")

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found!")

    # Load tokenizer
    print("📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=os.getenv('HF_TOKEN')
    )

    # Setup quantization config
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        model_name_or_path=model_path,
        model_file_base_name="model"
    )

    # Load model for quantization
    print("🤖 Loading model for quantization...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        token=os.getenv('HF_TOKEN')
    )

    # Create calibration dataset (small sample for quantization)
    print("🔧 Creating calibration dataset...")
    from datasets import load_dataset

    # Load a small sample from our training data for calibration
    try:
        calib_dataset = load_dataset("json", data_files="training_data.jsonl", split="train[:100]")
        print(f"Using {len(calib_dataset)} examples for calibration")
    except:
        print("⚠️  Could not load calibration data, using synthetic examples")
        # Create synthetic examples
        synthetic_examples = [
            "You are an elite Financial Analyst. What is the current market trend?",
            "Analyze the impact of interest rate changes on stock valuations.",
            "How do macroeconomic factors influence investment decisions?"
        ] * 10  # Repeat to get enough examples
        calib_dataset = [{"text": ex} for ex in synthetic_examples]

    # Format examples for quantization
    examples = []
    for example in calib_dataset[:100]:  # Limit to 100 examples
        if "messages" in example:
            # Format chat messages
            messages = example["messages"]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            examples.append(formatted)
        elif "text" in example:
            examples.append(example["text"])

    print(f"📊 Using {len(examples)} calibration examples")

    # Perform quantization
    print("🔄 Quantizing model...")
    model.quantize(
        examples=examples,
        batch_size=1,
        use_triton=True,
        autotune_warmup_after_quantized=False
    )

    # Save quantized model
    print(f"💾 Saving quantized model to {output_path}...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    # Verify quantization
    print("✅ Verifying quantization...")
    from auto_gptq import exllama_set_max_input_length
    model = exllama_set_max_input_length(model, 4096)

    # Test inference
    test_input = "You are an elite Financial Analyst. What is your analysis of the current market?"
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7
        )

    test_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🧪 Test inference: {test_output[:200]}...")

    # Calculate model size
    model_size_mb = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    ) / (1024 * 1024)

    print("✅ Quantization complete!")
    print(f"📊 Model size: {model_size_mb:.1f} MB")
    print(f"🎯 Quantization: Q4_K_M (4-bit with group size 128)")
    print(f"📁 Saved to: {output_path}")

    return output_path

def main():
    """Main quantization function."""
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Qwen3 model to Q4_K_M")
    parser.add_argument("--model_path", default="qwen3-lora",
                       help="Path to trained model (default: qwen3-lora)")
    parser.add_argument("--output_path", default="qwen3-q4km",
                       help="Output path for quantized model (default: qwen3-q4km)")

    args = parser.parse_args()

    try:
        quantize_to_q4km(args.model_path, args.output_path)
        print("\n🎉 Quantization successful!")
        print("🚀 Your model is ready for inference!")

    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        raise

if __name__ == "__main__":
    main()
