#!/usr/bin/env python3
"""
Inference test script for trained and quantized Qwen3 model.
Tests both LoRA and quantized versions.
"""

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer
import os

load_dotenv()

def test_lora_model():
    """Test the LoRA fine-tuned model."""
    print("🧪 Testing LoRA model...")

    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-lora",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(model)

        test_message = [
            {"role": "system", "content": "You are an elite Financial Analyst with decades of experience in financial markets. You provide comprehensive, deeply analytical responses that synthesize technical analysis, fundamental data, macroeconomic factors, and market sentiment."},
            {"role": "user", "content": "What is your analysis of the current stock market conditions?"}
        ]

        inputs = tokenizer.apply_chat_template(
            test_message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ LoRA model test successful!")
        print(f"📝 Response: {response[-200:]}...")

        return True

    except Exception as e:
        print(f"❌ LoRA model test failed: {e}")
        return False

def test_quantized_model():
    """Test the quantized Q4_K_M model."""
    print("🧪 Testing quantized model...")

    try:
        from auto_gptq import AutoGPTQForCausalLM, exllama_set_max_input_length

        model = AutoGPTQForCausalLM.from_quantized(
            "qwen3-q4km",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_triton=True
        )

        tokenizer = AutoTokenizer.from_pretrained("qwen3-q4km")
        model = exllama_set_max_input_length(model, 4096)

        test_message = "You are an elite Financial Analyst. Analyze the impact of interest rate changes on tech stocks."

        inputs = tokenizer(test_message, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✅ Quantized model test successful!")
        print(f"📝 Response: {response[len(test_message):][:200]}...")

        return True

    except Exception as e:
        print(f"❌ Quantized model test failed: {e}")
        return False

def benchmark_inference():
    """Benchmark inference speed."""
    print("⚡ Benchmarking inference speed...")

    try:
        from auto_gptq import AutoGPTQForCausalLM, exllama_set_max_input_length
        import time

        model = AutoGPTQForCausalLM.from_quantized(
            "qwen3-q4km",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            use_triton=True
        )

        tokenizer = AutoTokenizer.from_pretrained("qwen3-q4km")
        model = exllama_set_max_input_length(model, 4096)

        prompt = "You are an elite Financial Analyst. What are the key factors influencing market volatility?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        end_time = time.time()

        generated_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        time_taken = end_time - start_time
        tokens_per_second = generated_tokens / time_taken

        print("📊 Benchmark Results:")
        print(f"   Generated tokens: {generated_tokens}")
        print(f"   Time taken: {time_taken:.2f}s")
        print(f"   Tokens/second: {tokens_per_second:.1f}")

        return True

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running inference tests for Qwen3 models")
    print("=" * 50)

    # Check GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu} ({memory:.1f}GB)")
    else:
        print("⚠️  No GPU detected - using CPU")

    # Test LoRA model
    lora_ok = test_lora_model()
    print()

    # Test quantized model
    quantized_ok = test_quantized_model()
    print()

    # Benchmark
    if quantized_ok:
        benchmark_inference()
        print()

    # Summary
    print("📋 Test Summary:")
    print(f"   LoRA model: {'✅ PASS' if lora_ok else '❌ FAIL'}")
    print(f"   Quantized model: {'✅ PASS' if quantized_ok else '❌ FAIL'}")

    if lora_ok and quantized_ok:
        print("\n🎉 All tests passed! Your models are ready for inference.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
