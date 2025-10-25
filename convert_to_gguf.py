#!/usr/bin/env python3
"""
Convert the quantized model to GGUF format for efficient CPU inference.
GGUF is optimized for llama.cpp and works great on CPU-only VPS.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required tools are installed."""
    print("🔍 Checking requirements...")

    try:
        import transformers
        print("✅ transformers installed")
    except ImportError:
        print("❌ transformers not installed. Run: pip install transformers")
        return False

    # Check for llama.cpp convert script (new location)
    convert_script = "./llama.cpp/convert_hf_to_gguf.py"
    if os.path.exists(convert_script):
        print("✅ llama.cpp convert script found")
        return convert_script

    # Check for alternative Python conversion
    try:
        import llama_cpp.convert
        print("✅ llama-cpp-python convert module found")
        return "python_convert"
    except ImportError:
        pass

    print("❌ No conversion tools found")
    print("\n📦 Install with:")
    print("  bash setup_gguf.sh")
    return False

def convert_to_gguf(model_path, output_path):
    """Convert model to GGUF format."""
    print(f"🔄 Converting {model_path} to GGUF...")

    try:
        # Method 1: Use llama.cpp convert script (most efficient)
        convert_tool = check_requirements()
        if convert_tool and convert_tool != "python_convert" and convert_tool.endswith("convert_hf_to_gguf.py"):
            # Use llama.cpp convert script with correct arguments
            cmd = [
                sys.executable, convert_tool,
                model_path,  # Input model path
                output_path   # Output GGUF path
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ GGUF conversion successful!")

                # Now quantize to Q4_K_M using llama.cpp quantize tool
                quantize_path = "./llama.cpp/build/bin/quantize"  # CMake build location
                if not os.path.exists(quantize_path):
                    quantize_path = "./llama.cpp/quantize"  # Fallback

                if os.path.exists(quantize_path):
                    quantized_output = output_path.replace(".gguf", "-Q4_K_M.gguf")
                    quantize_cmd = [
                        quantize_path,
                        output_path,
                        quantized_output,
                        "Q4_K_M"
                    ]

                    print(f"Quantizing: {' '.join(quantize_cmd)}")
                    quantize_result = subprocess.run(quantize_cmd, capture_output=True, text=True)

                    if quantize_result.returncode == 0:
                        print("✅ Quantization successful!")
                        # Replace original file with quantized version
                        os.rename(quantized_output, output_path)
                        return True
                    else:
                        print(f"⚠️  Conversion successful but quantization failed: {quantize_result.stderr}")
                        return True  # Still return True since conversion worked
                else:
                    print("⚠️  Conversion successful but quantize tool not found")
                    print("💡 You can quantize later with: ./llama.cpp/build/bin/quantize")
                    return True
            else:
                print(f"❌ Conversion failed: {result.stderr}")
                return False

        # Method 2: Use llama-cpp-python converter
        try:
            from llama_cpp import convert_hf_to_gguf
            print("Using llama-cpp-python converter...")

            convert_hf_to_gguf.convert_hf_to_gguf(
                model_path=model_path,
                output_path=output_path,
                quantize="Q4_K_M"
            )

            print("✅ GGUF conversion successful!")
            return True

        except ImportError:
            print("❌ llama-cpp-python not installed")

        # Method 3: Manual conversion using transformers
        print("Falling back to manual conversion...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,  # Fixed: use dtype instead of deprecated torch_dtype
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # This is a simplified conversion - in practice you'd want to use llama.cpp tools
        print("⚠️  Manual conversion is limited. Consider installing llama.cpp for proper conversion.")
        print("💡 For CPU VPS hosting, GGUF format is still the most efficient option.")

        return False

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def test_gguf_model(gguf_path):
    """Test the converted GGUF model."""
    print(f"🧪 Testing GGUF model: {gguf_path}")

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,
            n_threads=4,  # Adjust based on your CPU cores
            verbose=False
        )

        # Test prompt
        prompt = """You are a financial analyst. Explain what affects stock market volatility in 2-3 sentences."""

        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            echo=False
        )

        response = output["choices"][0]["text"].strip()
        print(f"🤖 Test response: {response[:200]}...")
        print("✅ GGUF model working!")

        return True

    except ImportError:
        print("❌ llama-cpp-python not installed for testing")
        print("Install with: pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main conversion workflow."""
    print("🦙 Converting Qwen3 to GGUF format")
    print("=" * 40)

    # Determine input model
    lora_path = "qwen3-lora"
    quantized_path = "qwen3-quantized-q4km"

    if os.path.exists(quantized_path):
        input_model = quantized_path
        print(f"📂 Using quantized model: {input_model}")
    elif os.path.exists(lora_path):
        input_model = lora_path
        print(f"📂 Using LoRA model: {input_model}")
    else:
        print("❌ No model found! Run training first.")
        return

    # Output path
    output_gguf = "qwen3-gguf-q4km.gguf"

    # Convert to GGUF
    if convert_to_gguf(input_model, output_gguf):
        print(f"📁 GGUF model saved as: {output_gguf}")

        # Test the model
        test_gguf_model(output_gguf)

        print("\n🎉 Conversion complete!")
        print("📊 Efficiency comparison:")
        print("  • GGUF: Best for CPU VPS (optimized inference)")
        print("  • 4-bit: Good for GPU, but needs CUDA")
        print("  • Full precision: Slowest, needs lots of RAM")
    else:
        print("\n❌ Conversion failed. See errors above.")
        print("\n💡 Alternative: Use the quantized model with transformers:")
        print("  • Still efficient for GPU inference")
        print("  • Works with vLLM, Text Generation Inference, etc.")

if __name__ == "__main__":
    main()
