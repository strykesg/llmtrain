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

def merge_lora_for_gguf():
    """Merge LoRA weights back to full-precision model for GGUF conversion."""
    print("🔀 Merging LoRA weights for GGUF compatibility...")

    try:
        from unsloth import FastLanguageModel
        import torch

        # Load base model and LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="OpenPipe/Qwen3-14B-Instruct",
            max_seq_length=2048,
            load_in_4bit=False,  # Load in full precision for merging
            token=os.getenv('HF_TOKEN')
        )

        # Apply LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Load the trained LoRA weights
        model.load_adapter("qwen3-lora", "default")

        # Merge and save in full precision
        print("💾 Saving merged full-precision model...")
        model.save_pretrained_merged("qwen3-merged", tokenizer, save_method="merged_16bit")

        print("✅ LoRA merging completed!")
        return True

    except Exception as e:
        print(f"❌ LoRA merging failed: {e}")
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
                "--outfile", output_path,  # Output file flag
                "--outtype", "f16",        # Convert to f16 first
                model_path                 # Input model path
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ GGUF conversion successful!")

                # Now quantize to Q4_K_M using llama.cpp quantize tool
                # Check multiple possible quantize tool locations
                quantize_paths = [
                    "./llama.cpp/build/bin/quantize",      # CMake build
                    "./llama.cpp/build/quantize",           # Alternative CMake
                    "./llama.cpp/quantize",                 # Direct build
                    "/usr/local/bin/quantize",              # System install
                ]

                quantize_path = None
                for path in quantize_paths:
                    if os.path.exists(path):
                        quantize_path = path
                        print(f"✅ Found quantize tool: {path}")
                        break

                if quantize_path:
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
                    print("\n🔧 To build the quantize tool:"                    print("  cd llama.cpp")
                    print("  # If using CMake (recommended):")
                    print("  mkdir build && cd build")
                    print("  cmake .. -DLLAMA_CURL=ON")
                    print("  make -j$(nproc)")
                    print("  # quantize will be in: build/bin/quantize")
                    print()
                    print("  # Then quantize manually:")
                    print("  ./build/bin/quantize ../qwen3-gguf-q4km.gguf ../qwen3-gguf-q4km-Q4_K_M.gguf Q4_K_M")
                    print("  mv qwen3-gguf-q4km-Q4_K_M.gguf qwen3-gguf-q4km.gguf")
                    print()
                    print("📄 Your GGUF model is still usable (just larger)")
                    print("   File: qwen3-gguf-q4km.gguf")
                    print("   Size: ~14GB (f16), can be quantized later")
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

    # Determine input model - prefer merged model for GGUF conversion
    merged_path = "qwen3-merged"
    lora_path = "qwen3-lora"
    quantized_path = "qwen3-quantized-q4km"

    if os.path.exists(merged_path):
        input_model = merged_path
        print(f"📂 Using merged model: {input_model}")
    elif os.path.exists(lora_path):
        # Need to merge LoRA first for GGUF conversion
        print("🔄 LoRA model found - merging for GGUF conversion...")
        if merge_lora_for_gguf():
            input_model = "qwen3-merged"
            print(f"📂 Using newly merged model: {input_model}")
        else:
            print("❌ LoRA merging failed!")
            return
    elif os.path.exists(quantized_path):
        print("⚠️  Quantized model found but not compatible with GGUF (bitsandbytes)")
        print("💡 Need LoRA model for GGUF conversion")
        print("💡 Run training again or use the existing LoRA model")
        return
    else:
        print("❌ No compatible model found! Run training first.")
        print("💡 GGUF conversion needs LoRA model (qwen3-lora/)")
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
