#!/usr/bin/env python3
"""
Simple GGUF conversion using llama-cpp-python.
Easier to install than full llama.cpp.
"""

import os
import sys

def convert_with_llama_cpp():
    """Convert using llama-cpp-python convert function."""
    print("🦙 Converting to GGUF using llama-cpp-python...")

    try:
        from llama_cpp.convert_hf_to_gguf import convert_hf_to_gguf
    except ImportError:
        print("❌ llama-cpp-python[convert] not installed")
        print("Run setup first: bash setup_gguf.sh")
        return False

    # Determine input model
    if os.path.exists("qwen3-quantized-q4km"):
        input_model = "qwen3-quantized-q4km"
    elif os.path.exists("qwen3-lora"):
        input_model = "qwen3-lora"
    else:
        print("❌ No model found to convert!")
        return False

    output_file = "qwen3-gguf-q4km.gguf"

    print(f"📂 Converting: {input_model}")
    print(f"📝 Output: {output_file}")

    try:
        # Convert to GGUF
        convert_hf_to_gguf(
            model_path=input_model,
            output_path=output_file,
            outtype="f16",  # Base format, we'll quantize later
            # quantize="Q4_K_M"  # Let llama.cpp quantize it
        )

        print("✅ GGUF conversion completed!")

        # Check file size
        size_gb = os.path.getsize(output_file) / (1024**3)
        print(".1f"
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_conversion():
    """Test the converted model."""
    gguf_path = "qwen3-gguf-q4km.gguf"

    if not os.path.exists(gguf_path):
        print(f"❌ GGUF file not found: {gguf_path}")
        return False

    print("🧪 Testing converted model...")

    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=gguf_path,
            n_ctx=512,  # Small context for testing
            n_threads=2,
            verbose=False
        )

        # Simple test
        prompt = "You are a financial analyst. What affects stock prices?"
        output = llm(prompt, max_tokens=64, temperature=0.7, echo=False)

        response = output["choices"][0]["text"].strip()
        print("✅ Test successful!"        print(f"🤖 Response: {response[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main conversion function."""
    print("🦙 Qwen3 to GGUF Converter (Simple)")
    print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Just test existing conversion
        test_conversion()
        return

    # Convert model
    if convert_with_llama_cpp():
        print("\n🎉 Conversion complete!")
        print("📁 Model saved as: qwen3-gguf-q4km.gguf"

        # Test the conversion
        if test_conversion():
            print("\n✅ Everything working!")
            print("\n🚀 Run CPU inference:")
            print("   python infer_gguf_cpu.py")
        else:
            print("\n⚠️  Conversion completed but testing failed")
    else:
        print("\n❌ Conversion failed")
        print("\n💡 Make sure to run: bash setup_gguf.sh")

if __name__ == "__main__":
    main()
