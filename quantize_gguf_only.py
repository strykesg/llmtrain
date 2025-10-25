#!/usr/bin/env python3
"""
Quantize an existing GGUF model to Q4_K_M format.
Run this after GGUF conversion if quantization failed.
"""

import os
import subprocess
import sys

def quantize_gguf():
    """Quantize existing GGUF model to Q4_K_M."""
    gguf_path = "qwen3-gguf-q4km.gguf"

    if not os.path.exists(gguf_path):
        print(f"❌ GGUF model not found: {gguf_path}")
        print("Run conversion first: python convert_to_gguf.py")
        return False

    # Check file size to see if already quantized
    size_gb = os.path.getsize(gguf_path) / (1024**3)
    if size_gb < 10:  # Likely already quantized
        print(".1f"        print("✅ Model appears to already be quantized!")
        return True

    print(".1f"    # Find quantize tool
    quantize_paths = [
        "./llama.cpp/build/bin/quantize",
        "./llama.cpp/build/quantize",
        "./llama.cpp/quantize",
        "/usr/local/bin/quantize",
    ]

    quantize_path = None
    for path in quantize_paths:
        if os.path.exists(path):
            quantize_path = path
            print(f"✅ Found quantize tool: {path}")
            break

    if not quantize_path:
        print("❌ quantize tool not found!")
        print("\n🔧 Build llama.cpp with quantization:")
        print("  cd llama.cpp")
        print("  mkdir build && cd build")
        print("  cmake .. -DLLAMA_CURL=ON")
        print("  make -j$(nproc)")
        print("  # quantize will be in: build/bin/quantize")
        return False

    # Quantize the model
    quantized_output = gguf_path.replace(".gguf", "-Q4_K_M.gguf")
    quantize_cmd = [
        quantize_path,
        gguf_path,
        quantized_output,
        "Q4_K_M"
    ]

    print(f"🔄 Quantizing: {' '.join(quantize_cmd)}")
    result = subprocess.run(quantize_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        # Replace original with quantized version
        os.rename(quantized_output, gguf_path)

        new_size_gb = os.path.getsize(gguf_path) / (1024**3)
        print("✅ Quantization successful!")
        print(".1f"
        return True
    else:
        print(f"❌ Quantization failed: {result.stderr}")
        return False

if __name__ == "__main__":
    success = quantize_gguf()
    if success:
        print("\n🎉 GGUF model is now quantized and ready for CPU inference!")
        print("Run: python infer_gguf_cpu.py")
    else:
        print("\n❌ Quantization failed. Check error messages above.")
