#!/usr/bin/env python3
"""
Quick validation test for the GGUF model.
Tests basic functionality and shows response time.
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def quick_test():
    """Quick test of the GGUF model."""
    gguf_path = "qwen3-gguf-q4km.gguf"

    # Check if model exists
    if not os.path.exists(gguf_path):
        print("❌ GGUF model not found!")
        print("Run: python convert_to_gguf.py && python quantize_gguf_only.py")
        return False

    print("🧪 Quick Model Validation Test")
    print("=" * 40)

    try:
        from llama_cpp import Llama

        # Load model
        print("🔄 Loading GGUF model...")
        start_load = time.time()

        llm = Llama(
            model_path=gguf_path,
            n_ctx=1024,      # Smaller context for quick test
            n_threads=16,    # Use 16 threads
            verbose=False
        )

        load_time = time.time() - start_load
        print(".2f"
        # Test prompt
        test_prompt = """You are a financial analyst. In 2-3 sentences, explain what affects stock market volatility."""

        print(f"\n📝 Test Prompt: {test_prompt[:50]}...")
        print("\n🤖 Generating response...")

        # Generate response
        start_gen = time.time()

        output = llm(
            prompt=test_prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )

        gen_time = time.time() - start_gen
        response = output["choices"][0]["text"].strip()

        # Validation checks
        print("\n" + "=" * 40)
        print("📊 VALIDATION RESULTS:")
        print("=" * 40)

        # Check response quality
        has_financial_terms = any(term in response.lower() for term in [
            'volatility', 'market', 'stock', 'price', 'economic', 'factor', 'analysis'
        ])

        # Check response length
        word_count = len(response.split())
        is_reasonable_length = 20 <= word_count <= 200

        # Check for coherent response
        has_sentences = '.' in response and len(response.split('.')) >= 2

        print(f"✅ Model loaded: {load_time:.2f}s")
        print(f"✅ Response time: {gen_time:.2f}s")
        print(f"✅ Response length: {word_count} words")
        print(f"✅ Contains financial terms: {'✅' if has_financial_terms else '❌'}")
        print(f"✅ Coherent response: {'✅' if has_sentences else '❌'}")

        print("
🤖 RESPONSE:"        print("-" * 40)
        print(response)

        # Overall validation
        is_valid = (
            load_time < 30 and      # Loads within 30 seconds
            gen_time < 10 and       # Generates within 10 seconds
            has_financial_terms and # Contains relevant content
            is_reasonable_length and # Reasonable response length
            has_sentences           # Coherent response structure
        )

        print("\n" + "=" * 40)
        if is_valid:
            print("🎉 MODEL VALIDATION: PASSED ✅")
            print("   • Fast loading and inference")
            print("   • Relevant financial content")
            print("   • Coherent response structure")
            print("   • Ready for production use!")
        else:
            print("⚠️  MODEL VALIDATION: ISSUES DETECTED")
            print("   Check the issues above and model setup")

        return is_valid

    except ImportError:
        print("❌ llama-cpp-python not installed")
        print("Run: pip install llama-cpp-python")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
