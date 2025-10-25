#!/usr/bin/env python3
"""
CPU-optimized inference script for GGUF model on VPS.
This is the most efficient way to run your model on CPU-only systems.
"""

import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_gguf_model():
    """Load GGUF model for CPU inference."""
    print("🦙 Loading GGUF model for CPU inference...")

    gguf_path = "qwen3-gguf-q4km.gguf"
    if not os.path.exists(gguf_path):
        print(f"❌ GGUF model not found: {gguf_path}")
        print("Run conversion first: python convert_to_gguf.py")
        return None

    try:
        from llama_cpp import Llama

        # CPU-optimized settings
        llm = Llama(
            model_path=gguf_path,
            n_ctx=2048,              # Context window
            n_threads=-1,            # Use all CPU cores
            n_batch=512,             # Batch size for processing
            n_gpu_layers=0,          # 0 = CPU only (no GPU)
            verbose=False,           # Reduce output
            seed=42                  # For reproducible results
        )

        print("✅ GGUF model loaded successfully!")
        print(f"   • Context: {llm.n_ctx()} tokens")
        print(f"   • Threads: {llm.n_threads} CPU cores")
        print(f"   • Model size: ~{os.path.getsize(gguf_path) / (1024**3):.1f} GB")
        return llm

    except ImportError:
        print("❌ llama-cpp-python not installed")
        print("Install with: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def format_chat_prompt(system_msg, user_msg):
    """Format messages for Qwen chat template."""
    return f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""

def chat_with_gguf(llm):
    """Interactive chat with GGUF model."""
    print("\n💬 CPU-Powered Financial Analyst Chat")
    print("🦙 Running on GGUF (optimized for CPU)")
    print("-" * 50)

    system_message = """You are an elite financial analyst with decades of experience in financial markets.
You provide comprehensive, deeply analytical responses that synthesize technical analysis, fundamental data, macroeconomic factors, and market sentiment."""

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        # Format prompt
        prompt = format_chat_prompt(system_message, user_input)

        # Generate response
        print("🤖 Analyzing...", end="", flush=True)
        start_time = time.time()

        try:
            output = llm(
                prompt=prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                echo=False
            )

            response_time = time.time() - start_time
            response = output["choices"][0]["text"].strip()

            print(f"\r🤖 Qwen3: {response}")
            print(f"⏱️  Response time: {response_time:.2f} seconds")
        except Exception as e:
            print(f"\r❌ Error: {e}")

def benchmark_gguf(llm):
    """Benchmark the GGUF model performance."""
    print("\n⚡ Benchmarking GGUF model performance...")

    test_prompts = [
        "What are the key factors affecting stock market volatility?",
        "Explain the RSI indicator and its significance in trading.",
        "How does inflation impact investment portfolios?"
    ]

    total_time = 0
    total_tokens = 0

    for i, prompt in enumerate(test_prompts, 1):
        full_prompt = format_chat_prompt(
            "You are a financial analyst.",
            prompt
        )

        print(f"🧪 Test {i}/3: {prompt[:30]}...")

        start_time = time.time()
        output = llm(
            prompt=full_prompt,
            max_tokens=128,
            temperature=0.7,
            echo=False
        )
        end_time = time.time()

        response = output["choices"][0]["text"]
        tokens_generated = len(llm.tokenize(response.encode()))
        time_taken = end_time - start_time

        print(f"   Response: {time_taken:.2f}s, {tokens_generated} tokens")
        total_time += time_taken
        total_tokens += tokens_generated

    avg_time = total_time / len(test_prompts)
    avg_tokens = total_tokens / len(test_prompts)
    tokens_per_sec = avg_tokens / avg_time

    print(f"\n📊 Benchmark Results:")
    print(f"  • Average response time: {avg_time:.2f} seconds")
    print(f"  • Average tokens generated: {avg_tokens:.0f}")
    print(f"  • Token generation speed: {tokens_per_sec:.1f} tokens/second")
    print(f"  • Model size: {os.path.getsize('qwen3-gguf-q4km.gguf') / (1024**3):.1f} GB")
    
    def main():
    """Main function."""
    import sys

    llm = load_gguf_model()
    if not llm:
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_gguf(llm)
    else:
        chat_with_gguf(llm)

if __name__ == "__main__":
    main()
