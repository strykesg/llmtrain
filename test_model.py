#!/usr/bin/env python3
"""
Quick inference script to test the quantized Qwen3 model.
"""

import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel

# Load environment variables
load_dotenv()

def load_model():
    """Load the quantized model for inference."""
    print("🤖 Loading quantized Qwen3 model...")

    try:
        # Check if quantized model exists
        if not os.path.exists("qwen3-quantized-q4km"):
            print("❌ Quantized model not found! Run quantization first.")
            return None, None

        hf_token = os.getenv('HF_TOKEN')
        print(f"🔑 Using HF token: {'Present' if hf_token else 'None'}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-quantized-q4km",
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization
            token=hf_token
        )

        # Setup chat template
        try:
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
            print("✅ Chat template loaded")
        except Exception as e:
            print(f"⚠️  Chat template failed: {e}")

        print("✅ Model loaded successfully!")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def chat_with_model(model, tokenizer, messages):
    """Generate a response from the model."""
    try:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response (remove the prompt)
        response = response[len(prompt):].strip()

        return response

    except Exception as e:
        return f"Error generating response: {e}"

def test_model():
    """Test the model with some sample prompts."""
    model, tokenizer = load_model()
    if not model or not tokenizer:
        return

    print("\n🧪 Testing model with sample prompts...\n")

    # Test prompts
    test_cases = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": "What are the key factors affecting stock market volatility?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an expert in technical analysis."},
                {"role": "user", "content": "Explain the significance of RSI in trading."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a financial advisor specializing in risk management."},
                {"role": "user", "content": "How should I diversify my investment portfolio?"}
            ]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"📝 Test {i}: {test_case['messages'][1]['content'][:50]}...")
        response = chat_with_model(model, tokenizer, test_case["messages"])
        print(f"🤖 Response: {response[:200]}...")
        print("-" * 80)

    print("\n✅ Model testing complete!")

def interactive_chat():
    """Interactive chat with the model."""
    model, tokenizer = load_model()
    if not model or not tokenizer:
        return

    print("\n💬 Interactive chat with Qwen3 (type 'quit' to exit)")
    print("-" * 50)

    system_message = "You are an elite financial analyst with decades of experience in markets. Provide comprehensive, analytical responses."

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]

        print("🤖 Thinking...", end="", flush=True)
        response = chat_with_model(model, tokenizer, messages)
        print(f"\r🤖 Qwen3: {response}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_chat()
    else:
        test_model()
        print("\n💡 For interactive chat, run: python test_model.py --interactive")
