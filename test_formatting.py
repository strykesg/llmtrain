#!/usr/bin/env python3
"""
Test script to verify chat template formatting works correctly.
"""

import os
from dotenv import load_dotenv
from datasets import load_dataset
from unsloth import FastLanguageModel

load_dotenv()

def test_formatting():
    """Test the chat formatting function."""
    print("🧪 Testing chat formatting...")

    try:
        # Load a small sample of data
        dataset = load_dataset("json", data_files="training_data.jsonl", split="train[:5]")
        print(f"✅ Loaded {len(dataset)} sample entries")

        # Load tokenizer (lightweight)
        model_name = "OpenPipe/Qwen3-14B-Instruct"
        hf_token = os.getenv('HF_TOKEN')

        print("🤖 Loading tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            token=hf_token
        )

        # Test chat template
        try:
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
            print("✅ Chat template loaded")
        except Exception as e:
            print(f"⚠️  Chat template failed: {e}")
            return False

        # Test formatting function
        def format_conversation(examples):
            formatted_texts = []
            for messages in examples["messages"]:
                try:
                    # Try using chat template
                    formatted_text = tokenizer.apply_chat_template(
                        conversation=messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    formatted_texts.append(formatted_text)
                except Exception as e:
                    print(f"❌ Chat template failed: {e}")
                    return False
            return {"text": formatted_texts}

        # Test on sample data
        sample = dataset[:2]  # First 2 examples
        result = format_conversation(sample)

        if "text" in result and len(result["text"]) == 2:
            print("✅ Formatting function works!")
            print(f"   Sample formatted text length: {len(result['text'][0])}")
            return True
        else:
            print("❌ Formatting function returned unexpected result")
            return False

    except Exception as e:
        print(f"❌ Formatting test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_formatting()
    if success:
        print("🎉 Chat formatting is ready!")
    else:
        print("❌ Chat formatting needs fixes")
