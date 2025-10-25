#!/usr/bin/env python3
"""
Simple inference script using LoRA model (if quantization fails).
"""

import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel

# Load environment variables
load_dotenv()

def test_lora_model():
    """Test the LoRA model directly."""
    print("🤖 Loading LoRA model for inference...")

    try:
        # Check if LoRA model exists
        if not os.path.exists("qwen3-lora"):
            print("❌ LoRA model not found!")
            return

        hf_token = os.getenv('HF_TOKEN')

        # Load base model and apply LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="OpenPipe/Qwen3-14B-Instruct",
            max_seq_length=2048,
            load_in_4bit=True,
            token=hf_token
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

        # Setup chat template
        try:
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        except:
            pass

        print("✅ LoRA model loaded!")

        # Quick test
        messages = [
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": "What affects stock prices?"}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        print(f"🧪 Test response: {response[:100]}...")
        print("✅ LoRA model working!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_lora_model()
