#!/usr/bin/env python3
"""
Test script to verify environment variables are loaded correctly.
"""

import os
from dotenv import load_dotenv

def test_environment():
    """Test environment variable loading."""
    print("🧪 Testing environment variable loading...")

    # Load environment variables
    load_dotenv()

    # Test HF_TOKEN
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"✅ HF_TOKEN: Present (length: {len(hf_token)})")
        print(f"   Starts with: {hf_token[:10]}..." if len(hf_token) > 10 else f"   Full token: {hf_token}")
    else:
        print("❌ HF_TOKEN: Not found")

    # Test WANDB_API_KEY
    wandb_key = os.getenv('WANDB_API_KEY')
    if wandb_key:
        print(f"✅ WANDB_API_KEY: Present (length: {len(wandb_key)})")
    else:
        print("⚠️  WANDB_API_KEY: Not found")

    # Test other env vars
    wandb_project = os.getenv('WANDB_PROJECT', 'qwen3-finetune')
    wandb_entity = os.getenv('WANDB_ENTITY', 'default')

    print(f"📊 WANDB_PROJECT: {wandb_project}")
    print(f"👤 WANDB_ENTITY: {wandb_entity}")

    # Check if .env file exists
    if os.path.exists('.env'):
        print("📄 .env file: Found")
    else:
        print("❌ .env file: Not found")

    # Test HF Hub login (if token exists)
    if hf_token:
        try:
            from huggingface_hub import login, HfApi
            login(token=hf_token)
            api = HfApi()
            user = api.whoami()
            print(f"🔗 HF Hub login: Successful (user: {user['name']})")
        except Exception as e:
            print(f"❌ HF Hub login: Failed ({e})")

if __name__ == "__main__":
    test_environment()
