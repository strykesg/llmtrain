#!/usr/bin/env python3
"""
Environment setup script for LLM training.
Prompts for API keys and creates .env file.
"""

import os
import getpass
from pathlib import Path

def setup_environment():
    """Interactive setup of environment variables."""
    print("🤖 LLM Training Environment Setup")
    print("=" * 50)

    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        response = input(".env file already exists. Overwrite? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Setup cancelled.")
            return

    print("\n📋 Required API Keys:")
    print("1. HuggingFace Token (for model downloads/uploads)")
    print("   Get from: https://huggingface.co/settings/tokens")
    print("2. Weights & Biases API Key (for experiment tracking)")
    print("   Get from: https://wandb.ai/settings")

    print("\n🔑 Please enter your API keys:")

    # Get HuggingFace token
    hf_token = getpass.getpass("HuggingFace Token: ").strip()
    if not hf_token:
        print("❌ HuggingFace token is required!")
        return

    # Get WandB API key
    wandb_key = getpass.getpass("Weights & Biases API Key: ").strip()
    if not wandb_key:
        print("❌ WandB API key is required!")
        return

    # Optional: WandB project name
    wandb_project = input("WandB Project Name (default: qwen3-finetune): ").strip()
    if not wandb_project:
        wandb_project = "qwen3-finetune"

    # Optional: WandB entity
    wandb_entity = input("WandB Entity/Username (default: your username): ").strip()
    if not wandb_entity:
        # Try to get from system
        wandb_entity = os.environ.get('USER', 'your_username')

    # Create .env file
    env_content = f"""# HuggingFace API Token (for model downloads and uploads)
HF_TOKEN={hf_token}

# Weights & Biases API Key (for experiment tracking)
WANDB_API_KEY={wandb_key}

# WandB Project Name
WANDB_PROJECT={wandb_project}

# WandB Entity (your username or team)
WANDB_ENTITY={wandb_entity}
"""

    try:
        with open('.env', 'w') as f:
            f.write(env_content)

        print("\n✅ Environment setup complete!")
        print("📄 .env file created with your API keys")
        print("🔒 Keys are stored locally and will not be committed to git")

        # Test the keys
        print("\n🧪 Testing API keys...")

        # Test HF token
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            user = api.whoami()
            print(f"✅ HuggingFace: Logged in as {user['name']}")
        except Exception as e:
            print(f"⚠️  HuggingFace test failed: {e}")

        # Test WandB
        try:
            import wandb
            wandb.login(key=wandb_key)
            print("✅ Weights & Biases: Login successful")
        except Exception as e:
            print(f"⚠️  WandB test failed: {e}")

    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

if __name__ == "__main__":
    setup_environment()
