#!/usr/bin/env python3
"""
Test script to diagnose WandB configuration issues.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_wandb_config():
    """Test WandB configuration and permissions."""
    print("🧪 Testing WandB Configuration")
    print("=" * 40)

    # Check environment variables
    wandb_key = os.getenv('WANDB_API_KEY')
    wandb_project = os.getenv('WANDB_PROJECT', 'qwen3-finetune')
    wandb_entity = os.getenv('WANDB_ENTITY', os.getenv('USER', 'default'))

    print(f"WandB API Key: {'Present' if wandb_key else 'Missing'}")
    print(f"WandB Project: {wandb_project}")
    print(f"WandB Entity: {wandb_entity}")
    print()

    if not wandb_key:
        print("❌ WANDB_API_KEY not found in environment")
        print("   💡 Run: python3 setup_env.py")
        return False

    # Test WandB login
    print("🔑 Testing WandB login...")
    try:
        import wandb
        wandb.login(key=wandb_key, relogin=True)
        print("✅ WandB login successful")
    except Exception as e:
        print(f"❌ WandB login failed: {e}")
        print("   💡 Check your API key at: https://wandb.ai/settings")
        return False

    # Test project access
    print("📊 Testing project access...")
    try:
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name="test-run",
            config={"test": True}
        )
        print("✅ Project access successful")
        run.finish()
        print("✅ Test run completed")
        return True
    except Exception as e:
        print(f"❌ Project access failed: {e}")
        print()
        print("🔧 Troubleshooting steps:")
        print("1. Check if project exists: https://wandb.ai/home")
        print(f"2. Verify entity '{wandb_entity}' has access to project '{wandb_project}'")
        print("3. Try a different project name or create a new project")
        print("4. Update .env file with correct WANDB_PROJECT and WANDB_ENTITY")
        return False

def suggest_fixes():
    """Suggest fixes for common WandB issues."""
    print("\n🔧 Quick Fixes:")
    print("1. Update .env file:")
    print("   WANDB_PROJECT=your-project-name")
    print("   WANDB_ENTITY=your-username-or-team")
    print()
    print("2. Disable WandB temporarily:")
    print("   Remove WANDB_API_KEY from .env or comment it out")
    print()
    print("3. Create new WandB project:")
    print("   Visit: https://wandb.ai/new")
    print()
    print("4. Check API key permissions:")
    print("   Visit: https://wandb.ai/settings")

if __name__ == "__main__":
    success = test_wandb_config()
    if not success:
        suggest_fixes()
    print("\n" + "=" * 40)
    if success:
        print("🎉 WandB is ready for training!")
    else:
        print("⚠️  Fix WandB issues before training")
