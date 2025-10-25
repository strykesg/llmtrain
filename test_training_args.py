#!/usr/bin/env python3
"""
Test script to verify TrainingArguments work correctly.
"""

import os
from dotenv import load_dotenv
from transformers import TrainingArguments

load_dotenv()

def test_training_args():
    """Test that TrainingArguments can be created without errors."""
    print("🧪 Testing TrainingArguments creation...")

    try:
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=True,  # Simplified for testing
            bf16=False,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Disable wandb for testing
            run_name="test-run",

            # Memory optimization
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,

            # Monitoring
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=False,
        )

        print("✅ TrainingArguments created successfully!")
        print(f"   Epochs: {training_args.num_train_epochs}")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Output dir: {training_args.output_dir}")

        return True

    except Exception as e:
        print(f"❌ TrainingArguments creation failed: {e}")
        return False

def test_sft_trainer_import():
    """Test that SFTTrainer can be imported."""
    print("🧪 Testing SFTTrainer import...")

    try:
        from trl import SFTTrainer
        print("✅ SFTTrainer imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ SFTTrainer import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Training Components Test")
    print("=" * 40)

    args_ok = test_training_args()
    print()
    trainer_ok = test_sft_trainer_import()
    print()

    if args_ok and trainer_ok:
        print("🎉 All training components verified!")
    else:
        print("❌ Some components failed - check errors above")
