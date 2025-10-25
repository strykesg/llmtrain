#!/usr/bin/env python3
"""
Fine-tuning script for OpenPipe/Qwen3-14B-Instruct using Unsloth.
Optimized for H100 GPU with maximum efficiency and monitoring.
"""

import os
import torch
import wandb
from dotenv import load_dotenv
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
import psutil
import GPUtil
from datetime import datetime

# Load environment variables
load_dotenv()

# Global WandB state
wandb_enabled = False

def setup_environment():
    """Setup training environment and validate requirements."""
    print("🚀 Setting up training environment...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. This script requires a GPU.")

    # Get GPU info
    gpu = GPUtil.getGPUs()[0]
    print(f"🎮 GPU: {gpu.name}")
    print(f"📊 GPU Memory: {gpu.memoryTotal}MB")
    print(f"🧠 CUDA Version: {torch.version.cuda}")

    # Set memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.backends.cudnn.benchmark = True

    # Login to WandB
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("✅ WandB login successful")
    else:
        print("⚠️  WandB API key not found - logging disabled")

    # Login to HuggingFace
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"🔑 HF_TOKEN found (length: {len(hf_token)})")
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ HuggingFace login successful")
    else:
        print("⚠️  HuggingFace token not found - may have download limits")
        print("💡 Make sure .env file exists with HF_TOKEN=your_token_here")

def load_training_data():
    """Load training datasets."""
    print("📚 Loading training data...")

    # Load only primary data (our custom dataset)
    primary_data = load_dataset("json", data_files="training_data.jsonl", split="train")
    print(f"📊 Training data: {len(primary_data)} examples")

    return primary_data

def setup_model():
    """Setup Qwen3 model with Unsloth for efficient training."""
    print("🤖 Loading Qwen3-14B-Instruct model...")

    model_name = "OpenPipe/Qwen3-14B-Instruct"

    # Load model with Unsloth
    hf_token = os.getenv('HF_TOKEN')
    print(f"🤖 Loading model with token: {'Present' if hf_token else 'None'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,  # Adjust based on your data
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        token=hf_token
    )

    # Setup chat template
    try:
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
        print("✅ Chat template loaded: qwen-2.5")
    except Exception as e:
        print(f"⚠️  Chat template loading failed: {e}")
        print("   Using default chat template")
        # Fallback to basic chat template
        try:
            tokenizer = get_chat_template(tokenizer, chat_template="chatml")
            print("✅ Fallback chat template loaded: chatml")
        except:
            print("⚠️  Fallback chat template also failed")
            # Continue without chat template - will use basic formatting

    # Add LoRA adapters for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"🏗️  Model loaded with {model.num_parameters():,} parameters")
    print(f"🎯 Trainable parameters: {model.num_parameters(only_trainable=True):,}")

    return model, tokenizer

def train_model(model, tokenizer, dataset):
    """Train the model with optimized settings for H100."""
    print("🎯 Starting training...")

    # Setup WandB
    wandb_project = os.getenv('WANDB_PROJECT', 'qwen3-finetune')
    wandb_entity = os.getenv('WANDB_ENTITY', os.getenv('USER', 'anonymous'))

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        per_device_train_batch_size=2,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=5,
        num_train_epochs=3,  # Train for 3 full epochs
        # max_steps=200,  # Removed - using epochs instead
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="wandb" if os.getenv('WANDB_API_KEY') else "none",
        run_name=f"qwen3-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",

        # Memory optimization
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,

        # Monitoring
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    # Initialize WandB with error handling
    global wandb_enabled
    wandb_enabled = False
    if os.getenv('WANDB_API_KEY'):
        try:
            print(f"📊 Initializing WandB (project: {wandb_project}, entity: {wandb_entity})...")
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "model": "OpenPipe/Qwen3-14B-Instruct",
                    "dataset_size": len(dataset),
                    "batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation": training_args.gradient_accumulation_steps,
                    "learning_rate": training_args.learning_rate,
                    "num_train_epochs": training_args.num_train_epochs,
                    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                }
            )
            wandb_enabled = True
            print("✅ WandB initialized successfully")
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            print("   Continuing without WandB logging...")
            print("   💡 Check your WANDB_API_KEY and project permissions")
            training_args.report_to = "none"  # Disable WandB reporting
    else:
        print("ℹ️  WandB API key not found - logging disabled")

    # Formatting function for chat templates
    def format_conversation(examples):
        # Unsloth expects a list of processed strings directly
        messages_list = examples["messages"]

        # Check if this is batched (list of lists) or single (list of dicts)
        if isinstance(messages_list[0], list):
            # Batched: messages_list is a list of conversations
            formatted_texts = []
            for messages in messages_list:
                try:
                    # Try using chat template
                    formatted_text = tokenizer.apply_chat_template(
                        conversation=messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception as e:
                    print(f"⚠️  Chat template failed for message, using manual formatting: {e}")
                    # Fallback to manual formatting
                    formatted_parts = []
                    for msg in messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            formatted_parts.append(f"System: {content}")
                        elif role == "user":
                            formatted_parts.append(f"Human: {content}")
                        elif role == "assistant":
                            formatted_parts.append(f"Assistant: {content}")
                    formatted_text = "\n\n".join(formatted_parts)

                formatted_texts.append(formatted_text)
            return formatted_texts  # Return list directly for Unsloth
        else:
            # Single conversation: messages_list is a list of message dicts
            messages = messages_list
            try:
                # Try using chat template
                formatted_text = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"⚠️  Chat template failed for message, using manual formatting: {e}")
                # Fallback to manual formatting
                formatted_parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        formatted_parts.append(f"System: {content}")
                    elif role == "user":
                        formatted_parts.append(f"Human: {content}")
                    elif role == "assistant":
                        formatted_parts.append(f"Assistant: {content}")
                formatted_text = "\n\n".join(formatted_parts)

            return [formatted_text]  # Return as list for consistency

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can set to True for better efficiency
        args=training_args,
        formatting_func=format_conversation,
    )

    # Monitor GPU usage
    def log_gpu_usage():
        gpu = GPUtil.getGPUs()[0]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        memory_percent = (memory_used / memory_total) * 100

        if wandb_enabled:
            wandb.log({
                "gpu_memory_used_mb": memory_used,
                "gpu_memory_percent": memory_percent,
                "gpu_temperature": gpu.temperature
            })

        print(f"🎮 GPU Memory: {memory_used}/{memory_total}MB ({memory_percent:.1f}%)")

    # Start training with monitoring
    print("🔥 Training started...")
    log_gpu_usage()

    trainer_stats = trainer.train()

    log_gpu_usage()
    print("✅ Training completed!")
    print(f"📊 Training stats: {trainer_stats}")

    return trainer

def save_model(trainer, tokenizer):
    """Save the trained model."""
    print("💾 Saving model...")

    # Save LoRA adapters
    trainer.model.save_pretrained("qwen3-lora")
    tokenizer.save_pretrained("qwen3-lora")

    print("✅ Model saved to 'qwen3-lora' directory")

    return "qwen3-lora"

def quantize_model(model_path):
    """Quantize the model to Q4_K_M using Unsloth."""
    print("⚡ Quantizing model to Q4_K_M using Unsloth...")

    try:
        from unsloth import FastLanguageModel
        import torch

        # Load the trained model with LoRA adapters
        print("🔄 Loading trained model with LoRA adapters...")
        hf_token = os.getenv('HF_TOKEN')

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-lora",  # Load from our saved LoRA model
            token=hf_token
        )

        # Merge LoRA adapters to base model for quantization
        print("🔀 Merging LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Same as training
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # Save merged model temporarily
        print("💾 Saving merged model...")
        model.save_pretrained_merged("qwen3-merged", tokenizer, save_method="merged_16bit")

        # Now quantize the merged model
        print("🔢 Loading merged model for quantization...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="qwen3-merged",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=hf_token
        )

        # Save quantized model
        model.save_pretrained("qwen3-quantized-q4km")
        tokenizer.save_pretrained("qwen3-quantized-q4km")
        print("✅ Quantized model saved to 'qwen3-quantized-q4km'")

    except Exception as e:
        print(f"⚠️  Unsloth quantization failed: {e}")
        print("💡 Consider using GGUF quantization or skipping quantization for now")
        print("   Your LoRA adapters are saved in 'qwen3-lora' and ready to use!")

        # Alternative: Just save the LoRA model without quantization
        print("📦 Skipping quantization - LoRA model is ready for inference")
        return "qwen3-lora"  # Return LoRA path instead

def main():
    """Main training pipeline."""
    try:
        # Setup
        setup_environment()
        dataset = load_training_data()
        model, tokenizer = setup_model()

        # Train
        trainer = train_model(model, tokenizer, dataset)
        model_path = save_model(trainer, tokenizer)

        # Quantize
        quantized_path = quantize_model(model_path)

        print("\n🎉 Training complete!")
        print("📁 Outputs:")
        print("  - LoRA adapters: qwen3-lora/")
        if quantized_path == "qwen3-lora":
            print("  - ⚠️  Quantization skipped - use LoRA model for inference")
        else:
            print("  - Quantized model: qwen3-quantized-q4km/")

        if wandb_enabled:
            wandb.finish()

    except Exception as e:
        print(f"❌ Error during training: {e}")
        if wandb_enabled:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()
