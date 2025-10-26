"""
Fine-tune GPT-OSS-120B for employee analytics predictions.

This script supports distributed training across multiple DGX Spark systems.
"""

import os
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse

def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def load_model_and_tokenizer(model_name, use_lora=True):
    """
    Load model and tokenizer with optional LoRA for efficient fine-tuning.
    
    Args:
        model_name: HuggingFace model name or path
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if use_lora:
        print("Applying LoRA configuration...")
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def preprocess_function(examples, tokenizer, max_length=2048):
    """Preprocess training examples into tokenized format."""
    
    def format_messages(messages):
        """Format messages into a single string."""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted += f"<|system|>\n{content}\n"
            elif role == 'user':
                formatted += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                formatted += f"<|assistant|>\n{content}\n"
        formatted += tokenizer.eos_token
        return formatted
    
    # Format all messages
    texts = [format_messages(msg) for msg in examples['messages']]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-OSS-120B for employee analytics')
    parser.add_argument('--model_name', type=str, default='openai/gpt-oss-120b',
                       help='Model name or path')
    parser.add_argument('--train_file', type=str, default='data/train_dataset.jsonl',
                       help='Path to training data')
    parser.add_argument('--val_file', type=str, default='data/val_dataset.jsonl',
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='./employee_analytics_model',
                       help='Output directory for model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        print("="*70)
        print("FINE-TUNING GPT-OSS-120B FOR EMPLOYEE ANALYTICS")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Model: {args.model_name}")
        print(f"  Training file: {args.train_file}")
        print(f"  Validation file: {args.val_file}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Use LoRA: {args.use_lora}")
        print(f"  World size: {world_size}")
        print()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_lora)
    
    # Load datasets
    if rank == 0:
        print("Loading datasets...")
    
    train_dataset = load_dataset('json', data_files=args.train_file, split='train')
    val_dataset = load_dataset('json', data_files=args.val_file, split='train')
    
    if rank == 0:
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Validation examples: {len(val_dataset)}")
    
    # Preprocess datasets
    if rank == 0:
        print("\nPreprocessing datasets...")
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=False,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=False,
        remove_columns=val_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="none",  # Disable wandb/tensorboard for now
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    if rank == 0:
        print("\nStarting training...")
        print("="*70)
    
    trainer.train()
    
    # Save final model
    if rank == 0:
        print("\nSaving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"✓ Model saved to {args.output_dir}")
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

if __name__ == "__main__":
    main()

