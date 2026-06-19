import argparse
import os
import sys
import json
import random
import subprocess
import yaml
from typing import List

def split_dataset(input_file: str, temp_dir: str, train_ratio: float = 0.9) -> int:
    """
    Reads the input dataset JSONL file, shuffles the rows, and splits it into
    train.jsonl and valid.jsonl inside the specified temp directory.
    
    Args:
        input_file (str): Path to the input JSONL file.
        temp_dir (str): Directory where train.jsonl and valid.jsonl will be saved.
        train_ratio (float): Ratio of training data to total data. Default is 0.9.
        
    Returns:
        int: Number of training examples.
    """
    print(f"Reading dataset from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Dataset file not found: {input_file}")
        
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = [line.strip() for line in infile if line.strip()]
        
    print(f"Total examples found: {len(lines)}")
    if len(lines) == 0:
        raise ValueError("The dataset is empty.")
        
    # Shuffle and split
    random.seed(42)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]
    
    os.makedirs(temp_dir, exist_ok=True)
    
    train_path = os.path.join(temp_dir, "train.jsonl")
    valid_path = os.path.join(temp_dir, "valid.jsonl")
    
    # mlx-lm also needs a test.jsonl file (can be same as valid or empty)
    test_path = os.path.join(temp_dir, "test.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as train_file:
        train_file.write("\n".join(train_lines) + "\n")
        
    with open(valid_path, "w", encoding="utf-8") as valid_file:
        valid_file.write("\n".join(valid_lines) + "\n")
        
    with open(test_path, "w", encoding="utf-8") as test_file:
        test_file.write("\n".join(valid_lines) + "\n")
        
    print(f"Dataset split completed:")
    print(f"  - Train path: {train_path} ({len(train_lines)} examples)")
    print(f"  - Valid path: {valid_path} ({len(valid_lines)} examples)")
    print(f"  - Test path: {test_path} ({len(valid_lines)} examples)")
    
    return len(train_lines)

def main() -> None:
    """
    Main entry point for the fine-tuning script.
    Parses args, splits data, builds config YAML, and invokes mlx_lm.lora.
    """
    parser = argparse.ArgumentParser(description="Wrapper for MLX QLoRA fine-tuning.")
    parser.add_argument("--model", type=str, required=True, help="Base model path or Hugging Face repo.")
    parser.add_argument("--data", type=str, required=True, help="Input JSONL file to split and train on.")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-targets", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="LoRA target layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--output", type=str, required=True, help="Output adapters directory.")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Use gradient checkpointing to reduce memory.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length.")
    args = parser.parse_args()
    
    # 1. Setup temp data directory
    temp_data_dir = "data/mlx_temp_data"
    num_train_examples = split_dataset(args.data, temp_data_dir)
    
    # 2. Compute iterations
    # iterations = (num_train_examples / batch_size) * epochs
    iters = int((num_train_examples / args.batch_size) * args.epochs)
    print(f"Calculated iterations for training: {iters} (examples: {num_train_examples}, batch size: {args.batch_size}, epochs: {args.epochs})")
    
    # 3. Map friendly targets to full Mistral-style sub-module paths
    mapped_keys = []
    mapping = {
        "q_proj": "self_attn.q_proj",
        "v_proj": "self_attn.v_proj",
        "k_proj": "self_attn.k_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj"
    }
    for target in args.lora_targets:
        if target in mapping:
            mapped_keys.append(mapping[target])
        else:
            mapped_keys.append(target)

    # Create temporary config YAML for lora_parameters
    temp_config_path = "data/mlx_temp_config.yaml"
    config_data = {
        "lora_parameters": {
            "rank": args.lora_rank,
            "scale": float(args.lora_alpha),
            "dropout": 0.0,
            "keys": mapped_keys
        }
    }
    
    os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
    with open(temp_config_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False)
    print(f"Saved temporary MLX training config to {temp_config_path}")
    
    # 4. Construct command to call mlx_lm lora
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", temp_data_dir,
        "--iters", str(iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", args.output,
        "--config", temp_config_path,
        "--seed", "42",
        "--val-batches", "50"
    ]
    
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training process
    result = subprocess.run(cmd)
    
    # Cleanup temp files if desired
    # We leave them so user can inspect if needed, or delete config
    if os.path.exists(temp_config_path):
        try:
            os.remove(temp_config_path)
        except Exception:
            pass
            
    if result.returncode != 0:
        print("MLX LoRA fine-tuning failed.")
        sys.exit(result.returncode)
    else:
        print("MLX LoRA fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
