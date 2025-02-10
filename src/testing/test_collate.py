import logging
import torch
from transformers import AutoTokenizer
from src.data.data_loader import DataConfig, EfficientDataset, collate_fn

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_collate():
    # Create a simple DataConfig for testing
    config = DataConfig(
        dataset_name="c4",
        dataset_config_name="en",
        text_column="text",
        max_length=128,    # reduced max length for testing
        batch_size=2,
        cache_size=10      # reduced cache size for testing
    )
    
    # Initialize the tokenizer for GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Instantiate the dataset (using the train split)
    dataset = EfficientDataset(config, tokenizer, split="train")
    
    # Get a few samples from the dataset
    samples = [dataset[idx] for idx in range(4)]
    print("\nIndividual sample keys and shapes:")
    for i, sample in enumerate(samples):
        print(f"\nSample {i} keys: {list(sample.keys())}")
        for key in sample:
            print(f"  {key} shape: {sample[key].shape}")
            print(f"  {key} type: {type(sample[key])}")
            print(f"  {key} device: {sample[key].device}")
    
    # Use our custom collate function on these samples
    print("\nAttempting to batch samples...")
    batched = collate_fn(samples)
    print("\nBatched output:")
    print(f"Keys: {list(batched.keys())}")
    for key in batched:
        print(f"  {key} shape: {batched[key].shape}")
        print(f"  {key} type: {type(batched[key])}")
        print(f"  {key} device: {batched[key].device}")

if __name__ == "__main__":
    test_collate() 