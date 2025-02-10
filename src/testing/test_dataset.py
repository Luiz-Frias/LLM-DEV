import logging
from transformers import AutoTokenizer
from src.data.data_loader import DataConfig, EfficientDataset

# Setup basic logging so we can see debug/info messages on the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_dataset():
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
    
    # Iterate over the first few samples and print their results
    for idx in range(3):
        sample = dataset[idx]
        print(f"\nSample {idx} keys: {list(sample.keys())}")
        for key, tensor in sample.items():
            print(f"  {key} shape: {tensor.shape}")
            print(f"  {key} type: {type(tensor)}")
            print(f"  {key} device: {tensor.device}")

if __name__ == "__main__":
    test_dataset() 