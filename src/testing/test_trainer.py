import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig
from src.data.data_loader import DataConfig, EfficientDataset, collate_fn
import pytest

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_trainer():
    # Create a simple DataConfig for testing
    config = DataConfig(
        dataset_name="c4",
        dataset_config_name="en",
        text_column="text",
        max_length=128,    # reduced max length for testing
        batch_size=2,
        cache_size=10
    )
    
    # Initialize the tokenizer for GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model (using GPT-2)
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        output_hidden_states=True,
        pad_token_id=tokenizer.pad_token_id
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=model_config)
    model.to("cpu")  # for testing, place on CPU
    
    # Instantiate the training dataset
    train_dataset = EfficientDataset(config, tokenizer, split="train")
    
    # Define training arguments (simplified for testing)
    training_args = TrainingArguments(
        output_dir="test_trainer_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        do_train=True,
        evaluation_strategy="no",
        fp16=False
    )
    
    print("\nInitializing Trainer with custom collate function...")
    # Initialize the Trainer with our custom collate function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer
    )
    
    print("\nGetting dataloader from Trainer...")
    # Instead of full training, get one batch from the Trainer's dataloader
    train_dataloader = trainer.get_train_dataloader()
    
    print("\nFetching one batch from Trainer's dataloader:")
    batch = next(iter(train_dataloader))
    print("\nBatch information:")
    print("Keys:", list(batch.keys()))
    for key, tensor in batch.items():
        print(f"\n{key}:")
        print(f"  shape = {tensor.shape}")
        print(f"  type = {type(tensor)}")
        print(f"  device = {tensor.device}")
        if isinstance(tensor, torch.Tensor):
            print(f"  dtype = {tensor.dtype}")
            print(f"  requires_grad = {tensor.requires_grad}")

@pytest.fixture
def data_config():
    return DataConfig(
        dataset_name="c4",
        dataset_config_name="en",
        max_length=128,
        batch_size=2,
        cache_size=10
    )

if __name__ == "__main__":
    test_trainer() 