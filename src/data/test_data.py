import unittest
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer

from src.data.data_loader import DataConfig, EfficientDataset, create_dataloaders, DynamicBatchSampler

class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DataConfig(
            dataset_name="c4",
            dataset_config_name="en",
            text_column="text",
            max_length=512,
            batch_size=2,  # Hard cap at 2 for memory constraints
            cache_size=100,
            num_workers=2,
            training_phase="c4",  # Start with C4
            curriculum_switch_epoch=5  # Switch to CoT at epoch 5
        )
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    def test_dataset_initialization(self):
        dataset = EfficientDataset(
            self.config,
            self.tokenizer,
            split="train"
        )
        self.assertGreater(len(dataset), 0)
        
        # Test single item retrieval
        item = dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        
    def test_curriculum_switch(self):
        # Test C4 phase
        self.assertEqual(self.config.training_phase, "c4")
        train_loader, val_loader = create_dataloaders(
            self.config,
            self.tokenizer
        )
        
        # Test CoT phase
        self.config.training_phase = "cot"
        train_loader_cot, val_loader_cot = create_dataloaders(
            self.config,
            self.tokenizer
        )
        
        # Verify we get different datasets
        self.assertNotEqual(
            train_loader.dataset.dataset.info.dataset_name,
            train_loader_cot.dataset.dataset.info.dataset_name
        )
        
    def test_dynamic_batch_sampler(self):
        dataset = EfficientDataset(
            self.config,
            self.tokenizer,
            split="train"
        )
        
        sampler = DynamicBatchSampler(
            dataset,
            batch_size=self.config.batch_size,
            drop_last=False,
            max_memory_usage=0.65,
            min_batch_size=1,
            check_memory_every=2
        )
        
        # Verify batch indices
        batch_indices = next(iter(sampler))
        self.assertIsInstance(batch_indices, list)
        self.assertLessEqual(len(batch_indices), self.config.batch_size)
        
        # Test memory monitoring
        stats = sampler.get_statistics()
        self.assertIn("memory_rss_mean", stats)
        self.assertIn("batch_size_mean", stats)

if __name__ == "__main__":
    unittest.main() 