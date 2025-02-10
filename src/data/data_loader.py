from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizer
from datasets import load_dataset, concatenate_datasets
import numpy as np
import logging
from multiprocessing import cpu_count
from .synthetic_data import SyntheticDataGenerator, integrate_synthetic_data
import os
import psutil

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    dataset_name: str = "c4"  # Default to C4 dataset
    dataset_config_name: Optional[str] = "en"
    text_column: str = "text"
    max_length: int = 512
    batch_size: int = 2  # Hard cap at batch size 2 for memory constraints
    cache_size: int = 1000
    max_examples: Optional[int] = None
    base_synthetic_ratio: float = 0.2  # Initial synthetic ratio
    final_synthetic_ratio: float = 0.9  # Final synthetic ratio after transition
    synthetic_ramp_start: float = 0.85  # Start increasing synthetic ratio at 85% of training
    min_thinking_steps: int = 3
    max_thinking_steps: int = 8
    num_workers: int = min(4, cpu_count())
    train_split: str = "train"
    val_split: str = "validation"
    sliding_window: bool = True
    stride: int = 512  # For sliding window
    # Batch size warmup parameters
    warmup_steps: int = 100
    warmup_factor: float = 0.5  # Start at batch size 1 (0.5 * 2)
    
    # New curriculum learning parameters
    training_phase: str = "c4"  # Indicates current training phase; 'c4' for general text, 'cot' for chain-of-thought
    curriculum_switch_epoch: Optional[int] = None  # Epoch at which to switch from C4 to CoT dataset
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback to default."""
        return getattr(self, key, default)

class EfficientDataset(Dataset):
    """Memory efficient dataset implementation."""
    def __init__(
        self,
        config: DataConfig,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        current_epoch: int = 0,
        total_epochs: int = 1,
        dataset_override: Optional[Any] = None  # Added parameter for curriculum override
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # Create cache directories
        os.makedirs("cache/datasets", exist_ok=True)
        os.makedirs("cache/synthetic", exist_ok=True)
        
        # Load dataset based on configuration or use the override if provided
        if dataset_override is not None:
            self.dataset = dataset_override
            self._dataset_iter = iter(self.dataset)
        else:
            logger.info(f"Loading {config.dataset_name} dataset...")
            try:
                # Load dataset in streaming mode
                self.dataset = load_dataset(
                    config.dataset_name,
                    config.dataset_config_name,
                    split=split,
                    streaming=True,
                    cache_dir="cache/datasets"
                )
                # Convert to iterator but keep reference to original dataset
                self._dataset_iter = iter(self.dataset)
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise
        
        # Initialize cache and synthetic data management
        self.cache = {}
        self.cache_size = config.cache_size
        self.current_size = 0
        self.synthetic_cache_file = f"cache/synthetic/epoch_{current_epoch}.pt"
        
        # Setup synthetic data generation if training
        if split == "train":
            self.synthetic_generator = SyntheticDataGenerator(
                tokenizer=tokenizer,
                min_thinking_steps=config.min_thinking_steps,
                max_thinking_steps=config.max_thinking_steps,
                max_length=config.max_length
            )
        
        # Load initial batch into cache
        self._fill_cache()
    
    def _cleanup_synthetic_cache(self):
        """Clean up synthetic data cache files from previous epochs."""
        try:
            for file in os.listdir("cache/synthetic"):
                if file.startswith("epoch_") and file != f"epoch_{self.current_epoch}.pt":
                    os.remove(os.path.join("cache/synthetic", file))
            logger.info("Cleaned up old synthetic cache files")
        except Exception as e:
            logger.warning(f"Error cleaning synthetic cache: {str(e)}")
    
    def _generate_synthetic_batch(self, real_examples: List[Dict], ratio: float) -> List[Dict]:
        """Generate a batch of synthetic examples and cache them to disk."""
        n_synthetic = int(len(real_examples) * ratio / (1 - ratio))
        
        # Generate synthetic examples
        synthetic_examples = integrate_synthetic_data(
            real_examples,
            synthetic_ratio=ratio,
            generator=self.synthetic_generator,
            max_length=self.config.max_length
        )
        
        # Cache to disk
        synthetic_only = [ex for ex in synthetic_examples if ex.get("is_synthetic", False)]
        if synthetic_only:
            torch.save(synthetic_only, self.synthetic_cache_file)
        
        return synthetic_examples
    
    def _load_or_generate_synthetic_data(self, real_examples: List[Dict]) -> List[Dict]:
        """Load synthetic data from cache or generate new batch."""
        if not self.split == "train":
            return real_examples
            
        current_ratio = self._get_current_synthetic_ratio()
        
        # Try to load from cache first
        try:
            if os.path.exists(self.synthetic_cache_file):
                synthetic_examples = torch.load(self.synthetic_cache_file)
                combined = real_examples + synthetic_examples
                logger.info(f"Loaded {len(synthetic_examples)} synthetic examples from cache")
                return combined
        except Exception as e:
            logger.warning(f"Error loading synthetic cache: {str(e)}")
        
        # Generate new if cache missing or invalid
        return self._generate_synthetic_batch(real_examples, current_ratio)
    
    def _process_examples(self, raw_examples: List[Dict]) -> List[Dict]:
        """Process raw examples and combine with synthetic data."""
        processed_examples = []
        
        for example in raw_examples:
            try:
                text = example.get(self.config.text_column, "").strip()
                if not text:
                    continue
                
                # Clean up text
                words = text.split()
                if len(words) < 10:
                    continue
                if len(words) > 1000:
                    text = " ".join(words[:1000])
                
                # Tokenize
                tokenized = self.tokenizer(
                    text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                processed = {
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0),
                    "labels": tokenized["input_ids"].squeeze(0).clone(),
                    "is_synthetic": False,
                    "input": text,
                    "reward": 0.0
                }
                
                processed_examples.append(processed)
                
            except Exception as e:
                logger.warning(f"Error processing example: {str(e)}")
                continue
        
        if not processed_examples:
            raise RuntimeError("No examples could be processed successfully")
        
        # Add synthetic data if training
        if self.split == "train":
            processed_examples = self._load_or_generate_synthetic_data(processed_examples)
            # Clean up old cache files
            self._cleanup_synthetic_cache()
        
        return processed_examples
    
    def _fill_cache(self):
        """Fill the cache with processed examples."""
        logger.info("Filling dataset cache...")
        
        raw_examples = []
        for _ in range(self.cache_size):
            try:
                example = self._get_next_example()
                if self.config.text_column in example:
                    raw_examples.append(example)
                if len(raw_examples) >= self.cache_size:
                    break
            except StopIteration:
                logger.warning("Dataset iterator exhausted, restarting from beginning")
                break
        
        if not raw_examples:
            raise RuntimeError("No valid examples found in dataset")
        
        # Process examples and update cache
        processed_examples = self._process_examples(raw_examples)
        self.cache = {i: example for i, example in enumerate(processed_examples)}
        self.current_size = len(self.cache)
        
        logger.info(f"Cache filled with {self.current_size} examples")
    
    def __len__(self):
        if self.config.max_examples:
            return min(self.config.max_examples, self.current_size)
        return self.current_size
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        return self.cache[idx]
    
    def _get_current_synthetic_ratio(self) -> float:
        """Calculate the current synthetic ratio based on training progress."""
        if self.split != "train" or self.total_epochs <= 1:
            return self.config.base_synthetic_ratio
            
        progress = self.current_epoch / self.total_epochs
        
        # Before ramp start, use base ratio
        if progress < self.config.synthetic_ramp_start:
            return self.config.base_synthetic_ratio
            
        # After ramp start, linearly interpolate to final ratio
        ramp_progress = (progress - self.config.synthetic_ramp_start) / (1 - self.config.synthetic_ramp_start)
        current_ratio = self.config.base_synthetic_ratio + (
            self.config.final_synthetic_ratio - self.config.base_synthetic_ratio
        ) * ramp_progress
        
        logger.info(f"Training progress: {progress:.2f}, Current synthetic ratio: {current_ratio:.2f}")
        return current_ratio
    
    def update_epoch(self, epoch: int):
        """Update the current epoch and manage synthetic data cache."""
        self.current_epoch = epoch
        self.synthetic_cache_file = f"cache/synthetic/epoch_{epoch}.pt"
        # Clean up old cache files when moving to new epoch
        self._cleanup_synthetic_cache()
        # Refresh cache with new synthetic ratio
        self._fill_cache()

    def _get_next_example(self):
        """Get next example from dataset iterator, recreating if needed."""
        try:
            return next(self._dataset_iter)
        except StopIteration:
            # Recreate iterator and try again
            self._dataset_iter = iter(self.dataset)
            return next(self._dataset_iter)

class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        max_memory_usage: float = 0.9,
        min_batch_size: int = 1,
        check_memory_every: int = 10,
        warmup_steps: int = 100,
        warmup_factor: float = 0.5
    ):
        # Validate batch size constraints
        if batch_size > 2:
            logger.warning(f"Batch size {batch_size} exceeds maximum of 2. Setting to 2.")
            batch_size = 2
        if batch_size < min_batch_size:
            raise ValueError(f"Batch size {batch_size} cannot be smaller than min_batch_size {min_batch_size}")
        
        self.dataset = dataset
        self.initial_batch_size = batch_size
        self.current_batch_size = max(min_batch_size, int(batch_size * warmup_factor))
        
        # Validate warmup configuration
        if self.current_batch_size > self.initial_batch_size:
            raise ValueError(
                f"Initial warmup batch size {self.current_batch_size} cannot exceed "
                f"target batch size {self.initial_batch_size}"
            )
        
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.max_memory_usage = max_memory_usage
        self.min_batch_size = min_batch_size
        self.check_memory_every = check_memory_every
        self.batch_counter = 0
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.in_warmup = True
        
        # Track memory stats
        self.memory_history = []
        self.batch_size_history = []
        
        # For length calculation
        self._current_epoch_size = len(dataset)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage including MPS memory."""
        process = psutil.Process()
        mem_info = process.memory_info()
        total_memory = psutil.virtual_memory().total
        system_mem = psutil.virtual_memory()
        
        # Calculate system memory usage
        system_usage = mem_info.rss / total_memory
        
        # Calculate MPS memory usage if available
        mps_usage = 0.0
        if hasattr(torch.mps, 'current_allocated_memory'):
            mps_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
            mps_max = 9.07 * 1024  # Convert 9.07 GB to MB
            mps_usage = mps_allocated / mps_max
        
        # Use the higher of system memory or MPS memory usage
        usage = max(system_usage, mps_usage)
        
        # Store memory info for logging
        memory_info = {
            "rss": mem_info.rss / (1024 * 1024),  # MB
            "vms": mem_info.vms / (1024 * 1024),  # MB
            "shared": getattr(mem_info, "shared", 0) / (1024 * 1024),  # MB
            "system_used": (system_mem.total - system_mem.available) / (1024 * 1024),  # MB
            "system_total": system_mem.total / (1024 * 1024),  # MB
            "system_percent": system_mem.percent,
            "mps_allocated_mb": mps_allocated if hasattr(torch.mps, 'current_allocated_memory') else 0,
            "mps_usage": mps_usage
        }
        
        # Log detailed memory info periodically
        if self.batch_counter % 50 == 0:
            logger.info("\nDetailed Memory Usage:")
            logger.info(f"System Memory: {system_usage:.1%}")
            logger.info(f"MPS Memory: {mps_usage:.1%}")
            logger.info(f"Combined Usage: {usage:.1%}")
            logger.info(f"MPS Allocated: {memory_info['mps_allocated_mb']:.0f}MB")
            logger.info(f"RSS (Process Memory): {memory_info['rss']:.0f}MB")
            logger.info(f"System Memory Used: {memory_info['system_used']:.0f}MB")
        
        self.memory_history.append(memory_info)
        return usage
    
    def _get_warmup_batch_size(self) -> int:
        """Calculate batch size during warmup period."""
        if self.batch_counter >= self.warmup_steps:
            self.in_warmup = False
            return self.initial_batch_size
            
        # Linear warmup
        progress = self.batch_counter / self.warmup_steps
        target_size = self.initial_batch_size
        min_size = max(self.min_batch_size, int(target_size * self.warmup_factor))
        
        warmed_size = min_size + int((target_size - min_size) * progress)
        return warmed_size
    
    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage and warmup state."""
        current_usage = self._get_memory_usage()
        old_batch_size = self.current_batch_size
        
        if self.in_warmup:
            # During warmup, gradually increase batch size
            warmed_size = self._get_warmup_batch_size()
            
            # But still respect memory limits
            if current_usage > self.max_memory_usage:
                warmed_size = max(
                    self.min_batch_size,
                    int(warmed_size * (self.max_memory_usage / current_usage))
                )
            
            self.current_batch_size = warmed_size
            
            if old_batch_size != self.current_batch_size:
                logger.info(
                    f"Warmup step {self.batch_counter}/{self.warmup_steps}: "
                    f"batch size {old_batch_size} -> {self.current_batch_size}, "
                    f"memory usage: {current_usage:.1%}"
                )
        else:
            # After warmup, normal memory-based adjustment
            if current_usage > self.max_memory_usage:
                self.current_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * (self.max_memory_usage / current_usage))
                )
                logger.info(
                    f"Memory usage {current_usage:.1%} > {self.max_memory_usage:.1%}, "
                    f"reducing batch size: {old_batch_size} -> {self.current_batch_size}"
                )
            elif current_usage < self.max_memory_usage * 0.5:  # More conservative threshold (was 0.6)
                self.current_batch_size = min(
                    self.initial_batch_size,
                    int(self.current_batch_size * 1.1)  # Smaller increase (was 1.2)
                )
                if self.current_batch_size != old_batch_size:
                    logger.info(
                        f"Memory usage {current_usage:.1%} < {self.max_memory_usage*0.5:.1%}, "
                        f"increasing batch size: {old_batch_size} -> {self.current_batch_size}"
                    )
        
        self.batch_size_history.append(self.current_batch_size)
    
    def reset_warmup(self):
        """Reset warmup state for new epoch."""
        self.batch_counter = 0
        self.in_warmup = True
        self.current_batch_size = max(self.min_batch_size, int(self.initial_batch_size * self.warmup_factor))
        self.memory_history = []
        self.batch_size_history = []
        # Recalculate epoch size
        self._current_epoch_size = len(self.dataset)
        logger.info(f"Reset batch size to {self.current_batch_size} for warmup")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed statistics about memory usage and batch sizes."""
        if not self.memory_history or not self.batch_size_history:
            return {}
        
        # Calculate memory statistics
        memory_stats = {
            "memory_rss_mean": np.mean([m["rss"] for m in self.memory_history]),
            "memory_rss_max": np.max([m["rss"] for m in self.memory_history]),
            "memory_system_percent_mean": np.mean([m["system_percent"] for m in self.memory_history]),
            "memory_system_percent_max": np.max([m["system_percent"] for m in self.memory_history]),
            "memory_vms_mean": np.mean([m["vms"] for m in self.memory_history]),
            "memory_shared_mean": np.mean([m["shared"] for m in self.memory_history])
        }
        
        # Calculate batch size statistics
        batch_stats = {
            "batch_size_mean": np.mean(self.batch_size_history),
            "batch_size_max": np.max(self.batch_size_history),
            "batch_size_min": np.min(self.batch_size_history),
            "batch_size_std": np.std(self.batch_size_history)
        }
        
        return {**memory_stats, **batch_stats}
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            batch.append(idx)
            
            # Update batch counter and check memory periodically
            self.batch_counter += 1
            # TODO: Consider dynamically reducing the frequency of memory checks when memory usage is low to save processing time.
            if self.batch_counter % self.check_memory_every == 0:
                self._adjust_batch_size()
            
            if len(batch) == self.current_batch_size:
                yield batch
                batch = []
        
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        """Calculate length based on current batch size."""
        if self.drop_last:
            return self._current_epoch_size // self.current_batch_size
        return (self._current_epoch_size + self.current_batch_size - 1) // self.current_batch_size

def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate examples into batches."""
    # Separate synthetic and regular examples
    synthetic = [ex for ex in examples if ex.get("is_synthetic", False)]
    regular = [ex for ex in examples if not ex.get("is_synthetic", False)]
    
    # Process regular examples
    if regular:
        regular_batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in regular]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in regular]),
            "labels": torch.stack([ex["labels"] for ex in regular])
        }
    else:
        regular_batch = None
    
    # Process synthetic examples
    if synthetic:
        synthetic_batch = {
            "input_ids": torch.stack([ex["input_ids"] for ex in synthetic]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in synthetic]),
            "labels": torch.stack([ex["labels"] for ex in synthetic]),
            "rewards": torch.tensor([ex.get("reward", 0.0) for ex in synthetic])
        }
    else:
        synthetic_batch = None
    
    # Combine batches
    if regular_batch and synthetic_batch:
        return {
            "input_ids": torch.cat([regular_batch["input_ids"], synthetic_batch["input_ids"]]),
            "attention_mask": torch.cat([regular_batch["attention_mask"], synthetic_batch["attention_mask"]]),
            "labels": torch.cat([regular_batch["labels"], synthetic_batch["labels"]]),
            "rewards": torch.cat([
                torch.zeros(len(regular_batch["input_ids"])),
                synthetic_batch["rewards"]
            ])
        }
    elif regular_batch:
        return {**regular_batch, "rewards": torch.zeros(len(regular_batch["input_ids"]))}
    else:
        return synthetic_batch

def create_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    total_epochs: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with memory-aware batch sampling."""
    # For training dataset, decide based on training_phase
    if hasattr(config, 'training_phase') and config.training_phase == "cot":
        # Load and merge GSM8K and StrategyQA datasets for chain-of-thought training
        logger.info("Curriculum Phase: Loading CoT datasets (GSM8K and StrategyQA)...")
        gsm8k = load_dataset("gsm8k", split="train")
        strategyqa = load_dataset("strategyqa", split="train")
        merged_dataset = concatenate_datasets([gsm8k, strategyqa])
        train_dataset = EfficientDataset(
            config,
            tokenizer,
            split=config.train_split,
            current_epoch=0,
            total_epochs=total_epochs,
            dataset_override=merged_dataset  # Use the merged CoT dataset
        )
    else:
        train_dataset = EfficientDataset(
            config,
            tokenizer,
            split=config.train_split,
            current_epoch=0,
            total_epochs=total_epochs
        )
    
    val_dataset = EfficientDataset(
        config,
        tokenizer,
        split=config.val_split
    )
    
    # Create memory-aware samplers with warmup
    train_sampler = DynamicBatchSampler(
        train_dataset,
        batch_size=2,  # Hard cap at batch size 2
        shuffle=True,
        max_memory_usage=0.65,
        min_batch_size=1,
        check_memory_every=2,
        warmup_steps=100,  # Reduced warmup steps since we have a smaller range
        warmup_factor=0.5   # Start at batch size 1 (0.5 * 2)
    )
    
    val_sampler = DynamicBatchSampler(
        val_dataset,
        batch_size=2,  # Hard cap at batch size 2
        shuffle=False,
        max_memory_usage=0.65,
        min_batch_size=1,
        check_memory_every=2,
        warmup_steps=100,  # Reduced warmup steps since we have a smaller range
        warmup_factor=0.5   # Start at batch size 1 (0.5 * 2)
    )
    
    # Create dataloaders with custom collate function
    # Note: Using num_workers=0 for streaming datasets as they can't be pickled
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,  # Must be 0 for streaming datasets
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=0,  # Must be 0 for streaming datasets
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader 