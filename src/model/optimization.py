import torch
import gc
import psutil
import numpy as np
from transformers import PreTrainedModel
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    def __init__(
        self,
        defrag_threshold: float = 0.2,  # Trigger defrag when fragmentation exceeds 20%
        mps_memory_fraction: float = 0.8,  # Target MPS memory utilization
        check_interval: int = 100  # Check every 100 steps
    ):
        self.defrag_threshold = defrag_threshold
        self.mps_memory_fraction = mps_memory_fraction
        self.check_interval = check_interval
        self.step_counter = 0
        self.last_defrag_step = 0
        self.memory_stats = []
        
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        stats = {}
        
        # Get process memory info
        process = psutil.Process()
        mem_info = process.memory_info()
        
        # System memory
        stats['system_used'] = psutil.virtual_memory().percent / 100.0
        stats['process_rss'] = mem_info.rss / (1024 * 1024)  # MB
        
        # MPS memory if available
        if hasattr(torch.mps, 'current_allocated_memory'):
            stats['mps_allocated'] = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
            if hasattr(torch.mps, 'driver_allocated_memory'):
                stats['mps_driver'] = torch.mps.driver_allocated_memory() / (1024 * 1024)  # MB
        
        # Estimate fragmentation
        if len(self.memory_stats) > 0:
            peak_mem = max(stat['process_rss'] for stat in self.memory_stats)
            stats['fragmentation'] = (stats['process_rss'] - peak_mem) / peak_mem
        else:
            stats['fragmentation'] = 0.0
            
        return stats
    
    def _optimize_mps_memory(self):
        """Optimize MPS memory usage."""
        if not torch.backends.mps.is_available():
            return
            
        try:
            # Clear MPS cache
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
            # Synchronize MPS device
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            
            # Compact allocator if available
            if hasattr(torch.mps, 'compact_allocator'):
                torch.mps.compact_allocator()
                
        except Exception as e:
            logger.warning(f"Error optimizing MPS memory: {str(e)}")
    
    def _defragment_memory(self):
        """Perform memory defragmentation."""
        # Collect Python garbage
        gc.collect()
        
        # Clear CUDA/MPS cache
        if torch.backends.mps.is_available():
            self._optimize_mps_memory()
        
        # Move tensors to contiguous memory
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if not obj.is_contiguous():
                    try:
                        obj.storage().resize_(obj.storage().size())
                    except Exception:
                        continue
        
        # Force garbage collection again
        gc.collect()
        
        self.last_defrag_step = self.step_counter
    
    def step(self):
        """Perform one optimization step."""
        self.step_counter += 1
        
        # Check memory status periodically
        if self.step_counter % self.check_interval == 0:
            stats = self._get_memory_stats()
            self.memory_stats.append(stats)
            
            # Keep only recent history
            if len(self.memory_stats) > 1000:
                self.memory_stats = self.memory_stats[-1000:]
            
            # Check if defragmentation is needed
            if stats['fragmentation'] > self.defrag_threshold:
                logger.info(f"Memory fragmentation detected: {stats['fragmentation']:.1%}")
                self._defragment_memory()
            
            # Optimize MPS memory if utilization is high
            if 'mps_allocated' in stats:
                mps_utilization = stats['mps_allocated'] / (9.07 * 1024)  # 9.07 GB total
                if mps_utilization > self.mps_memory_fraction:
                    self._optimize_mps_memory()
            
            # Log memory stats
            logger.info("\nMemory Statistics:")
            logger.info(f"System Memory Used: {stats['system_used']:.1%}")
            logger.info(f"Process RSS: {stats['process_rss']:.0f}MB")
            if 'mps_allocated' in stats:
                logger.info(f"MPS Allocated: {stats['mps_allocated']:.0f}MB")
            logger.info(f"Fragmentation: {stats['fragmentation']:.1%}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return self._get_memory_stats()

def setup_device(use_mps: bool = True, fallback_to_cpu: bool = True) -> torch.device:
    """Set up the appropriate device for training."""
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif fallback_to_cpu:
        return torch.device("cpu")
    else:
        raise RuntimeError("MPS is not available and fallback to CPU is disabled")

def optimize_memory(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """Apply memory optimization techniques to the model."""
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    return model

def setup_mixed_precision() -> torch.amp.autocast:
    """Set up mixed precision training context."""
    # For MPS, we use float16 precision where supported
    return torch.amp.autocast(device_type="mps", dtype=torch.float16)

def count_trainable_parameters(model: PreTrainedModel) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_memory_optimizations(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_gradient_checkpointing: bool = True
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, MemoryOptimizer]:
    """Setup memory optimizations for training."""
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Move model to device efficiently
    model = model.to(device, non_blocking=True)
    
    # Optimize optimizer state placement
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)
    
    # Create memory optimizer
    memory_optimizer = MemoryOptimizer(
        defrag_threshold=0.2,
        mps_memory_fraction=0.8,
        check_interval=100
    )
    
    return model, optimizer, memory_optimizer

def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics."""
    optimizer = MemoryOptimizer()
    return optimizer.get_memory_stats()
