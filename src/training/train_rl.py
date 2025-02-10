import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import psutil
import time
import wandb
from tqdm import tqdm
import numpy as np

from src.data.data_loader import (
    DataConfig,
    EfficientDataset,
    collate_fn,
    create_dataloaders,
    DynamicBatchSampler
)
from src.rl.rl_components import (
    PolicyNetwork,
    ValueNetwork,
    calculate_reward,
    PPOMemory,
    compute_gae,
    ReplayBuffer
)
from src.memory.memory_components import (
    EphemeralMemory,
    EpisodicMemory,
    MonolithIntake,
    EpisodicTrace
)
from src.memory.types import MemoryConfig
from src.model.optimization import setup_memory_optimizations

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class RLTrainer:
    def __init__(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "PreTrainedTokenizer",
        train_dataset: "EfficientDataset",
        val_dataset: "EfficientDataset",
        device: torch.device,
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config
        
        # Initialize wandb
        if config.get("use_wandb", True):
            wandb.init(
                project="llm-training",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["rl-training", config.get("experiment_tag", "default")]
            )
        
        # Create memory config
        memory_config = MemoryConfig(
            n_clusters=config.get("n_clusters", 5),
            batch_size=config.get("cluster_batch_size", 10),
            salience_threshold=config.get("salience_threshold", 0.6),
            similarity_threshold=config.get("similarity_threshold", 0.5)
        )
        
        # Initialize memory components
        self.ephemeral_memory = EphemeralMemory()
        self.episodic_memory = EpisodicMemory(
            salience_threshold=memory_config.salience_threshold,
            device=device
        )
        self.monolith_intake = MonolithIntake(
            batch_size=config.get("monolith_batch_size", 32),
            device=device
        )
        
        # Initialize RL components
        self.policy_net = PolicyNetwork(
            hidden_size=model.config.hidden_size,
            action_size=model.config.vocab_size,
            dropout=config.get("policy_dropout", 0.1)
        ).to(device)
        
        self.value_net = ValueNetwork(
            hidden_size=model.config.hidden_size,
            dropout=config.get("value_dropout", 0.1)
        ).to(device)
        
        # Initialize optimizers
        self.lm_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        self.rl_optimizer = torch.optim.AdamW(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=config["rl_learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Initialize memory optimizer
        self.model, self.lm_optimizer, self.memory_optimizer = setup_memory_optimizations(
            model=self.model,
            optimizer=self.lm_optimizer,
            device=self.device,
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", True)
        )
        
        # Initialize PPO memory and replay buffer
        self.ppo_memory = PPOMemory()
        self.replay_buffer = ReplayBuffer(
            max_size=config.get("replay_buffer_size", 10000),
            device=device
        )
        
        # Training parameters
        self.n_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.rl_weight = config.get("rl_weight", 0.1)
        
        # Note: Dataloaders will be set externally
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Early stopping parameters
        self.patience = config.get("patience", 3)
        self.min_delta = config.get("min_delta", 1e-4)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize learning rate scheduler
        # We'll set the number of training steps after dataloader is set
        self.lm_scheduler = None
        self.rl_scheduler = None
        
        # Add batch counter and memory tracking
        self.batch_counter = 0
        self.memory_stats = {
            "model_memory": [],
            "optimizer_memory": [],
            "batch_memory": [],
            "total_memory": [],
            "mps_memory": []
        }
    
    def setup_schedulers(self):
        """Setup learning rate schedulers after dataloaders are set."""
        if self.train_dataloader is None:
            raise ValueError("Dataloader must be set before setting up schedulers")
            
        num_training_steps = len(self.train_dataloader) * self.n_epochs
        num_warmup_steps = self.config.get("lr_warmup_steps", int(0.1 * num_training_steps))
        
        self.lm_scheduler = get_linear_schedule_with_warmup(
            self.lm_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.rl_scheduler = get_linear_schedule_with_warmup(
            self.rl_optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def compute_salience(self, text: str, embedding: torch.Tensor) -> float:
        """Compute salience score for a text and its embedding."""
        # Simplified salience calculation
        # In practice, this could be more sophisticated
        embedding_norm = embedding.norm().item()
        text_length = len(text.split())
        return min(0.3 * embedding_norm + 0.7 * (text_length / 100), 1.0)
    
    def _process_batch_monolith(self, batch: Dict[str, torch.Tensor]) -> List[EpisodicTrace]:
        """Process a batch through the monolith intake system."""
        # Extract hidden states from model output
        hidden_states = self._extract_hidden_states(
            self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )
        )
        
        # Add each example to the monolith batch
        for i in range(len(batch["input_ids"])):
            text = self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
            self.monolith_intake.add_to_batch(text, hidden_states[i])
        
        # Process the batch and return traces
        traces = self.monolith_intake.process_batch()
        return traces
    
    def _extract_hidden_states(self, model_output) -> torch.Tensor:
        """Extract and process hidden states from model output."""
        hidden_states = model_output.hidden_states[-1]
        return hidden_states.mean(dim=1)
    
    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        train: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss (LM + RL) and metrics."""
        # Track memory before computation
        if train and self.batch_counter % 50 == 0:  # Log every 50 batches
            # Get process memory info
            process = psutil.Process()
            mem_info = process.memory_info()
            system_mem = psutil.virtual_memory()
            
            # Calculate model memory
            model_memory = sum(
                param.element_size() * param.nelement()
                for param in self.model.parameters()
            ) / (1024 * 1024)  # MB
            
            # Calculate optimizer memory
            optimizer_memory = sum(
                param.element_size() * param.nelement()
                for group in self.lm_optimizer.param_groups
                for param in group["params"]
            ) / (1024 * 1024)  # MB
            
            # Calculate batch memory
            batch_memory = sum(
                tensor.element_size() * tensor.nelement()
                for tensor in batch.values()
            ) / (1024 * 1024)  # MB
            
            # Get MPS memory if available
            mps_memory = 0
            if hasattr(torch.mps, 'current_allocated_memory'):
                mps_memory = torch.mps.current_allocated_memory() / (1024 * 1024)  # MB
            
            # Update memory stats
            self.memory_stats["model_memory"].append(model_memory)
            self.memory_stats["optimizer_memory"].append(optimizer_memory)
            self.memory_stats["batch_memory"].append(batch_memory)
            self.memory_stats["total_memory"].append(mem_info.rss / (1024 * 1024))
            self.memory_stats["mps_memory"].append(mps_memory)
            
            # Log memory breakdown
            logger.info("\nDetailed Memory Usage:")
            logger.info(f"Model Parameters: {model_memory:.0f}MB")
            logger.info(f"Optimizer States: {optimizer_memory:.0f}MB")
            logger.info(f"Current Batch: {batch_memory:.0f}MB")
            logger.info(f"MPS Allocated: {mps_memory:.0f}MB")
            logger.info(f"Total RSS: {mem_info.rss / (1024 * 1024):.0f}MB")
            logger.info(f"System Memory Usage: {system_mem.percent:.1f}%")
            
            # Log memory stats to wandb if enabled
            if self.config.get("use_wandb", True):
                wandb.log({
                    "memory/model": model_memory,
                    "memory/optimizer": optimizer_memory,
                    "memory/batch": batch_memory,
                    "memory/mps": mps_memory,
                    "memory/total": mem_info.rss / (1024 * 1024),
                    "memory/system_percent": system_mem.percent
                }, step=self.batch_counter)
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Process batch through monolith intake if training
        if train:
            traces = self._process_batch_monolith(batch)
            batch_stats = self.monolith_intake.compute_batch_statistics(traces)
        
        # Forward pass through language model
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True
        )
        
        lm_loss = outputs.loss
        
        if train:
            # Extract state representation from hidden states
            state = self._extract_hidden_states(outputs)
            
            # Calculate reward (using negative loss as immediate reward)
            reward = -lm_loss.detach()
            
            # Reshape and scale reward
            reward = reward.view(-1, 1)  # Shape: [batch_size, 1]
            reward = torch.tanh(reward)  # Scale to [-1, 1]
            
            # Get policy and value predictions
            policy_logits = self.policy_net(state)
            value_pred = self.value_net(state)
            
            # Sample actions from policy
            action_probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            # Store in PPO memory for later updates
            self.ppo_memory.store_memory(
                state.detach(),
                actions.detach(),
                log_probs.detach(),
                value_pred.detach(),
                reward.detach(),
                torch.zeros_like(reward),  # done flag
                outputs.hidden_states[-1].detach()  # store full hidden state
            )
            
            # Ensure all tensors have the same batch dimension
            batch_size = state.size(0)
            reward = reward.expand(batch_size, 1)
            value_pred = value_pred.view(batch_size, 1)
            
            # Compute advantage
            advantage = (reward - value_pred).detach()
            
            # Compute RL losses
            policy_loss = -(log_probs * advantage.squeeze(-1)).mean()
            value_loss = F.mse_loss(value_pred, reward)
            
            # Scale losses
            policy_loss = policy_loss * self.config.get("policy_loss_coef", 1.0)
            value_loss = value_loss * self.config.get("value_loss_coef", 0.5)
            
            # Combine losses
            total_loss = lm_loss + self.rl_weight * (policy_loss + value_loss)
            
            metrics = {
                "lm_loss": lm_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "total_loss": total_loss.item(),
                "reward": reward.mean().item(),
                "advantage": advantage.mean().item(),
                "policy_entropy": dist.entropy().mean().item(),
                **batch_stats  # Include monolith batch statistics
            }
        else:
            total_loss = lm_loss
            metrics = {"lm_loss": lm_loss.item()}
        
        return total_loss, metrics
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with dynamic learning rate adjustment."""
        self.model.train()
        self.policy_net.train()
        self.value_net.train()
        
        # Reset batch counter at start of epoch
        self.batch_counter = 0
        
        # Initialize reward tracking for dynamic LR adjustment
        reward_ema = 0.0
        ema_alpha = 0.1  # EMA decay rate
        base_lr = self.lm_optimizer.param_groups[0]['lr']
        min_lr = base_lr * 0.1
        max_lr = base_lr * 2.0
        
        # Update dataset epoch counter and manage synthetic data
        if isinstance(self.train_dataset, EfficientDataset):
            self.train_dataset.update_epoch(epoch)
        
        # Reset batch size warmup for the new epoch
        if hasattr(self.train_dataloader.batch_sampler, 'reset_warmup'):
            self.train_dataloader.batch_sampler.reset_warmup()
        
        total_metrics = {}
        num_batches = len(self.train_dataloader)
        log_interval = max(1, num_batches // 10)
        
        logger.info(f"\nEpoch {epoch+1}/{self.n_epochs}")
        logger.info("-" * 50)
        
        # Track memory usage and batch sizes
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        batch_sizes = []
        memory_usage = []
        
        # Initialize CPU accumulators for gradients and a counter for batches
        grad_accumulators = {name: None for name, _ in self.model.named_parameters()}
        batch_count = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Increment batch counter
            self.batch_counter += 1
            
            # Track batch size and memory
            current_batch_size = len(batch["input_ids"])
            batch_sizes.append(current_batch_size)
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)
            
            # Compute loss and metrics
            loss, metrics = self._compute_loss(batch, train=True)
            
            # Update reward EMA for dynamic LR adjustment
            current_reward = metrics.get('reward', 0.0)
            reward_ema = ema_alpha * current_reward + (1 - ema_alpha) * reward_ema
            
            # Adjust learning rate based on reward trend
            reward_ratio = max(0.1, min(2.0, (current_reward / (reward_ema + 1e-8))))
            adjusted_lr = min(max(base_lr * reward_ratio, min_lr), max_lr)
            
            # Update learning rates
            for param_group in self.lm_optimizer.param_groups:
                param_group['lr'] = adjusted_lr
            for param_group in self.rl_optimizer.param_groups:
                param_group['lr'] = adjusted_lr * (self.config['rl_learning_rate'] / self.config['learning_rate'])
            
            # Accumulate gradients on CPU
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_cpu = param.grad.detach().cpu()
                    if grad_accumulators[name] is None:
                        grad_accumulators[name] = grad_cpu.clone()
                    else:
                        grad_accumulators[name] += grad_cpu
            
            # Clear gradients to free memory on MPS device
            self.model.zero_grad()
            
            # Step memory optimizer
            self.memory_optimizer.step()
            
            # Get memory stats for logging
            if self.batch_counter % 50 == 0:  # Log every 50 batches
                memory_stats = self.memory_optimizer.get_memory_stats()
                
                # Log memory stats to wandb
                if self.config.get("use_wandb", True):
                    wandb.log({
                        "memory/system_used": memory_stats['system_used'],
                        "memory/process_rss": memory_stats['process_rss'],
                        "memory/fragmentation": memory_stats['fragmentation'],
                        **({"memory/mps_allocated": memory_stats['mps_allocated']} 
                           if 'mps_allocated' in memory_stats else {})
                    }, step=self.batch_counter)
            
            # Update metrics
            metrics['learning_rate'] = adjusted_lr
            metrics['reward_ema'] = reward_ema
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            
            # Update progress bar
            avg_metrics = {k: v / (batch_idx + 1) for k, v in total_metrics.items()}
            progress_bar.set_postfix(
                loss=avg_metrics["lm_loss"],
                reward=avg_metrics["reward"],
                lr=adjusted_lr,
                batch_size=current_batch_size,
                memory_mb=f"{current_memory:.0f}"
            )
            
            # Log to wandb
            if self.config.get("use_wandb", True) and (batch_idx + 1) % log_interval == 0:
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "learning_rate": adjusted_lr,
                    "reward_ema": reward_ema,
                    **metrics,
                    "batch_size": current_batch_size,
                    "memory_usage_mb": current_memory,
                    "effective_batch_size": current_batch_size * self.gradient_accumulation_steps
                })
        
        # End of epoch processing: Average the accumulated gradients and apply updates
        for name, param in self.model.named_parameters():
            if grad_accumulators[name] is not None:
                avg_grad = grad_accumulators[name] / batch_count
                param.grad = avg_grad.to(param.device)
        
        # Update model parameters using the aggregated gradients
        self.lm_optimizer.step()
        self.rl_optimizer.step()
        self.lm_scheduler.step()
        self.rl_scheduler.step()
        self.lm_optimizer.zero_grad()
        self.rl_optimizer.zero_grad()
        
        # Log final memory usage and batch size statistics
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_change = end_memory - start_memory
        
        logger.info("\nBatch size statistics:")
        logger.info(f"  Mean: {np.mean(batch_sizes):.1f}")
        logger.info(f"  Min: {np.min(batch_sizes)}")
        logger.info(f"  Max: {np.max(batch_sizes)}")
        logger.info(f"  Std: {np.std(batch_sizes):.1f}")
        logger.info(f"\nMemory change during epoch: {memory_change:.1f}MB")
        logger.info(f"Final learning rate: {adjusted_lr:.2e}")
        logger.info(f"Final reward EMA: {reward_ema:.3f}")
        
        return avg_metrics
    
    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        self.policy_net.eval()
        self.value_net.eval()
        
        total_metrics = {}
        num_batches = len(self.val_dataloader)
        start_time = time.time()
        
        logger.info("\nValidation:")
        logger.info("-" * 50)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                _, metrics = self._compute_loss(batch, train=False)
                
                # Update metrics
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                
                # Log progress periodically
                if (batch_idx + 1) % (num_batches // 5) == 0:  # Log 5 times during validation
                    progress = (batch_idx + 1) / num_batches * 100
                    avg_metrics = {k: v / (batch_idx + 1) for k, v in total_metrics.items()}
                    elapsed_time = time.time() - start_time
                    
                    logger.info(
                        f"Progress: {progress:3.0f}% [{batch_idx+1}/{num_batches}] | "
                        f"Time: {elapsed_time:.1f}s | "
                        f"Loss: {avg_metrics['lm_loss']:.4f}"
                    )
        
        # Compute average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Log validation metrics to wandb
        if self.config.get("use_wandb", True):
            wandb.log({
                "val_" + k: v for k, v in avg_metrics.items()
            })
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False, checkpoint_name: str = None):
        """Save a checkpoint of the model and training state."""
        logger.info("Preparing to save checkpoint...")
        
        # Ensure checkpoint directory exists with proper permissions
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.chmod(self.checkpoint_dir, 0o755)  # rwxr-xr-x
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'lm_optimizer_state_dict': self.lm_optimizer.state_dict(),
            'rl_optimizer_state_dict': self.rl_optimizer.state_dict(),
            'lm_scheduler_state_dict': self.lm_scheduler.state_dict() if self.lm_scheduler else None,
            'rl_scheduler_state_dict': self.rl_scheduler.state_dict() if self.rl_scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'memory_state': {
                'ephemeral_memory': self.ephemeral_memory.get_state(),
                'episodic_memory': self.episodic_memory.get_state(),
                'monolith_intake': self.monolith_intake.get_state()
            },
            'rl_state': {
                'replay_buffer': self.replay_buffer.get_state(),
                'ppo_memory': self.ppo_memory.get_state()
            },
            'batch_counter': self.batch_counter,
            'memory_stats': self.memory_stats,
            'best_val_loss': self.best_val_loss,
            'early_stopping_counter': self.early_stopping_counter
        }
        
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        logger.info(f"Saving checkpoint to: {checkpoint_path}")
        
        try:
            # Save to a temporary file first
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            # Then rename it to the final path (atomic operation)
            os.replace(temp_path, checkpoint_path)
            logger.info(f"Successfully saved checkpoint to {checkpoint_path}")
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                logger.info(f"Successfully saved best model to {best_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint and restore the training state."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.lm_optimizer.load_state_dict(checkpoint['lm_optimizer_state_dict'])
        self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer_state_dict'])
        
        # Restore scheduler states if they exist
        if checkpoint.get('lm_scheduler_state_dict') and self.lm_scheduler:
            self.lm_scheduler.load_state_dict(checkpoint['lm_scheduler_state_dict'])
        if checkpoint.get('rl_scheduler_state_dict') and self.rl_scheduler:
            self.rl_scheduler.load_state_dict(checkpoint['rl_scheduler_state_dict'])
        
        # Restore memory system state
        memory_state = checkpoint['memory_state']
        self.ephemeral_memory.set_state(memory_state['ephemeral_memory'])
        self.episodic_memory.set_state(memory_state['episodic_memory'])
        self.monolith_intake.set_state(memory_state['monolith_intake'])
        
        # Restore RL components state
        rl_state = checkpoint['rl_state']
        self.replay_buffer.set_state(rl_state['replay_buffer'])
        self.ppo_memory.set_state(rl_state['ppo_memory'])
        
        # Restore training state
        self.batch_counter = checkpoint.get('batch_counter', 0)
        self.memory_stats = checkpoint.get('memory_stats', {
            "model_memory": [],
            "optimizer_memory": [],
            "batch_memory": [],
            "total_memory": [],
            "mps_memory": []
        })
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
        
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def train(self):
        """Main training loop with early stopping and checkpointing."""
        logger.info("\nStarting training...")
        logger.info("=" * 50)
        logger.info(f"Number of training examples: {len(self.train_dataset)}")
        logger.info(f"Number of validation examples: {len(self.val_dataset)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"Early stopping patience: {self.patience}")
        logger.info("=" * 50 + "\n")
        
        start_epoch = 0
        
        # Load checkpoint if specified and exists
        if self.config.get("resume_from_checkpoint"):
            checkpoint_path = self.config["resume_from_checkpoint"]
            if os.path.exists(checkpoint_path):
                start_epoch, self.best_val_loss = self.load_checkpoint(checkpoint_path)
                logger.info(f"Resumed training from epoch {start_epoch+1}")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}, starting fresh training")
        
        for epoch in range(start_epoch, self.n_epochs):
            # Curriculum phase switching: if the current epoch equals curriculum_switch_epoch, reinitialize dataloaders
            if hasattr(self.config, 'curriculum_switch_epoch') and epoch == self.config.curriculum_switch_epoch:
                logger.info("Curriculum phase switching: transitioning from C4 to CoT dataset.")
                self.config.training_phase = "cot"
                # Reinitialize dataloaders with updated config
                self.train_loader, self.val_loader = create_dataloaders(self.config, self.tokenizer, total_epochs=self.total_epochs)
                logger.info("Dataloaders reinitialized for CoT training phase.")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Log training metrics
            logger.info("\nTraining metrics:")
            for k, v in train_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            # Validation phase
            val_metrics = self.evaluate()
            
            # Log validation metrics
            logger.info("\nValidation metrics:")
            for k, v in val_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
            
            val_loss = val_metrics["lm_loss"]
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss - self.min_delta
            if is_best:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                logger.info(f"\nNew best validation loss: {self.best_val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                logger.info(
                    f"\nValidation loss did not improve. "
                    f"Counter: {self.early_stopping_counter}/{self.patience}"
                )
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.patience:
                logger.info(
                    f"\nEarly stopping triggered after {epoch+1} epochs. "
                    f"Best validation loss: {self.best_val_loss:.4f}"
                )
                break
            
            logger.info("\n" + "=" * 50)
        
        logger.info("\nTraining completed successfully!")
        
        # Load best model for final save
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
            logger.info("Loaded best model for final save")
        
        # Save final model
        self.save_model("final_model")
        
        # Close wandb run
        if self.config.get("use_wandb", True):
            wandb.finish()
    
    def save_model(self, name: str):
        """Save model, tokenizer, and RL components."""
        output_dir = os.path.join(self.config["output_dir"], name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save language model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save RL components
        torch.save(self.policy_net.state_dict(), os.path.join(output_dir, "policy_net.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(output_dir, "value_net.pt"))
        
        logger.info(f"Model saved to {output_dir}")

    def save_full_checkpoint(self, filepath: str):
        """Save a complete checkpoint including model, optimizer, memory, and SSMM states."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory_optimizer_state': self.memory_optimizer.get_state() if self.memory_optimizer else None,
            'monolith_intake_state': self.monolith_intake.get_state() if self.monolith_intake else None,
            'episodic_memory_state': self.episodic_memory.get_state() if self.episodic_memory else None,
            'rl_components': {
                'policy_net': self.policy_net.state_dict() if self.policy_net else None,
                'value_net': self.value_net.state_dict() if self.value_net else None,
                'ppo_memory': self.ppo_memory if self.ppo_memory else None,
                'replay_buffer': self.replay_buffer if self.replay_buffer else None
            },
            # Save SSMM states from attention layers
            'attention_states': {
                name: module.get_state()
                for name, module in self.model.named_modules()
                if isinstance(module, (HybridAttention, GlobalMemoryAttention))
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved full checkpoint to {filepath}")

    def load_full_checkpoint(self, filepath: str):
        """Load a complete checkpoint including model, optimizer, memory, and SSMM states."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        # Load memory components
        if 'memory_optimizer_state' in checkpoint and checkpoint['memory_optimizer_state']:
            self.memory_optimizer.set_state(checkpoint['memory_optimizer_state'])
        
        if 'monolith_intake_state' in checkpoint and checkpoint['monolith_intake_state']:
            self.monolith_intake.set_state(checkpoint['monolith_intake_state'])
        
        if 'episodic_memory_state' in checkpoint and checkpoint['episodic_memory_state']:
            self.episodic_memory.set_state(checkpoint['episodic_memory_state'])
        
        # Load RL components
        if 'rl_components' in checkpoint:
            if self.policy_net and checkpoint['rl_components']['policy_net']:
                self.policy_net.load_state_dict(checkpoint['rl_components']['policy_net'])
            if self.value_net and checkpoint['rl_components']['value_net']:
                self.value_net.load_state_dict(checkpoint['rl_components']['value_net'])
            if checkpoint['rl_components']['ppo_memory']:
                self.ppo_memory = checkpoint['rl_components']['ppo_memory']
            if checkpoint['rl_components']['replay_buffer']:
                self.replay_buffer = checkpoint['rl_components']['replay_buffer']
        
        # Load SSMM states from attention layers
        if 'attention_states' in checkpoint:
            for name, module in self.model.named_modules():
                if isinstance(module, (HybridAttention, GlobalMemoryAttention)):
                    if name in checkpoint['attention_states']:
                        module.set_state(checkpoint['attention_states'][name])
        
        logger.info(f"Loaded full checkpoint from {filepath}")

def train():
    """Main training function."""
    # Updated configuration
    config = {
        # Model config
        "output_dir": "model_outputs",
        "num_epochs": 15,
        "batch_size": 2,  # Hard cap at batch size 2
        "gradient_accumulation_steps": 8,  # Increased to maintain effective batch size
        "learning_rate": 2e-5,
        "rl_learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "cache_size": 1000,
        "lr_warmup_steps": 1000,
        "max_grad_norm": 1.0,
        "rl_weight": 0.1,
        "policy_dropout": 0.1,
        "value_dropout": 0.1,
        "policy_loss_coef": 1.0,
        "value_loss_coef": 0.5,
        
        # Memory config
        "salience_threshold": 0.6,
        "n_clusters": 5,
        "cluster_batch_size": 10,
        "similarity_threshold": 0.5,
        "monolith_batch_size": 32,
        
        # Generation config
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "use_cache": True,
        
        # Early stopping parameters
        "patience": 5,
        "min_delta": 1e-4,
        
        # Checkpointing parameters
        "resume_from_checkpoint": None,
        "checkpoint_interval": 100,
        "replay_buffer_size": 10000,
        
        # Experiment tracking
        "use_wandb": True,
        "experiment_tag": "progressive_synthetic",
        
        # Dataset config
        "dataset": "c4",
        "dataset_subset": "en",
        "max_examples": 1000000,
        "base_synthetic_ratio": 0.2,
        "final_synthetic_ratio": 0.9,
        "synthetic_ramp_start": 0.85
    }
    
    # Create output directories
    os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "logs"), exist_ok=True)
    
    # Set environment variable for tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer with logging level
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize model with updated config
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        output_hidden_states=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        use_cache=True
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=model_config)
    model.to(device)
    
    # Create datasets with reduced logging
    logging.getLogger("datasets").setLevel(logging.WARNING)
    data_config = DataConfig(
        dataset_name=config["dataset"],
        dataset_config_name=config["dataset_subset"],
        text_column="text",
        max_length=config["max_length"],
        batch_size=config["batch_size"],
        cache_size=config["cache_size"],
        base_synthetic_ratio=config["base_synthetic_ratio"],
        final_synthetic_ratio=config["final_synthetic_ratio"],
        synthetic_ramp_start=config["synthetic_ramp_start"],
        min_thinking_steps=3,
        max_thinking_steps=12,
        warmup_steps=100,  # Default batch size warmup steps
        warmup_factor=0.3  # Default warmup factor
    )
    
    logger.info("Creating datasets...")
    # Create dataloaders with total epochs information
    train_loader, val_loader = create_dataloaders(
        data_config,
        tokenizer,
        total_epochs=config["num_epochs"]
    )
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    logger.info(f"Training with {len(train_dataset)} examples")
    logger.info(f"Validating with {len(val_dataset)} examples")
    
    # Initialize trainer with datasets
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=config
    )
    
    # Set dataloaders and initialize schedulers
    trainer.train_dataloader = train_loader
    trainer.val_dataloader = val_loader
    trainer.setup_schedulers()
    
    logger.info("Starting training with the following configuration:")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  Gradient accumulation steps: {config['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  RL learning rate: {config['rl_learning_rate']}")
    logger.info(f"  Number of epochs: {config['num_epochs']}")
    logger.info(f"  LR warmup steps: {config['lr_warmup_steps']}")
    logger.info(f"  Batch warmup steps: {data_config.warmup_steps}")
    logger.info(f"  Initial batch size factor: {data_config.warmup_factor}")
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    # Set warning filters
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")
    
    train() 