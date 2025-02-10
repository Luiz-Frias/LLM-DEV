import torch
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer
import random
from dataclasses import dataclass
import numpy as np

@dataclass
class SyntheticExample:
    prompt: str
    response: str
    thinking_steps: int
    reward: float

class RewardShaper:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        base_reward: float = 1.0,
        thinking_bonus: float = 0.5,
        max_thinking_steps: int = 8,
        correctness_bonus: float = 2.0  # Large bonus for correct answers
    ):
        self.tokenizer = tokenizer
        self.base_reward = base_reward
        self.thinking_bonus = thinking_bonus
        self.max_thinking_steps = max_thinking_steps
        self.correctness_bonus = correctness_bonus
    
    def count_thinking_steps(self, text: str) -> int:
        """Count the number of explicit thinking steps in the response."""
        # Look for numbered lists, step-by-step indicators, etc.
        indicators = [
            "1.", "First,", "Step 1:",
            "2.", "Second,", "Step 2:",
            "3.", "Third,", "Step 3:",
            # Add more indicators as needed
        ]
        
        count = 0
        for indicator in indicators:
            if indicator in text:
                count += 1
        
        return min(count, self.max_thinking_steps)
    
    def _check_answer_correctness(self, response: str, expected_answer: str) -> float:
        """
        Check how correct the answer is compared to expected answer.
        Returns a score between 0 and 1.
        """
        # Simple exact match (1.0 for exact match)
        if response.strip() == expected_answer.strip():
            return 1.0
            
        # Partial matching (can be enhanced with more sophisticated metrics)
        response_words = set(response.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(response_words.intersection(expected_words))
        union = len(response_words.union(expected_words))
        
        if union == 0:
            return 0.0
            
        similarity = intersection / union
        return similarity
    
    def calculate_reward(
        self,
        response: str,
        prompt: str,
        base_quality: float,
        expected_answer: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward with bonus for structured thinking and correctness."""
        # Count thinking steps
        thinking_steps = self.count_thinking_steps(response)
        
        # Calculate thinking bonus
        thinking_multiplier = min(thinking_steps / self.max_thinking_steps, 1.0)
        thinking_reward = self.thinking_bonus * thinking_multiplier
        
        # Calculate correctness bonus if expected answer is provided
        correctness_reward = 0.0
        if expected_answer is not None:
            correctness_score = self._check_answer_correctness(response, expected_answer)
            correctness_reward = self.correctness_bonus * correctness_score
        
        # Combine all rewards
        total_reward = (
            self.base_reward * base_quality +  # Base quality reward
            thinking_reward +                  # Reward for structured thinking
            correctness_reward                 # Big bonus for correctness
        )
        
        # Return reward and components for logging
        components = {
            "base_reward": self.base_reward * base_quality,
            "thinking_bonus": thinking_reward,
            "thinking_steps": thinking_steps,
            "correctness_reward": correctness_reward,
            "total_reward": total_reward
        }
        
        return total_reward, components

class SyntheticDataGenerator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        base_prompts: List[str] = None,
        min_thinking_steps: int = 3,
        max_thinking_steps: int = 12,
        reward_scale: float = 1.0,
        reward_shaper: Optional[RewardShaper] = None,
        max_length: int = 512  # Add max_length parameter
    ):
        self.tokenizer = tokenizer
        self.base_prompts = base_prompts or self._get_default_prompts()
        self.min_thinking_steps = min_thinking_steps
        self.max_thinking_steps = max_thinking_steps
        self.reward_scale = reward_scale
        self.max_length = max_length
        self.reward_shaper = reward_shaper or RewardShaper(
            tokenizer=tokenizer,
            correctness_bonus=2.0
        )
    
    def _get_default_prompts(self) -> List[str]:
        """Default prompts encouraging structured thinking."""
        return [
            "Let's approach this step by step:\n1.",
            "Let's think about this carefully:\n1.",
            "To solve this, we should:\n1.",
            "Let's break this down:\n1.",
            "Here's my thought process:\n1."
        ]
    
    def _generate_thinking_steps(self, n_steps: int) -> str:
        """Generate numbered thinking steps."""
        steps = []
        for i in range(n_steps):
            steps.append(f"{i+1}. [THINKING_STEP_{i+1}]")
        return "\n".join(steps)
    
    def _calculate_reward(
        self,
        thinking_steps: int,
        response_quality: float,
        response: str = None,
        expected_answer: str = None
    ) -> float:
        """Calculate reward based on thinking steps, response quality, and correctness."""
        # Use reward shaper if response and expected answer are provided
        if response is not None and expected_answer is not None:
            total_reward, _ = self.reward_shaper.calculate_reward(
                response=response,
                prompt="",  # Not needed for synthetic data
                base_quality=response_quality,
                expected_answer=expected_answer
            )
            return float(total_reward * self.reward_scale)
        
        # Fallback to original calculation
        thinking_reward = np.clip(
            (thinking_steps - self.min_thinking_steps) / 
            (self.max_thinking_steps - self.min_thinking_steps),
            0, 1
        )
        
        total_reward = (thinking_reward * 0.4 + response_quality * 0.6) * self.reward_scale
        return float(total_reward)
    
    def generate_batch(
        self,
        batch_size: int,
        task_prompts: List[str],
        expected_answers: Optional[List[str]] = None
    ) -> List[Dict]:  # Change return type to List[Dict]
        """Generate a batch of synthetic examples with proper tokenization."""
        examples = []
        
        for i, task_prompt in enumerate(task_prompts[:batch_size]):
            # Randomly select base prompt
            base_prompt = random.choice(self.base_prompts)
            
            # Generate random number of thinking steps
            n_steps = random.randint(self.min_thinking_steps, self.max_thinking_steps)
            thinking_sequence = self._generate_thinking_steps(n_steps)
            
            # Combine prompts
            full_prompt = f"{task_prompt}\n\n{base_prompt}\n{thinking_sequence}"
            
            # Generate synthetic response (placeholder)
            response = f"After careful consideration, [RESPONSE]"
            
            # Calculate reward with correctness bonus if expected answer is provided
            expected_answer = expected_answers[i] if expected_answers and i < len(expected_answers) else None
            reward = self._calculate_reward(
                thinking_steps=n_steps,
                response_quality=random.random(),
                response=response,
                expected_answer=expected_answer
            )
            
            # Tokenize the full text
            full_text = f"{full_prompt}\n{response}"
            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create processed example with tokenization
            examples.append({
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": tokenized["input_ids"].squeeze(0).clone(),
                "reward": reward,
                "is_synthetic": True,
                "input": full_prompt,
                "output": response
            })
        
        return examples

def integrate_synthetic_data(
    original_dataset: List[Dict],
    synthetic_ratio: float = 0.2,
    generator: SyntheticDataGenerator = None,
    tokenizer: PreTrainedTokenizer = None,
    max_length: int = 512  # Add max_length parameter
) -> List[Dict]:
    """Integrate synthetic examples into the training dataset."""
    if generator is None:
        assert tokenizer is not None, "Must provide tokenizer if generator not provided"
        generator = SyntheticDataGenerator(tokenizer, max_length=max_length)
    
    # Calculate number of synthetic examples to generate
    n_synthetic = int(len(original_dataset) * synthetic_ratio)
    
    # Extract prompts from original dataset
    original_prompts = [d["input"] for d in original_dataset[:n_synthetic]]
    
    # Generate synthetic examples (now properly tokenized)
    synthetic_data = generator.generate_batch(n_synthetic, original_prompts)
    
    # Combine datasets
    combined_dataset = original_dataset + synthetic_data
    random.shuffle(combined_dataset)
    
    return combined_dataset 