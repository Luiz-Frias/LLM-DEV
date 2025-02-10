import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Dict, Optional, Literal
import logging
import time
from transformers import BitsAndBytesConfig
import gc
from ..model.attention import SSMMLayer
from ..memory.memory_components import MonolithIntake, EpisodicMemory
from ..memory.hierarchical_memory import HierarchicalMemoryTree

class TextGenerator:
    """Class for generating text using a trained language model with SSSM and memory optimizations."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        max_length: int = 512,  # Changed from 1024 to match training
        quantization: Optional[Literal["none", "8bit", "4bit", "4bit_groupwise"]] = "none",
        optimize_memory: bool = True,
        ssmm_config: Optional[Dict] = None,
        use_memory_components: bool = True
    ):
        """Initialize the generator.
        
        Args:
            model_path: Path to the saved model
            device: Device to run inference on ('cuda', 'cpu', 'mps', or None for auto)
            max_length: Maximum length of generated sequences
            quantization: Quantization method to use
            optimize_memory: Whether to optimize memory usage during inference
            ssmm_config: Configuration for Selective State Space Model
            use_memory_components: Whether to use advanced memory components
        """
        # Setup device
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure quantization
        quantization_config = None
        if quantization != "none" and self.device.type == "cuda":
            if quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif quantization in ["4bit", "4bit_groupwise"]:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_groupwise=(quantization == "4bit_groupwise")
                )
        
        # Load model with appropriate settings
        torch_dtype = torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32
        
        # Clear cache before loading model
        self._optimize_memory()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if self.device.type == "cuda" else None,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        self.max_length = max_length
        self.optimize_memory = optimize_memory
        
        # Initialize SSSM if config provided
        self.ssmm = None
        if ssmm_config is not None:
            self.ssmm = SSMMLayer(
                hidden_size=self.model.config.hidden_size,
                state_size=ssmm_config.get("state_size", 16),
                spike_threshold=ssmm_config.get("spike_threshold", 0.1),
                dt_min=ssmm_config.get("dt_min", 0.001),
                dt_max=ssmm_config.get("dt_max", 0.1)
            ).to(self.device)
        
        # Initialize memory components if requested
        self.memory_components = None
        if use_memory_components:
            self.memory_components = {
                "tree": HierarchicalMemoryTree(
                    base_capacity=5,
                    error_threshold=0.1,
                    imbalance_threshold=2.0,
                    device=self.device
                ),
                "episodic": EpisodicMemory(
                    input_dim=self.model.config.hidden_size,
                    device=self.device
                ),
                "intake": MonolithIntake(
                    batch_size=2,
                    device=self.device
                )
            }
        
        # Log configuration
        logging.info(f"Loaded model from {model_path}")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Quantization: {quantization}")
        logging.info(f"Model dtype: {self.model.dtype}")
        logging.info(f"SSSM enabled: {self.ssmm is not None}")
        logging.info(f"Memory components enabled: {self.memory_components is not None}")
        
        # Calculate and log memory usage
        if self.device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(f"GPU memory used: {memory_used:.2f} MB")
        elif self.device.type == "mps":
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_used = torch.mps.current_allocated_memory() / 1024**2
                logging.info(f"MPS memory used: {memory_used:.2f} MB")
    
    def _optimize_memory(self):
        """Optimize memory usage during generation."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        gc.collect()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        use_ssmm: bool = True,
        **kwargs
    ) -> Dict[str, Union[str, List[str], float]]:
        """Generate text based on the prompt using SSSM and memory optimizations."""
        start_time = time.time()
        
        # Optimize memory before generation
        if self.optimize_memory:
            self._optimize_memory()
        
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Set max_new_tokens if not provided
        if max_new_tokens is None:
            max_new_tokens = min(self.max_length - inputs["input_ids"].shape[1], 100)
        
        # Generate with SSSM if enabled
        with torch.no_grad():
            # Get initial hidden states
            outputs = self.model(
                inputs["input_ids"],
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            
            # Apply SSSM if available and requested
            if self.ssmm is not None and use_ssmm:
                hidden_states, _ = self.ssmm(hidden_states)
            
            # Update memory components if available
            if self.memory_components is not None:
                self.memory_components["intake"].add_to_batch(prompt, hidden_states.mean(dim=1))
                if len(self.memory_components["intake"].current_batch) >= 2:
                    traces = self.memory_components["intake"].process_batch()
                    self.memory_components["episodic"].add_traces(traces)
            
            # Generate tokens
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                **kwargs
            )
        
        # Decode outputs
        generated_sequences = []
        for output in outputs:
            # Remove the prompt from the generated text
            prompt_length = inputs["input_ids"].shape[1]
            generated_tokens = output[prompt_length:]
            
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            generated_sequences.append(generated_text)
        
        generation_time = time.time() - start_time
        
        # Optimize memory after generation
        if self.optimize_memory:
            self._optimize_memory()
        
        # Get memory statistics if components are enabled
        memory_stats = None
        if self.memory_components is not None:
            memory_stats = {
                "tree_stats": self.memory_components["tree"].get_statistics(),
                "episodic_stats": self.memory_components["episodic"].get_statistics()
            }
        
        return {
            "generated_text": generated_sequences[0] if num_return_sequences == 1 else generated_sequences,
            "prompt": prompt,
            "generation_time": generation_time,
            "memory_stats": memory_stats
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None,
        use_ssmm: bool = True,
        **kwargs
    ) -> Dict[str, Union[str, float]]:
        """Generate a response in a chat-like format using SSSM and memory optimizations."""
        # Format conversation
        conversation = []
        if system_prompt:
            conversation.append(f"System: {system_prompt}\n")
        
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            conversation.append(f"{role}: {content}")
        
        # Join with newlines and add assistant prompt
        prompt = "\n".join(conversation) + "\nAssistant:"
        
        # Generate response with SSSM
        response = self.generate(prompt, use_ssmm=use_ssmm, **kwargs)
        
        return {
            "response": response["generated_text"].strip(),
            "generation_time": response["generation_time"],
            "memory_stats": response.get("memory_stats")
        }

def main():
    """Example usage of the text generator with SSSM and memory optimizations."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # SSSM configuration
    ssmm_config = {
        "state_size": 16,
        "spike_threshold": 0.1,
        "dt_min": 0.001,
        "dt_max": 0.1
    }
    
    # Initialize generator with SSSM and memory components
    generator = TextGenerator(
        "models/final_model",
        ssmm_config=ssmm_config,
        use_memory_components=True
    )
    
    # Example prompts
    prompts = [
        "The future of artificial intelligence will",
        "Once upon a time in a distant galaxy",
        "The secret to happiness is"
    ]
    
    # Generate text with SSMM
    print("\nText Generation Examples:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = generator.generate(prompt, max_new_tokens=50)
        print(f"Generated: {result['generated_text']}")
        print(f"Time: {result['generation_time']:.2f}s")
        if result.get("memory_stats"):
            print("Memory Stats:", result["memory_stats"])
    
    # Chat example with SSSM
    print("\nChat Example:")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are three interesting facts about space?"}
    ]
    
    result = generator.chat(messages)
    print(f"\nResponse: {result['response']}")
    print(f"Time: {result['generation_time']:.2f}s")
    if result.get("memory_stats"):
        print("Memory Stats:", result["memory_stats"])

if __name__ == "__main__":
    main() 