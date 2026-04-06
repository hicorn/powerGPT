"""
Unit tests for PowerGPT model.

Tests:
- Model initialization
- Forward pass
- Loss computation
- Generation
- KV cache
- Gradient checkpointing
"""

import unittest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from powergpt.config import ModelArchConfig
from powergpt.model import GPT


class TestGPTModel(unittest.TestCase):
    """Test cases for GPT model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test configuration and model."""
        cls.config = ModelArchConfig()
        cls.config.n_layer = 2
        cls.config.n_head = 2
        cls.config.n_embd = 64
        cls.config.block_size = 128
        cls.config.vocab_size = 1000
        cls.config.flash_attention = False  # Disable for CPU testing
        cls.config.gradient_checkpointing = False
        
        cls.model = GPT(cls.config)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model.to(cls.device)
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config.n_layer, 2)
        self.assertEqual(self.model.config.n_head, 2)
        self.assertEqual(self.model.config.n_embd, 64)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)
        print(f"Model has {total_params:,} parameters")
    
    def test_forward_pass(self):
        """Test forward pass returns correct shapes."""
        batch_size = 2
        seq_len = 64
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        logits, loss, _ = self.model(x)
        
        # Check output shapes
        self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNone(loss)  # No targets provided
    
    def test_forward_pass_with_loss(self):
        """Test forward pass with loss computation."""
        batch_size = 2
        seq_len = 64
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        y = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        logits, loss, _ = self.model(x, targets=y)
        
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0)
        self.assertLess(loss.item(), 20)  # Random should be ~log(vocab_size)
    
    def test_generation(self):
        """Test text generation."""
        batch_size = 1
        seq_len = 10
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        output = self.model.generate(
            x,
            max_new_tokens=20,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )
        
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], seq_len + 20)
    
    def test_generation_with_kv_cache(self):
        """Test generation with KV cache enabled."""
        batch_size = 1
        seq_len = 10
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        output = self.model.generate(
            x,
            max_new_tokens=20,
            use_kv_cache=True,
        )
        
        self.assertEqual(output.shape[1], seq_len + 20)
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing toggle."""
        self.model.gradient_checkpointing_enable()
        self.assertTrue(self.model._gradient_checkpointing)
        
        self.model.gradient_checkpointing_disable()
        self.assertFalse(self.model._gradient_checkpointing)
    
    def test_optimizer_creation(self):
        """Test optimizer configuration."""
        optimizer = self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            optimizer_type='adamw'
        )
        self.assertIsNotNone(optimizer)
    
    def test_training_step(self):
        """Test a single training step."""
        self.model.train()
        optimizer = self.model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95)
        )
        
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        y = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        optimizer.zero_grad()
        _, loss, _ = self.model(x, targets=y)
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with fixed seed."""
        torch.manual_seed(42)
        batch_size = 1
        seq_len = 5
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        output1 = self.model.generate(x, max_new_tokens=10, temperature=0.0)  # greedy
        output2 = self.model.generate(x, max_new_tokens=10, temperature=0.0)
        
        self.assertTrue(torch.equal(output1, output2))
    
    def test_repetition_penalty(self):
        """Test that repetition penalty affects output."""
        batch_size = 1
        seq_len = 5
        # Create input with repeated tokens
        x = torch.full((batch_size, seq_len), 42, dtype=torch.long).to(self.device)
        
        output_no_penalty = self.model.generate(
            x, max_new_tokens=20, temperature=0.0, repetition_penalty=1.0
        )
        output_with_penalty = self.model.generate(
            x, max_new_tokens=20, temperature=0.0, repetition_penalty=1.5
        )
        
        # With penalty, model should avoid repeating token 42
        # This is a weak test but checks that penalty doesn't crash
        self.assertIsNotNone(output_no_penalty)
        self.assertIsNotNone(output_with_penalty)
    
    def test_kv_cache_memory_limit(self):
        """Test that KV cache limiting works."""
        batch_size = 1
        seq_len = 10
        x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # Generate with small cache limit
        output = self.model.generate(
            x,
            max_new_tokens=100,
            use_kv_cache=True,
            max_kv_cache_tokens=50,
        )
        
        self.assertEqual(output.shape[1], seq_len + 100)
    
    def test_model_device(self):
        """Test model is on correct device."""
        device = next(self.model.parameters()).device
        self.assertEqual(device, self.device)
    
    def test_embedding_weight_tie(self):
        """Test that embedding and lm_head weights are tied."""
        self.assertIs(self.model.token_embedding.weight, self.model.lm_head.weight)


class TestModelArchConfig(unittest.TestCase):
    """Test cases for model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelArchConfig()
        self.assertEqual(config.n_layer, 12)
        self.assertEqual(config.n_head, 12)
        self.assertEqual(config.n_embd, 768)
    
    def test_head_dim_computation(self):
        """Test that head_dim is computed correctly."""
        config = ModelArchConfig()
        config.n_embd = 768
        config.n_head = 12
        config.__post_init__()
        self.assertEqual(config.head_dim, 64)
    
    def test_moe_config(self):
        """Test MoE configuration."""
        config = ModelArchConfig()
        config.use_moe = True
        config.num_experts = 8
        config.top_k_experts = 2
        config.__post_init__()
        self.assertTrue(config.use_moe)
    
    def test_invalid_moe_config(self):
        """Test that invalid MoE config raises error."""
        config = ModelArchConfig()
        config.use_moe = True
        config.num_experts = 1  # Invalid
        config.top_k_experts = 2
        with self.assertRaises(AssertionError):
            config.__post_init__()


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == '__main__':
    run_tests()