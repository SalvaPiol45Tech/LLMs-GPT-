import math
import torch
import torch.nn as nn 

class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) for better long-range dependencies"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequency basis
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_len=None):
        """
        x: (batch, seq_len, dim)
        Returns: rotary embeddings to apply to attention queries/keys
        """
        seq_len = seq_len or x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) # type: ignore
        
        # Compute frequencies: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Duplicate to full dim: (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Convert to rotation matrices (complex exponential)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE"""
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Causal mask: upper triangular (attend only to past tokens)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        Q = self._apply_rope(Q, cos, sin)
        K = self._apply_rope(K, cos, sin)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, num_heads, seq_len, seq_len)
        
        # Apply causal mask
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len], float('-inf')) # type: ignore
        
        # Softmax + dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to V
        output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        
        # Reshape back to (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(output)
        return output
    
    def _apply_rope(self, x, cos, sin):
        """Apply rotary embeddings to query/key"""
        # x: (batch, num_heads, seq_len, head_dim)
        # cos, sin: (seq_len, head_dim)
        x_rot = (x * cos.unsqueeze(0).unsqueeze(0)) + (self._rotate_half(x) * sin.unsqueeze(0).unsqueeze(0))
        return x_rot
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    def __init__(self, d_model, num_heads, d_ff=3072, dropout=0.1):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm and residual
        x = x + self.attn(self.ln1(x))
        
        # Feed-forward with pre-norm and residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class GPTLike(nn.Module):
    """ChatGPT-like language model: 125M parameters"""
    def __init__(
        self,
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding (learned)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output projection to vocab
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie token embedding and output projection weights
        self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids, return_loss=False, targets=None):
        """
        input_ids: (batch, seq_len) - token indices
        targets: (batch, seq_len) - for computing loss (optional)
        Returns: logits (batch, seq_len, vocab_size) or (loss, logits)
        """
        batch, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_emb(input_ids)  # (batch, seq_len, d_model)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions)  # (batch, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if return_loss and targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        
        if return_loss:
            return loss, logits
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text from input_ids
        input_ids: (batch, seq_len)
        Returns: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop input if exceeds max context
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # Get predictions
            logits = self.forward(input_ids)  # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]  # type: ignore # Take last token: (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            original_logits = logits.clone()
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')

            # If filtering removes every token in a row, restore the original logits for that row.
            valid_logits = torch.isfinite(logits).any(dim=-1, keepdim=True)
            logits = torch.where(valid_logits, logits, original_logits)
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if (probs.sum(dim=-1) <= 0).any():
                fallback_rows = probs.sum(dim=-1) <= 0
                logits[fallback_rows] = original_logits[fallback_rows]
                probs = torch.softmax(logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================================================
# TEST: Initialize and forward pass
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = GPTLike(
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=2048,
    ).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1e6:.1f}M parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    
    logits = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"✓ Forward pass successful!")
    
    # Test generation
    prompt = torch.tensor([[100, 200, 300]]).to(device)
    generated = model.generate(prompt, max_new_tokens=50)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"✓ Generation successful!")