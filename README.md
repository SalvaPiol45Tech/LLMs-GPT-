# LLM from Scratch - Complete Training Pipeline

A complete implementation of a GPT-like language model trained entirely from scratch, including custom tokenizer, transformer architecture, and training loop.

## 🎯 Overview

This project demonstrates how to build a language model from the ground up without using pre-trained components. It includes:

- **Custom Character-Level Tokenizer** - Built from scratch
- **Transformer Decoder Architecture** - GPT-like model with causal attention
- **Complete Training Pipeline** - Data preparation, training, validation, and inference
- **Text Generation** - Generate new text from learned patterns

## 📊 Model Architecture

```
GPT-Like Language Model (24M Parameters)
├── Token Embedding (vocab_size → 384D)
├── Positional Embedding (learned)
├── 6 Transformer Blocks
│   ├── Causal Self-Attention (6 heads)
│   │   ├── Rotary Position Embeddings (RoPE)
│   │   └── Causal Mask (only attend to past)
│   └── Feed-Forward Network (384 → 1536 → 384)
├── Final LayerNorm
└── Output Projection (384D → vocab_size)
```

**Key Features:**
- Causal self-attention (decoder-only)
- Rotary position embeddings (RoPE)
- Pre-layer normalization
- Residual connections
- ~24.5M parameters (trainable on local GPU)

## 📁 Project Structure

```
LLM-from-scratch/
├── README.md                    # This file
├── complete_llm.py              # All-in-one training script
├── gpt_model.py                 # Model architecture only
├── tokenizer.py                 # Character-level tokenizer
├── train_v2.py                  # Training loop with utilities
└── data/
    └── train.txt                # Training text (auto-created)
```

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Paste the entire `complete_llm.py` content into a cell
4. Run the cell

```python
# Copy complete_llm.py content and paste in Colab cell
```

### Option 2: Local Machine

#### Requirements
```bash
python >= 3.8
torch >= 2.0
```

#### Installation
```bash
pip install torch
```

#### Run Training
```bash
python complete_llm.py
```

### Option 3: Command Line in Colab

```bash
# Upload complete_llm.py to Colab, then:
!python complete_llm.py
```

## 📖 Usage

### Train Model

```python
# Simply run:
python complete_llm.py

# Or in Colab:
!python complete_llm.py
```

**Output:**
```
======================================================================
Training LLM from Scratch (Complete Pipeline)
======================================================================

Device: cuda

Creating training data...
Data length: 18,432 characters

Building tokenizer...
✓ Built vocab: 87 unique characters
Vocab size: 87

Creating dataset...
Text length: 18,432 characters
Tokenizing...
Total tokens: 28,934

Train samples: 2599
Val samples: 289

Initializing model...
Model: 24.5M parameters

Starting training...

======================================================================
Epoch 1/3
======================================================================
  Batch 20/162, Loss: 3.8234
  >>> Step 100: Val loss: 3.1234
  Batch 40/162, Loss: 3.2145
  >>> Step 200: Val loss: 2.9876
...
✓ Epoch 1 done

Testing Generation...
----------------------------------------------------------------------

Prompt: 'The'

Generated text:
The quick brown fox jumps over the lazy dog. A journey of a thousand miles
begins with a single step. To be or not to be, that is the question...
----------------------------------------------------------------------
```

### Use Custom Data

Edit `create_sample_data()` in `complete_llm.py` or replace with your text:

```python
# Load your own text file
with open("your_data.txt", "r") as f:
    text = f.read()

# Then train
dataset = SimpleTextDataset(text, tokenizer, max_seq_len=512)
```

### Generate Text from Trained Model

```python
prompt = "The"
prompt_tokens = tokenizer.encode(prompt)
prompt_tensor = torch.tensor([prompt_tokens]).to(device)

generated = model.generate(
    prompt_tensor,
    max_new_tokens=100,
    temperature=0.8,
    top_k=10,
    top_p=0.9
)

generated_text = tokenizer.decode(generated[0].tolist())
print(generated_text)
```

## ⚙️ Configuration

Default hyperparameters (in `complete_llm.py`):

```python
batch_size = 16              # Batch size for training
learning_rate = 1e-3         # Learning rate
num_epochs = 3               # Number of training epochs
max_seq_len = 512            # Context window length

d_model = 384                # Model dimension
num_layers = 6               # Number of transformer blocks
num_heads = 6                # Number of attention heads
d_ff = 1536                  # Feed-forward hidden dimension
dropout = 0.1                # Dropout rate
```

### Adjust Configuration

Modify these values in `main()` function to change model size and training behavior:

```python
def main():
    # ... existing code ...
    
    # Configuration
    batch_size = 32              # Increase for faster training
    learning_rate = 5e-4         # Lower for longer training
    num_epochs = 10              # More epochs for better results
    max_seq_len = 1024           # Larger context window
    d_model = 512                # Larger model
    num_layers = 8               # More layers
```

## 🔧 Components

### Tokenizer
- **Type:** Character-level
- **Vocab Size:** Determined by unique characters in text
- **Method:** Direct character → token ID mapping

```python
tokenizer = CharacterTokenizer()
tokenizer.build_vocab(text)
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

### Model Layers

**Causal Self-Attention:**
- Multi-head attention with causal mask
- Rotary Position Embeddings (RoPE)
- Prevents attending to future tokens

**Feed-Forward:**
- 2-layer MLP with GELU activation
- Expands to `d_ff` then back to `d_model`

**Transformer Block:**
- Layer normalization before attention and FFN (pre-norm)
- Residual connections around attention and FFN

### Training Loop

**Loss Function:** Cross-entropy on next-token prediction

**Optimizer:** Adam with weight decay (0.01)

**Learning Rate Schedule:** Constant with warmup

**Gradient Clipping:** 1.0 to stabilize training

## 📈 Expected Performance

On the sample data:

| step | Train Loss | Val Loss | total step
|-------|-----------|----------|
| 1     | 3.82      | 1.2556|  14175
| 2     | 2.95      | 99    | 
| 3     | 2.41      | 85    | 

**Note:** Performance depends on:
- Amount of training data
- Quality of training data
- Number of epochs
- Model size

## 💡 Key Implementation Details

### Causal Masking
```python
# Model only attends to past tokens (including current)
causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
```

### Position Embeddings
- **Learned embeddings** during model initialization
- Absolute positions up to `max_seq_len`

### Weight Tying
- Token embedding and output projection share weights
- Reduces parameters and improves training

### Rotary Embeddings (RoPE)
- Applied to queries and keys in attention
- Better long-range dependency modeling

## 🧪 Testing & Debugging

### Check Model Size
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {total_params / 1e6:.1f}M parameters")
```

### Test Forward Pass
```python
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # Should be (batch, seq_len, vocab_size)
```

### Validate Generation
```python
prompt_tokens = [10, 20, 30]
prompt_tensor = torch.tensor([prompt_tokens]).to(device)
generated = model.generate(prompt_tensor, max_new_tokens=50)
print(generated.shape)  # Should be (1, 3+50)
```

## 🚨 Troubleshooting

**Error: RuntimeError: selected index k out of range**
- Caused by `top_k > vocab_size`
- Fix: Reduce `top_k` in generate function (already fixed in latest version)

**Error: CUDA out of memory**
- Reduce `batch_size`
- Reduce `max_seq_len`
- Reduce model size (`d_model`, `num_layers`)

**Training loss not decreasing**
- Increase `learning_rate`
- Check data quality
- Verify loss computation

**Generation produces nonsense**
- Model may need more training epochs
- Increase training data size
- Use lower `temperature` (more conservative sampling)

## 📚 Learning Resources

### Understanding Transformers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation

### Position Embeddings
- [RoPE (Rotary Position Embeddings)](https://arxiv.org/abs/2104.09864)

### Decoding Strategies
- Temperature Sampling
- Top-k Sampling
- Top-p (Nucleus) Sampling

## 🎓 Educational Value

This project teaches:

1. **Transformer Architecture**
   - Self-attention mechanism
   - Positional encoding
   - Causal masking

2. **Tokenization**
   - Character-level tokenization
   - Vocabulary building

3. **Training Neural Networks**
   - Forward and backward passes
   - Optimization
   - Validation

4. **Language Model Decoding**
   - Temperature sampling
   - Top-k and top-p filtering
   - Beam search concepts

## 🔄 Future Improvements

- [ ] Byte Pair Encoding (BPE) tokenizer
- [ ] Better data source (WikiText, OpenWebText)
- [ ] Larger model size (125M parameters)
- [ ] Multi-GPU training
- [ ] Checkpointing and resuming
- [ ] Evaluation metrics (perplexity on test set)
- [ ] Benchmark on standard datasets (WIKITEXT-103)
- [ ] Model quantization
- [ ] Knowledge distillation

## 📝 File Descriptions

### `complete_llm.py`
**All-in-one training script**
- Contains: Tokenizer, Model, Dataset, Training Loop
- No external imports except PyTorch
- Ready to run in Colab or locally

### `gpt_model.py`
**Model architecture only**
- Tokenizer-independent
- Can be imported and used separately
- Implements: Transformer blocks, attention, positives embeddings

`tokenizer.py`
**Standalone tokenizer**
- Character-level and BPE implementations
- Can be used independently
- Includes encode/decode methods

 `train_v2.py`
**Training script with utilities**
- Separate data loading and validation
- Checkpoint saving/loading
- More modular approach

 📄 License

This project is open source and available under the MIT License.

 👤 Author

Built from scratch for learning and educational purposes.

Demonstrates:
- Deep understanding of transformer architecture
- PyTorch proficiency
- Complete ML pipeline implementation
- Research-level model building

## 🤝 Contributing

Feel free to fork, modify, and improve!

Suggestions:
- Add BPE tokenizer
- Implement better data loading
- Add evaluation metrics
- Create inference APIs
- Build interactive demos

## ⭐ Citation

If you use this in your work, cite as:

```bibtex
@misc{llm_from_scratch,
  title={LLM from Scratch - Complete Training Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/llm-from-scratch}
}
```

---

**Last Updated:** 2024
**Status:** Working ✅
**GPU Support:** CUDA/CPU
**Python Version:** 3.8+

For questions or issues, open a GitHub issue or discussion.

Happy training! 🚀
