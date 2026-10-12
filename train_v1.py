"""
Training Script - Fixed Version
Run: python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import os

from gpt_model import GPTLike
from tokenizer import CharacterTokenizer

# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Model - SMALL (10.9M for learning)
    vocab_size = None
    d_model = 384
    num_layers = 6
    num_heads = 6
    d_ff = 1536
    max_seq_len = 512
    dropout = 0.1
    
    # Training
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 2
    weight_decay = 1e-5
    grad_clip = 1.0
    
    # Data
    data_file = "./data/train.txt"
    
    # Checkpointing
    checkpoint_dir = "./checkpoints"
    save_every = 100
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = None
    vocab_size = None

config = Config()
os.makedirs(config.checkpoint_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================================================================
# Create Training Data
# ============================================================================
def create_training_data():
    """Create sample mathematical reasoning data"""
    examples = [
        # Addition
        """Question: Tom has 3 apples and gets 2 more. How many apples does he have?
Reasoning: Tom starts with 3 apples. He gets 2 more apples. So we calculate 3 + 2 = 5.
Answer: 5
""",
        """Question: Lisa has 7 books and buys 4 more. How many books does she have?
Reasoning: Lisa starts with 7 books. She buys 4 more books. So we calculate 7 + 4 = 11.
Answer: 11
""",
        # Subtraction
        """Question: John has 10 dollars and spends 3 dollars. How much money remains?
Reasoning: John starts with 10 dollars. He spends 3 dollars. So we calculate 10 - 3 = 7.
Answer: 7
""",
        """Question: Sarah has 15 candies and gives 6 away. How many candies remain?
Reasoning: Sarah starts with 15 candies. She gives away 6 candies. So we calculate 15 - 6 = 9.
Answer: 9
""",
        # Multiplication
        """Question: There are 4 boxes with 3 balls in each box. How many balls are there?
Reasoning: There are 4 boxes and each box has 3 balls. So we calculate 4 * 3 = 12.
Answer: 12
""",
        """Question: There are 5 bags with 6 apples in each bag. How many apples are there?
Reasoning: There are 5 bags and each bag contains 6 apples. So we calculate 5 * 6 = 30.
Answer: 30
""",
        # Division
        """Question: 12 candies are shared equally between 3 children. How many candies does each child get?
Reasoning: We have 12 candies and 3 children. We divide 12 by 3. So 12 / 3 = 4.
Answer: 4
""",
        """Question: 20 apples are placed equally into 5 boxes. How many apples are in each box?
Reasoning: We have 20 apples and 5 boxes. We divide 20 by 5. So 20 / 5 = 4.
Answer: 4
""",
    ]

    # Repeat to create enough data
    sample_text = "\n".join(examples) * 1000
    
    with open(config.data_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    file_size = os.path.getsize(config.data_file)
    print(f"✓ Created training data: {file_size / 1024:.1f} KB")
    return True


# ============================================================================
# Dataset
# ============================================================================
class SimpleTextDataset(Dataset):
    """Load text and create sequences"""
    def __init__(self, file_path, tokenizer, max_seq_len):
        print(f"\n📂 Loading {file_path}...")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        print(f"✓ Text length: {len(text):,} characters")
        
        # Tokenize
        print("🔤 Tokenizing...")
        self.tokens = tokenizer.encode(text)
        print(f"✓ Total tokens: {len(self.tokens):,}")
        
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return max(1, len(self.tokens) - self.max_seq_len - 1)
    
    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.max_seq_len]
        labels = self.tokens[idx + 1:idx + self.max_seq_len + 1]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return input_ids, labels


# ============================================================================
# Training Functions
# ============================================================================
def train_step(model, batch, optimizer, config):
    """Single training step"""
    input_ids, labels = batch
    input_ids = input_ids.to(config.device)
    labels = labels.to(config.device)
    
    # Forward
    loss, _ = model(input_ids, return_loss=True, targets=labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def validate(model, dataloader, config, num_batches=5):
    """Validation loop"""
    model.eval()
    losses = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        input_ids, labels = batch
        input_ids = input_ids.to(config.device)
        labels = labels.to(config.device)
        
        loss, _ = model(input_ids, return_loss=True, targets=labels)
        losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else 0


def save_checkpoint(model, optimizer, epoch, step, config):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
        }
    }
    path = os.path.join(config.checkpoint_dir, f"checkpoint_e{epoch}_s{step}.pt")
    torch.save(checkpoint, path)
    print(f"  💾 Saved: {path}")


# ============================================================================
# Main Training
# ============================================================================
def main():
    print("=" * 70)
    print("🚀 Training LLM from Scratch")
    print("=" * 70)
    print(f"📱 Device: {config.device}")
    print(f"🧠 Model: {config.d_model}D, {config.num_layers} layers")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"📏 Max seq len: {config.max_seq_len}")
    print("=" * 70 + "\n")
    
    # Create data if needed
    if not os.path.exists(config.data_file):
        print("📝 No training data found!")
        print("Creating sample data...\n")
        create_training_data()
    else:
        file_size = os.path.getsize(config.data_file)
        print(f"✓ Found data: {config.data_file} ({file_size / 1024:.1f} KB)\n")
    
    # Initialize tokenizer
    print("🔤 Building tokenizer...")
    tokenizer = CharacterTokenizer()
    
    with open(config.data_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    tokenizer.build_vocab(text)
    config.vocab_size = tokenizer.vocab_size # pyright: ignore[reportAttributeAccessIssue]
    print(f"✓ Vocab size: {config.vocab_size}\n")
    
    # Initialize model
    print("🧠 Initializing model...")
    model = GPTLike(
        vocab_size=config.vocab_size, # pyright: ignore[reportArgumentType]
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {total_params / 1e6:.1f}M parameters\n")
    
    # Create datasets
    print("📊 Creating dataset...")
    dataset = SimpleTextDataset(config.data_file, tokenizer, config.max_seq_len)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Val: {len(val_dataset)} samples\n")
    
    # Optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Training loop
    print("▶️  Starting training...\n")
    global_step = 0
    
    try:
        for epoch in range(config.num_epochs):
            print(f"{'=' * 70}")
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"{'=' * 70}")
            model.train()
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    loss = train_step(model, batch, optimizer, config)
                    epoch_losses.append(loss)
                    global_step += 1
                    
                    if (batch_idx + 1) % 100 == 0:
                        avg_loss = sum(epoch_losses[-20:]) / len(epoch_losses[-20:])
                        print(f"  Batch {batch_idx + 1:3d}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                    
                    # Validate & save
                    if global_step % config.save_every == 0:
                        val_loss = validate(model, val_loader, config, num_batches=3)
                        print(f"  >>> Step {global_step}: Val loss: {val_loss:.4f}")
                        save_checkpoint(model, optimizer, epoch, global_step, config)
                
                except Exception as e:
                    print(f"  ❌ Error in batch {batch_idx}: {e}")
                    continue
            
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"✓ Epoch {epoch + 1} done. Avg loss: {avg_epoch_loss:.4f}\n")
    
    except KeyboardInterrupt:
        print("\n\n✓ Training interrupted by user")
        save_checkpoint(model, optimizer, epoch, global_step, config)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'=' * 70}")
    print(f"✓ Training complete! Total steps: {global_step}")
    print(f"📁 Checkpoints: {config.checkpoint_dir}")
    print(f"{'=' * 70}")
    
    # Test generation
    print("\n📝 Testing Generation...")
    print("-" * 70)
    
    model.eval()
    prompt = "Question"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens]).to(config.device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=80,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"\nGenerated:\n{generated_text}")
    print("-" * 70)


if __name__ == "__main__":
    main()