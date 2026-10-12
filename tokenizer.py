import json
import os
from collections import defaultdict

# ============================================================================
# Character-Level Tokenizer (Simplest)
# ============================================================================
class CharacterTokenizer:
    """Simplest tokenizer: character-level"""
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, text):
        """Build vocab from text"""
        unique_chars = sorted(list(set(text)))
        self.char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(unique_chars)
        print(f"✓ Built vocab: {self.vocab_size} unique characters")
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.char_to_id[char] for char in text]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        return "".join([self.id_to_char[tid] for tid in token_ids])
    
    def save(self, path):
        """Save tokenizer"""
        data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"✓ Saved tokenizer to {path}")
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, "r") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in enumerate(data["char_to_id"].values())}
        self.vocab_size = data["vocab_size"]
        print(f"✓ Loaded tokenizer from {path}")


# ============================================================================
# BPE Tokenizer (More Sophisticated)
# ============================================================================
class BPETokenizer:
    """Byte Pair Encoding tokenizer - built from scratch"""
    def __init__(self, vocab_size=256, num_merges=1000):
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self.merges = {}  # Token pair -> new token
        self.vocab = {}   # Token ID -> token (bytes or merged pairs)
    
    def build_vocab(self, text):
        """Train BPE on text"""
        # Start with character-level vocab
        chars = set(text)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Tokenize text into bytes
        tokens = list(text.encode('utf-8'))
        
        print(f"Initial vocab size: 256")
        print(f"Starting BPE merging ({self.num_merges} merges)...")
        
        # Merging loop
        for merge_step in range(self.num_merges):
            # Count all adjacent pairs
            pairs = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            most_common = max(pairs, key=pairs.get) # type: ignore
            
            # Assign new token ID
            new_token_id = 256 + merge_step
            self.merges[most_common] = new_token_id
            vocab[new_token_id] = most_common
            
            # Merge in token sequence
            tokens = self._merge_tokens(tokens, most_common, new_token_id)
            
            if (merge_step + 1) % 100 == 0:
                print(f"  Merge {merge_step + 1}/{self.num_merges}, Vocab size: {256 + merge_step + 1}")
        
        self.vocab = vocab
        self.vocab_size = 256 + len(self.merges)
        print(f"✓ Built BPE vocab: {self.vocab_size} tokens")
    
    def _merge_tokens(self, tokens, pair, new_id):
        """Merge a pair in token sequence"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def encode(self, text):
        """Encode text using BPE"""
        # Start with byte-level encoding
        tokens = list(text.encode('utf-8'))
        
        # Apply all merges
        for pair, new_id in self.merges.items():
            tokens = self._merge_tokens(tokens, pair, new_id)
        
        return tokens
    
    def decode(self, token_ids):
        """Decode tokens back to text"""
        # Reconstruct byte sequence
        byte_sequence = []
        for tid in token_ids:
            if tid in self.vocab:
                token_bytes = self.vocab[tid]
                if isinstance(token_bytes, tuple):
                    # Reconstruct merged tokens
                    byte_sequence.extend(self._flatten_token(token_bytes))
                else:
                    byte_sequence.append(token_bytes[0])
        
        try:
            return bytes(byte_sequence).decode('utf-8', errors='ignore')
        except:
            return ""
    
    def _flatten_token(self, token):
        """Flatten nested token pairs to bytes"""
        if isinstance(token, tuple):
            result = []
            for t in token:
                result.extend(self._flatten_token(t))
            return result
        else:
            return [token]
    
    def save(self, path):
        """Save tokenizer"""
        # Convert merges to serializable format
        merges_list = []
        for (t1, t2), new_id in self.merges.items():
            merges_list.append((t1, t2, new_id))
        
        data = {
            "merges": merges_list,
            "vocab_size": self.vocab_size,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"✓ Saved BPE tokenizer to {path}")
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for t1, t2, new_id in data["merges"]:
            self.merges[(t1, t2)] = new_id
            vocab[new_id] = (t1, t2) # type: ignore
        
        self.vocab = vocab
        self.vocab_size = data["vocab_size"]
        print(f"✓ Loaded BPE tokenizer from {path}")


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Sample text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    This is a test of the tokenizer.
    Hello world! How are you today?
    """
    
    print("=" * 70)
    print("Testing Character-Level Tokenizer")
    print("=" * 70)
    
    char_tok = CharacterTokenizer()
    char_tok.build_vocab(sample_text)
    
    encoded = char_tok.encode("Hello")
    print(f"Encoded 'Hello': {encoded}")
    
    decoded = char_tok.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print("\n" + "=" * 70)
    print("Testing BPE Tokenizer")
    print("=" * 70)
    
    bpe_tok = BPETokenizer(vocab_size=256, num_merges=100)
    bpe_tok.build_vocab(sample_text)
    
    encoded = bpe_tok.encode("Hello world")
    print(f"Encoded 'Hello world': {encoded}")
    print(f"Number of tokens: {len(encoded)}")