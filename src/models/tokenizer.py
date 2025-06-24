"""
Arabic Text Tokenizer
Handles Arabic text preprocessing and vocabulary management
"""

import re
from typing import List, Dict

class ArabicTokenizer:
    """Specialized tokenizer for Arabic text processing"""
    
    def __init__(self):
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.vocab_built = False
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
    
    def clean_arabic_text(self, text: str) -> str:
        """Clean and normalize Arabic text"""
        # Remove diacritics (tashkeel marks)
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-Arabic characters
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        
        # Keep only Arabic characters and basic punctuation
        pattern = r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\u060C\u061B\u061F\u0640]'
        text = re.sub(pattern, '', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Split Arabic text into tokens"""
        cleaned_text = self.clean_arabic_text(text)
        
        # Find Arabic word sequences
        pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+'
        tokens = re.findall(pattern, cleaned_text)
        
        return [token for token in tokens if token.strip()]
    
    def encode(self, text: str) -> List[int]:
        """Convert text to sequence of token indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        tokens = self.tokenize(text)
        return [self.word_to_idx.get(token, self.word_to_idx.get(self.UNK_TOKEN, 1)) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Convert sequence of indices back to text"""
        tokens = []
        for idx in indices:
            token = self.idx_to_word.get(idx, self.UNK_TOKEN)
            if token not in [self.PAD_TOKEN, self.UNK_TOKEN]:
                tokens.append(token)
        return ' '.join(tokens)
    
    def load_from_dict(self, tokenizer_data: dict):
        """Load tokenizer from saved data"""
        self.word_to_idx = tokenizer_data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in tokenizer_data['idx_to_word'].items()}
        self.vocab_size = tokenizer_data['vocab_size']
        
        if 'special_tokens' in tokenizer_data:
            special = tokenizer_data['special_tokens']
            self.PAD_TOKEN = special.get('pad_token', '<PAD>')
            self.UNK_TOKEN = special.get('unk_token', '<UNK>')
            self.START_TOKEN = special.get('start_token', '<START>')
            self.END_TOKEN = special.get('end_token', '<END>')
        
        self.vocab_built = True