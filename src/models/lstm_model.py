"""
LSTM Model Architecture and Predictor Class
"""

import torch
import torch.nn as nn
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .tokenizer import ArabicTokenizer

logger = logging.getLogger(__name__)

class ArabicLSTM(nn.Module):
    """LSTM Language Model for Arabic text processing"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 64, 
                 hidden_dim: int = 128, num_layers: int = 1, dropout: float = 0.2):
        super(ArabicLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """Forward pass through the network"""
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.output_layer(lstm_out)
        return output, hidden
    
    def init_hidden(self, batch_size: int):
        """Initialize hidden state"""
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))

class ArabicLSTMPredictor:
    """High-level interface for LSTM predictions"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model: Optional[ArabicLSTM] = None
        self.tokenizer: Optional[ArabicTokenizer] = None
        self.device = torch.device('cpu')
        self.loaded = False
        
    def load_model(self) -> str:
        """Load model, tokenizer, and configuration"""
        try:
            # Load model configuration
            config_path = Path(self.config.get("config_path", "models/config.json"))
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
            else:
                logger.warning("Config file not found, using defaults")
                model_config = self._get_default_model_config()
            
            # Load and initialize tokenizer
            tokenizer_path = Path(self.config.get("tokenizer_path", "models/tokenizer.json"))
            self.tokenizer = ArabicTokenizer()
            
            if tokenizer_path.exists():
                with open(tokenizer_path, 'r', encoding='utf-8') as f:
                    tokenizer_data = json.load(f)
                self.tokenizer.load_from_dict(tokenizer_data)
            else:
                logger.warning("Tokenizer file not found, creating dummy tokenizer")
                self.tokenizer = self._create_demo_tokenizer()
            
            # Initialize model
            self.model = ArabicLSTM(
                vocab_size=model_config['vocab_size'],
                embedding_dim=model_config['embedding_dim'],
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers']
            ).to(self.device)
            
            # Load trained weights
            model_path = Path(self.config.get("path", "models/pytorch_model.bin"))
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                status = "✅ Model loaded successfully with trained weights"
            else:
                logger.warning("Model weights not found, using random initialization")
                status = "⚠️ Demo mode: Using untrained model (upload model files for full functionality)"
            
            self.model.eval()
            self.loaded = True
            return status
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return f"❌ Failed to load model: {str(e)}"
    
    def _get_default_model_config(self) -> dict:
        """Default model configuration"""
        return {
            "vocab_size": 200,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 1
        }
    
    def _create_demo_tokenizer(self) -> ArabicTokenizer:
        """Create demo tokenizer for when files are missing"""
        tokenizer = ArabicTokenizer()
        
        # Basic Arabic vocabulary for demo
        vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        vocab.extend([
            'اللغة', 'العربية', 'التعلم', 'الآلي', 'الذكاء', 'الاصطناعي',
            'التكنولوجيا', 'الحاسوب', 'البرمجة', 'المستقبل', 'العلم',
            'المعرفة', 'التطوير', 'الإبداع', 'الابتكار', 'التقدم'
        ])
        
        tokenizer.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        tokenizer.idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        tokenizer.vocab_size = len(vocab)
        tokenizer.vocab_built = True
        
        return tokenizer
    
    def predict_next_token(self, text: str, top_k: int = 5) -> List[Dict]:
        """Predict next tokens with confidence scores"""
        if not self.loaded:
            return [{"token": "⚠️ Model not loaded", "probability": 0.0}]
        
        try:
            encoded = self.tokenizer.encode(text)
            if not encoded:
                return [{"token": "❌ Invalid input", "probability": 0.0}]
            
            input_seq = torch.tensor([encoded], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                hidden = self.model.init_hidden(1)
                outputs, _ = self.model(input_seq, hidden)
                
                last_predictions = outputs[0, -1, :]
                probabilities = torch.softmax(last_predictions, dim=0)
                
                top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(probabilities)))
                
                results = []
                for i in range(len(top_k_probs)):
                    token_idx = top_k_indices[i].item()
                    prob = top_k_probs[i].item()
                    token = self.tokenizer.idx_to_word.get(token_idx, self.tokenizer.UNK_TOKEN)
                    
                    if token not in [self.tokenizer.PAD_TOKEN, self.tokenizer.UNK_TOKEN]:
                        results.append({"token": token, "probability": prob})
                
                return results or [{"token": "❌ No valid predictions", "probability": 0.0}]
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [{"token": f"❌ Error: {str(e)}", "probability": 0.0}]
    
    def generate_text(self, seed_text: str, max_length: int = 15, temperature: float = 0.8) -> str:
        """Generate text with controllable creativity"""
        if not self.loaded:
            return "⚠️ Model not loaded"
        
        try:
            encoded = self.tokenizer.encode(seed_text)
            if not encoded:
                return "❌ Invalid seed text"
            
            generated = encoded.copy()
            sequence_length = self.config.get("sequence_length", 10)
            
            with torch.no_grad():
                hidden = self.model.init_hidden(1)
                
                for _ in range(max_length):
                    input_seq = torch.tensor([generated[-sequence_length:]], dtype=torch.long).to(self.device)
                    outputs, hidden = self.model(input_seq, hidden)
                    
                    last_output = outputs[0, -1, :] / temperature
                    probabilities = torch.softmax(last_output, dim=0)
                    next_token = torch.multinomial(probabilities, 1).item()
                    
                    # Stop conditions
                    if next_token == self.tokenizer.word_to_idx.get(self.tokenizer.END_TOKEN, -1):
                        break
                    if next_token == self.tokenizer.word_to_idx.get(self.tokenizer.PAD_TOKEN, 0):
                        continue
                    
                    generated.append(next_token)
            
            return self.tokenizer.decode(generated)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Generation failed: {str(e)}"