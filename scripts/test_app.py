"""
Application testing script
Run this to test the app functionality before deployment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_model import ArabicLSTMPredictor
from src.utils.validation import validate_arabic_text

def test_model_loading():
    """Test model loading functionality"""
    print("🧪 Testing model loading...")
    
    config = {
        "path": "./models/pytorch_model.bin",
        "config_path": "./models/config.json",
        "tokenizer_path": "./models/tokenizer.json",
        "sequence_length": 10
    }
    
    predictor = ArabicLSTMPredictor(config)
    status = predictor.load_model()
    print(f"Status: {status}")
    
    return predictor.loaded

def test_predictions():
    """Test prediction functionality"""
    print("\n🧪 Testing predictions...")
    
    config = {"sequence_length": 10}
    predictor = ArabicLSTMPredictor(config)
    predictor.load_model()
    
    test_texts = [
        "اللغة العربية",
        "التعلم الآلي",
        "الذكاء الاصطناعي"
    ]
    
    for text in test_texts:
        predictions = predictor.predict_next_token(text, top_k=3)
        print(f"Text: '{text}' -> {[p['token'] for p in predictions[:3]]}")

def test_generation():
    """Test text generation"""
    print("\n🧪 Testing text generation...")
    
    config = {"sequence_length": 10}
    predictor = ArabicLSTMPredictor(config)
    predictor.load_model()
    
    seed_texts = ["اللغة", "التكنولوجيا"]
    
    for seed in seed_texts:
        generated = predictor.generate_text(seed, max_length=10, temperature=0.8)
        print(f"Seed: '{seed}' -> Generated: '{generated}'")

def test_validation():
    """Test input validation"""
    print("\n🧪 Testing validation...")
    
    test_cases = [
        ("اللغة العربية جميلة", True),
        ("Hello Arabic", False),
        ("", False),
        ("اللغة العربية " * 100, False)  # Too long
    ]
    
    for text, expected in test_cases:
        is_valid, message = validate_arabic_text(text)
        status = "✅" if is_valid == expected else "❌"
        print(f"{status} '{text[:20]}...': {message}")

def main():
    """Run all tests"""
    print("🚀 Starting Arabic LSTM App Tests")
    print("=" * 50)
    
    try:
        # Test model loading
        model_loaded = test_model_loading()
        
        if model_loaded:
            # Test predictions and generation
            test_predictions()
            test_generation()
        else:
            print("⚠️ Model not loaded, skipping prediction tests")
        
        # Test validation (always works)
        test_validation()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)