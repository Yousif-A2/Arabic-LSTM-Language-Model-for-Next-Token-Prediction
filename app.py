"""
Arabic LSTM Language Model - Gradio Application
Main entry point for the HuggingFace Spaces deployment
"""

import gradio as gr
import yaml
import logging
from pathlib import Path

# Import custom modules
from src.models.lstm_model import ArabicLSTMPredictor
from src.interface.prediction_tab import create_prediction_tab
from src.interface.generation_tab import create_generation_tab
from src.interface.info_tab import create_info_tab
from src.utils.validation import validate_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """Load application configuration"""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()

def get_default_config() -> dict:
    """Get default configuration if config file fails"""
    return {
        "app": {"title": "Arabic LSTM Language Model", "debug": False},
        "model": {"max_length": 30, "temperature_range": [0.1, 2.0]},
        "interface": {"theme": "soft", "examples": {"prediction": ["اللغة العربية"]}}
    }

def create_app() -> gr.Blocks:
    """Create the main Gradio application"""
    config = load_config()
    
    # Initialize model predictor
    predictor = ArabicLSTMPredictor(config["model"])
    
    # Load model
    startup_status = predictor.load_model()
    logger.info(f"Model startup: {startup_status}")
    
    # Custom CSS for Arabic text
    css = """
    .rtl { direction: rtl; text-align: right; }
    .arabic-text { 
        font-family: 'Amiri', 'Tahoma', 'Arial Unicode MS', sans-serif; 
        font-size: 18px; 
        line-height: 1.6;
    }
    .title { text-align: center; color: #2E8B57; }
    .gradient-text {
        background: linear-gradient(45deg, #2E8B57, #4169E1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    """
    
    # Create main application
    with gr.Blocks(
        css=css, 
        title=config["app"]["title"],
        theme=gr.themes.Soft()
    ) as app:
        
        # Header
        gr.Markdown(f"""
        # 🤖 {config["app"]["title"]}
        ### Interactive Arabic Text Prediction & Generation
        
        Explore Arabic language patterns with our LSTM-based model trained on real Arabic text!
        """, elem_classes="title gradient-text")
        
        # Create tabs
        with gr.Tabs():
            # Model Information Tab
            create_info_tab(predictor, config)
            
            # Next Token Prediction Tab
            create_prediction_tab(predictor, config)
            
            # Text Generation Tab
            create_generation_tab(predictor, config)
            
            # Instructions Tab
            with gr.Tab("📖 How to Use"):
                gr.Markdown("""
                ## 🎯 Getting Started
                
                ### 🔮 Next Token Prediction
                1. Enter Arabic text (2-8 words work best)
                2. Adjust number of predictions (1-10)
                3. Click "Predict" to see likely next words
                4. View results with confidence scores
                
                ### ✍️ Text Generation
                1. Provide Arabic seed text (1-5 words)
                2. Set maximum generation length
                3. Adjust temperature for creativity control
                4. Generate and explore different outputs
                
                ## 💡 Pro Tips
                
                **For Better Results:**
                - Use proper Arabic text without diacritics
                - Start with common Arabic words
                - Experiment with different temperature values
                - Try various seed text lengths
                
                **Temperature Guide:**
                - 🎯 **0.1-0.5**: Conservative, coherent text
                - ⚖️ **0.6-1.0**: Balanced creativity and coherence
                - 🎨 **1.1-2.0**: Creative, diverse output
                
                ## 🌟 Example Workflows
                
                **Educational Use:**
                - Input: "العلم" → Discover related vocabulary
                - Low temperature for grammatically correct text
                
                **Creative Writing:**
                - Use inspiring seed words
                - High temperature for unexpected combinations
                - Longer generation for storytelling
                
                **Research Applications:**
                - Test domain-specific vocabulary
                - Analyze model behavior on different text types
                - Compare predictions with linguistic expectations
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **🚀 Deployed on HuggingFace Spaces** | **Built with Gradio & PyTorch**
        
        💫 Advancing Arabic NLP through open-source collaboration
        """, elem_classes="title")
    
    return app

def main():
    """Main application entry point"""
    try:
        app = create_app()
        
        # Launch configuration for HuggingFace Spaces
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise

if __name__ == "__main__":
    main()