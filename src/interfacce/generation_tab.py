"""
Text Generation Interface Tab
"""

import gradio as gr
from typing import Dict

def generate_text_interface(predictor, seed_text: str, max_length: int = 15, temperature: float = 0.8) -> str:
    """Interface function for text generation"""
    if not seed_text.strip():
        return "⚠️ Please enter some Arabic seed text"
    
    generated = predictor.generate_text(seed_text, max_length, temperature)
    
    result = f"""
## 🎯 Generated Text

**Seed Text:** `{seed_text}`

**Generated Text:** 
> {generated}

**Statistics:**
- **Total Length:** {len(generated.split())} words
- **Generated Length:** {len(generated.split()) - len(seed_text.split())} new words
- **Temperature:** {temperature} {'(Conservative)' if temperature < 0.7 else '(Balanced)' if temperature < 1.2 else '(Creative)'}
    """
    
    return result

def create_generation_tab(predictor, config: Dict):
    """Create the text generation tab"""
    with gr.Tab("✍️ Text Generation"):
        gr.Markdown("### Generate new Arabic text from a seed phrase")
        
        with gr.Row():
            with gr.Column(scale=2):
                generation_input = gr.Textbox(
                    label="Seed Text (Arabic)",
                    placeholder="اللغة العربية",
                    lines=3,
                    elem_classes="arabic-text"
                )
                
                max_length_slider = gr.Slider(
                    minimum=5, 
                    maximum=config["model"]["max_length"], 
                    value=15, 
                    step=1,
                    label="Max Generated Length (words)"
                )
                
                temperature_slider = gr.Slider(
                    minimum=config["model"]["temperature_range"][0], 
                    maximum=config["model"]["temperature_range"][1], 
                    value=0.8, 
                    step=0.1,
                    label="Temperature (Creativity Control)"
                )
                
                generate_btn = gr.Button("✍️ Generate Text", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                generation_output = gr.Markdown(
                    value="Enter seed text and click 'Generate' to create new Arabic text...",
                    elem_classes="arabic-text"
                )
        
        # Temperature guide
        gr.Markdown("""
        **🌡️ Temperature Guide:**
        - **0.1-0.5**: Conservative, predictable text
        - **0.6-1.0**: Balanced creativity and coherence
        - **1.1-2.0**: Creative, diverse output
        """)
        
        # Example inputs
        gr.Examples(
            examples=config["interface"]["examples"]["generation"],
            inputs=generation_input,
            label="💡 Try these seed texts:"
        )
        
        generate_btn.click(
            lambda text, length, temp: generate_text_interface(predictor, text, length, temp),
            inputs=[generation_input, max_length_slider, temperature_slider],
            outputs=generation_output
        )