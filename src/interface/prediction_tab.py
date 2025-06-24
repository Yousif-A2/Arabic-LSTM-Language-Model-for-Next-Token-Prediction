"""
Next Token Prediction Interface Tab
"""

import gradio as gr
from typing import Dict

def predict_next_tokens_interface(predictor, text: str, top_k: int = 5) -> str:
    """Interface function for next token prediction"""
    if not text.strip():
        return "⚠️ Please enter some Arabic text"
    
    predictions = predictor.predict_next_token(text, top_k)
    
    result = f"🔮 **Next Token Predictions for:** `{text}`\n\n"
    
    for i, pred in enumerate(predictions[:top_k], 1):
        probability = pred['probability'] * 100
        bar_length = int(probability / 5)  # Simple ASCII progress bar
        bar = "█" * bar_length + "░" * (20 - bar_length)
        result += f"**{i}.** `{pred['token']}` - {probability:.1f}% {bar}\n"
    
    return result

def create_prediction_tab(predictor, config: Dict):
    """Create the next token prediction tab"""
    with gr.Tab("🔮 Next Token Prediction"):
        gr.Markdown("### Predict the most likely next Arabic words")
        
        with gr.Row():
            with gr.Column(scale=2):
                prediction_input = gr.Textbox(
                    label="Arabic Text Input",
                    placeholder="اللغة العربية",
                    lines=3,
                    elem_classes="arabic-text"
                )
                
                top_k_slider = gr.Slider(
                    minimum=1, 
                    maximum=config["model"]["top_k_range"][1], 
                    value=5, 
                    step=1,
                    label="Number of Predictions"
                )
                
                predict_btn = gr.Button("🔮 Predict Next Tokens", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                prediction_output = gr.Markdown(
                    value="Enter Arabic text and click 'Predict' to see results...",
                    elem_classes="arabic-text"
                )
        
        # Example inputs
        gr.Examples(
            examples=config["interface"]["examples"]["prediction"],
            inputs=prediction_input,
            label="💡 Try these examples:"
        )
        
        predict_btn.click(
            lambda text, k: predict_next_tokens_interface(predictor, text, k),
            inputs=[prediction_input, top_k_slider],
            outputs=prediction_output
        )