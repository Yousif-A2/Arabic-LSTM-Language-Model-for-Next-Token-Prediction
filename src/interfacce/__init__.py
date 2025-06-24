"""
Gradio interface components
"""

from .prediction_tab import create_prediction_tab
from .generation_tab import create_generation_tab
from .info_tab import create_info_tab

__all__ = [
    "create_prediction_tab",
    "create_generation_tab",
    "create_info_tab"
]