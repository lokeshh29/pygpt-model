"""
Legacy import compatibility wrapper.
Refactored implementation is located in app/tokenizer/tokenizer.py.
"""

from app.tokenizer.tokenizer import PyGPTTokenizer, pygpt_tokenizer

__all__ = ["PyGPTTokenizer", "pygpt_tokenizer"]
