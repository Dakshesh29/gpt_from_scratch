"""
Model module for GPT from scratch project.

This module contains the implementation of a transformer-based language model,
including token/positional embeddings, self-attention, transformer blocks,
and the full GPT architecture.
"""

__version__ = "0.1.0"
__author__ = "GPT from Scratch"

from .gpt import GPT
from .attention import Head, MultiHeadAttention
from .transformer_block import Block, FeedForward