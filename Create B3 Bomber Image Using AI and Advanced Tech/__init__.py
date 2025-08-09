"""
B2 Spirit AI: Advanced Multi-Modal Defense Intelligence System

This package provides a comprehensive suite of AI models and tools for defense
intelligence applications, integrating cutting-edge technologies from Meta AI,
Anduril Industries, and Palantir Technologies.

Main Components:
- Meta AI: Language models, computer vision, and multi-modal understanding
- Anduril: Defense systems integration and autonomous decision-making
- Palantir: Data fusion, analytics, and strategic planning
- Fusion: Multi-system integration and ensemble methods
- Visualization: 5D holographic rendering and data visualization

Author: B2 Spirit AI Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "B2 Spirit AI Research Team"
__email__ = "research@b2spirit-ai.org"
__license__ = "MIT"

# Core imports
from .fusion import B2SpiritAI
from .meta_ai import MetaAISystem
from .anduril import AndurilDefenseSystem
from .palantir import PalantirAnalytics

# Utility imports
from .utils import (
    load_config,
    setup_logging,
    validate_inputs,
    performance_monitor
)

__all__ = [
    "B2SpiritAI",
    "MetaAISystem", 
    "AndurilDefenseSystem",
    "PalantirAnalytics",
    "load_config",
    "setup_logging",
    "validate_inputs",
    "performance_monitor"
]

