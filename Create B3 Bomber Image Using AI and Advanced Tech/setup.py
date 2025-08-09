#!/usr/bin/env python3
"""
B2 Spirit AI: Advanced Multi-Modal Defense Intelligence System
Setup configuration for package installation and distribution.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("B2 Spirit AI requires Python 3.8 or higher")

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version information
VERSION = "1.0.0"
DESCRIPTION = "Advanced Multi-Modal Defense Intelligence System"
LONG_DESCRIPTION = read_readme()

# Package configuration
setup(
    name="b2-spirit-ai",
    version=VERSION,
    author="B2 Spirit AI Research Team",
    author_email="research@b2spirit-ai.org",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/b2-spirit-ai-model",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/b2-spirit-ai-model/issues",
        "Documentation": "https://b2spirit-ai.org/docs",
        "Research Papers": "https://b2spirit-ai.org/papers",
        "Demo": "https://b2spirit-ai.org/demo"
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "tensorrt>=8.6.0",
        ],
        "quantum": [
            "qiskit>=0.43.0",
            "cirq>=1.1.0",
            "pennylane>=0.31.0",
        ],
        "all": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
            "torch[cuda]>=2.0.0",
            "tensorrt>=8.6.0",
            "qiskit>=0.43.0",
            "cirq>=1.1.0",
            "pennylane>=0.31.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "b2spirit-train=src.training.cli:main",
            "b2spirit-infer=src.inference.cli:main",
            "b2spirit-deploy=src.deployment.cli:main",
            "b2spirit-viz=src.visualization.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
        "configs": ["*.yaml", "*.json"],
        "assets": ["*.png", "*.jpg", "*.mp4"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "machine learning",
        "defense technology",
        "multi-modal AI",
        "computer vision",
        "natural language processing",
        "stealth technology",
        "autonomous systems",
        "data fusion",
        "strategic planning"
    ],
    license="MIT",
    platforms=["any"],
)

