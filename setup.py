"""
Setup script for LlamaCanvas package.
"""

from setuptools import setup, find_packages
import os
import re

# Read the version from __init__.py
with open(os.path.join("src", "llama_canvas", "__init__.py")) as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = "0.0.1"

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llama-canvas-llamasearch",
    version=version,
    author="LlamaSearch AI",
    author_email="nikjois@llamasearch.ai",
    description="Advanced AI-driven multi-modal generation platform with Claude API integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://llamasearch.ai",
    project_urls={
        "Bug Tracker": "https://github.com/llamacanvas/llamacanvas/issues",
        "Documentation": "https://llamacanvas.ai/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=9.0.0",
        "pydantic>=1.8.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "jinja2>=3.0.0",
            "python-multipart>=0.0.5",
        ],
        "claude": [
            "anthropic>=0.3.0",
        ],
        "diffusion": [
            "diffusers>=0.14.0",
            "torch>=1.12.0",
            "transformers>=4.25.0",
        ],
        "video": [
            "moviepy>=1.0.3",
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.9.0",
        ],
        "full": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "jinja2>=3.0.0",
            "python-multipart>=0.0.5",
            "anthropic>=0.3.0",
            "diffusers>=0.14.0",
            "torch>=1.12.0",
            "transformers>=4.25.0",
            "moviepy>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "llama-canvas=llama_canvas.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 
# Updated in commit 5 - 2025-04-04 17:00:29

# Updated in commit 13 - 2025-04-04 17:00:30

# Updated in commit 21 - 2025-04-04 17:00:32

# Updated in commit 29 - 2025-04-04 17:00:33

# Updated in commit 5 - 2025-04-05 14:24:31

# Updated in commit 13 - 2025-04-05 14:24:31

# Updated in commit 21 - 2025-04-05 14:24:31

# Updated in commit 29 - 2025-04-05 14:24:31

# Updated in commit 5 - 2025-04-05 15:00:24

# Updated in commit 13 - 2025-04-05 15:00:24

# Updated in commit 21 - 2025-04-05 15:00:25

# Updated in commit 29 - 2025-04-05 15:00:25

# Updated in commit 5 - 2025-04-05 15:10:01

# Updated in commit 13 - 2025-04-05 15:10:01

# Updated in commit 21 - 2025-04-05 15:10:01

# Updated in commit 29 - 2025-04-05 15:10:02

# Updated in commit 5 - 2025-04-05 15:37:31

# Updated in commit 13 - 2025-04-05 15:37:31

# Updated in commit 21 - 2025-04-05 15:37:31

# Updated in commit 29 - 2025-04-05 15:37:31

# Updated in commit 5 - 2025-04-05 16:42:43

# Updated in commit 13 - 2025-04-05 16:42:43
