#!/bin/bash
set -e

# ================================================================
#            🦙 ANTHROPIC LLAMA_CANVAS SUPER INSTALLER 🦙
#       Advanced AI-driven Multi-modal Generation Platform
#                 with Claude API Integration
# ================================================================

# ANSI color codes for llama-themed terminal output
LLAMA_PINK='\033[38;5;219m'
LLAMA_BLUE='\033[38;5;39m'
LLAMA_GREEN='\033[38;5;83m'
LLAMA_YELLOW='\033[38;5;227m'
LLAMA_PURPLE='\033[38;5;141m'
LLAMA_ORANGE='\033[38;5;214m'
RESET='\033[0m'
BOLD='\033[1m'
# Standard colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'

# Configuration variables
PKG_NAME="llama_canvas"
PKG_VERSION="1.0.0"
GITHUB_USERNAME="yourusername"
EMAIL="your.email@example.com"
LICENSE="MIT"
PYTHON_MIN_VERSION="3.8"
ANTHROPIC_API_URL="https://api.anthropic.com"
DEFAULT_MODEL="claude-3-opus-20240229"

# Project structure
PROJECT_ROOT="./$PKG_NAME"
SRC_DIR="$PROJECT_ROOT/src"
TESTS_DIR="$PROJECT_ROOT/tests"
DOCS_DIR="$PROJECT_ROOT/docs"
RESOURCES_DIR="$PROJECT_ROOT/resources"
EXAMPLES_DIR="$PROJECT_ROOT/examples"

# Display a colorful llama banner
display_llama_banner() {
    clear
    echo -e "${LLAMA_PINK}"
    echo -e "                        ${BOLD}ANTHROPIC LLAMA_CANVAS${RESET}${LLAMA_PINK}"
    echo -e "                        ~~~~~~~~~~~~~~~~~~~"
    echo "                 /\     /\ "
    echo "                ( /\---/\ )"
    echo "                 \ |   | /"
    echo "                  \|___|/"
    echo "                   |   |"
    echo "                   |   |"
    echo -e "  ${LLAMA_YELLOW}Ultimate AI-driven Multi-modal Generation Platform${RESET}"
    echo -e "  ${LLAMA_GREEN}✨ Claude API Integration ✨ Responsible AI ✨ Agent Architecture ✨${RESET}"
    echo -e "  ${LLAMA_GREEN}✨ Style Transfer ✨ Image Blending ✨ Video Generation ✨${RESET}"
    echo ""
}

# Display progress message with spinner animation
progress() {
    local message=$1
    echo -ne "${LLAMA_BLUE}==>${RESET} ${message}... "
    
    # Optional spinner
    if [ "${2:-}" = "spinner" ]; then
        local spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        local i=0
        local pid=$!
        while kill -0 $pid 2>/dev/null; do
            local char="${spinner:i++%${#spinner}:1}"
            echo -ne "\b${char}"
            sleep 0.1
        done
        echo -ne "\b"
    fi
}

# Display success message
success() {
    local message=$1
    echo -e "${LLAMA_GREEN}✓${RESET} ${message}"
}

# Display warning message
warning() {
    local message=$1
    echo -e "${LLAMA_YELLOW}⚠️${RESET} ${message}"
}

# Display error message and exit
error() {
    local message=$1
    echo -e "${LLAMA_ORANGE}❌ ERROR:${RESET} ${message}" >&2
    exit 1
}

# Display info message
info() {
    local message=$1
    echo -e "${LLAMA_PURPLE}ℹ️${RESET} ${message}"
}

# Animate dots for loading effect
animate_loading() {
    local message=$1
    local duration=$2
    echo -ne "${LLAMA_PURPLE}$message${RESET}"
    for ((i=0; i<duration; i++)); do
        echo -ne "${LLAMA_PURPLE}.${RESET}"
        sleep 0.3
    done
    echo
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if file exists
file_exists() {
    [ -f "$1" ]
}

# Check if directory exists
dir_exists() {
    [ -d "$1" ]
}

# Create directory if it doesn't exist
ensure_dir() {
    [ -d "$1" ] || mkdir -p "$1"
}

# Get absolute path
get_abs_path() {
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

# Print a horizontal separator line
print_separator() {
    echo -e "${LLAMA_BLUE}================================================================${RESET}"
}

# Check if system meets requirements
check_system_requirements() {
    progress "Checking system requirements"
    
    # Check Python version
    if ! command_exists python3; then
        error "Python 3 is required but not installed. Please install Python $PYTHON_MIN_VERSION or higher."
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if ! command_exists bc; then
        warning "bc is not installed, using basic version comparison"
        PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_VERSION_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        PYTHON_MIN_VERSION_MAJOR=$(echo $PYTHON_MIN_VERSION | cut -d'.' -f1)
        PYTHON_MIN_VERSION_MINOR=$(echo $PYTHON_MIN_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_VERSION_MAJOR" -lt "$PYTHON_MIN_VERSION_MAJOR" ] || 
           ([ "$PYTHON_VERSION_MAJOR" -eq "$PYTHON_MIN_VERSION_MAJOR" ] && 
            [ "$PYTHON_VERSION_MINOR" -lt "$PYTHON_MIN_VERSION_MINOR" ]); then
            error "Python $PYTHON_MIN_VERSION or higher is required (found $PYTHON_VERSION)"
        fi
    else
        if (( $(echo "$PYTHON_VERSION < $PYTHON_MIN_VERSION" | bc -l) )); then
            error "Python $PYTHON_MIN_VERSION or higher is required (found $PYTHON_VERSION)"
        fi
    fi
    
    # Check for required commands
    for cmd in git curl pip; do
        if ! command_exists $cmd; then
            error "$cmd is required but not installed"
        fi
    done
    
    # Check for pip and venv
    if ! python3 -c 'import venv' &> /dev/null; then
        error "python3-venv is required but not installed"
    fi
    
    # Check for GPU support (optional but recommended)
    HAS_GPU=0
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        success "NVIDIA GPU detected: $GPU_INFO - GPU acceleration will be enabled"
        HAS_GPU=1
    elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
        success "Apple Silicon detected - MPS acceleration will be enabled"
        HAS_GPU=2  # Apple Silicon MPS
    else
        warning "No compatible GPU detected - will run in CPU mode (slower generation)"
    fi
    
    # Check for network connectivity
    if ! curl -s --head https://pypi.org >/dev/null; then
        warning "Network connectivity issues detected - may affect package downloads"
    fi
    
    # Check for Anthropic API key
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        warning "ANTHROPIC_API_KEY environment variable not set. You'll need to provide it later."
        NEED_API_KEY=1
    else
        success "Anthropic API key found"
        NEED_API_KEY=0
    fi
    
    success "System requirements satisfied"
}

# Create project structure
create_project_structure() {
    progress "Creating project structure"
    
    # Create base directory if it doesn't exist
    if dir_exists "$PROJECT_ROOT"; then
        warning "Directory $PROJECT_ROOT already exists. Files may be overwritten."
    else
        mkdir -p "$PROJECT_ROOT"
    fi
    
    # Create project subdirectories
    ensure_dir "$SRC_DIR"
    ensure_dir "$TESTS_DIR"
    ensure_dir "$DOCS_DIR"
    ensure_dir "$RESOURCES_DIR"
    ensure_dir "$EXAMPLES_DIR"
    ensure_dir "$PROJECT_ROOT/.github/workflows"
    
    # Create Python package subdirectories
    ensure_dir "$SRC_DIR/$PKG_NAME"
    ensure_dir "$SRC_DIR/$PKG_NAME/core"
    ensure_dir "$SRC_DIR/$PKG_NAME/agents"
    ensure_dir "$SRC_DIR/$PKG_NAME/generators"
    ensure_dir "$SRC_DIR/$PKG_NAME/processors"
    ensure_dir "$SRC_DIR/$PKG_NAME/ui"
    ensure_dir "$SRC_DIR/$PKG_NAME/utils"
    ensure_dir "$SRC_DIR/$PKG_NAME/models"
    ensure_dir "$SRC_DIR/$PKG_NAME/claude"
    ensure_dir "$SRC_DIR/$PKG_NAME/api"
    ensure_dir "$SRC_DIR/$PKG_NAME/resources"
    
    # Create empty __init__.py files to make directories importable
    touch "$SRC_DIR/$PKG_NAME/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/core/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/agents/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/generators/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/processors/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/ui/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/utils/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/models/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/claude/__init__.py"
    touch "$SRC_DIR/$PKG_NAME/api/__init__.py"
    
    # Create test directories
    ensure_dir "$TESTS_DIR/unit"
    ensure_dir "$TESTS_DIR/integration"
    ensure_dir "$TESTS_DIR/e2e"
    
    # Create example directories
    ensure_dir "$EXAMPLES_DIR/basic"
    ensure_dir "$EXAMPLES_DIR/advanced"
    ensure_dir "$EXAMPLES_DIR/claude_integration"
    
    # Create docs subdirectories
    ensure_dir "$DOCS_DIR/api"
    ensure_dir "$DOCS_DIR/guides"
    ensure_dir "$DOCS_DIR/examples"
    
    success "Project structure created"
}

# Set up virtual environment and install dependencies
setup_virtual_environment() {
    progress "Setting up virtual environment"
    
    cd "$PROJECT_ROOT"
    
    # Create and activate virtual environment
    python3 -m venv venv
    
    # Determine OS for activation
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        ACTIVATE_CMD="venv/Scripts/activate"
    else
        ACTIVATE_CMD="venv/bin/activate"
    fi
    
    # Use source or . based on shell
    if [[ -n "$BASH_VERSION" || -n "$ZSH_VERSION" ]]; then
        # For Bash or Zsh
        source "$ACTIVATE_CMD"
    else
        # For other shells
        . "$ACTIVATE_CMD"
    fi
    
    # Upgrade pip and setuptools
    progress "Upgrading pip and setuptools"
    pip install --upgrade pip setuptools wheel
    
    # Install development tools
    progress "Installing development tools"
    pip install black flake8 mypy pytest pytest-cov tox pre-commit
    
    # Install core dependencies based on GPU availability
    progress "Installing core dependencies"
    pip install \
        numpy \
        pandas \
        matplotlib \
        pillow \
        scikit-image \
        scikit-learn \
        opencv-python \
        requests \
        tqdm \
        rich \
        typer \
        pydantic \
        fastapi \
        uvicorn \
        jinja2 \
        python-dotenv \
        anthropic \
        huggingface_hub
    
    # Install GPU-specific packages
    if [ "$HAS_GPU" -eq 1 ]; then
        # NVIDIA GPU
        progress "Installing NVIDIA GPU packages"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    elif [ "$HAS_GPU" -eq 2 ]; then
        # Apple Silicon
        progress "Installing Apple Silicon optimized packages"
        pip install torch torchvision
    else
        # CPU only
        progress "Installing CPU-only packages"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install diffusers and transformers
    pip install \
        diffusers[torch] \
        transformers \
        accelerate \
        safetensors \
        moviepy \
        einops
    
    success "Virtual environment set up and dependencies installed"
}

# Set up pre-commit hooks
setup_precommit_hooks() {
    progress "Setting up pre-commit hooks"
    
    cd "$PROJECT_ROOT"
    
    cat > .pre-commit-config.yaml << 'EOF'
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-ast
    -   id: check-json
    -   id: check-merge-conflict
    -   id: detect-private-key

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
    -   id: flake8
        additional_dependencies: [flake8-docstrings]

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        exclude: ^tests/
        additional_dependencies: [types-requests]
EOF
    
    # Initialize pre-commit
    pre-commit install
    
    success "Pre-commit hooks set up"
}

# Create package configuration files
create_package_config() {
    progress "Creating package configuration files"
    
    cd "$PROJECT_ROOT"
    
    # Create setup.py
    cat > setup.py << EOF
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="$PKG_NAME",
    version="$PKG_VERSION",
    author="$GITHUB_USERNAME",
    author_email="$EMAIL",
    description="Advanced AI-driven multi-modal generation platform with Claude API integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/$GITHUB_USERNAME/$PKG_NAME",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=$PYTHON_MIN_VERSION",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "requests>=2.25.0",
        "tqdm>=4.61.1",
        "matplotlib>=3.4.2",
        "scipy>=1.7.0",
        "opencv-python>=4.5.2",
        "diffusers>=0.15.0",
        "transformers>=4.13.0",
        "huggingface-hub>=0.4.0",
        "moviepy>=1.0.3",
        "rich>=11.2.0",
        "typer>=0.4.0",
        "pydantic>=1.9.0",
        "safetensors>=0.3.1",
        "accelerate>=0.12.0",
        "anthropic>=0.5.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "python-dotenv>=1.0.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.6b0",
            "isort>=5.9.2",
            "mypy>=0.910",
            "flake8>=4.0.0",
            "pre-commit>=2.15.0",
            "tox>=3.24.0",
            "twine>=3.4.1",
            "build>=0.7.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llama-canvas=llama_canvas.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llama_canvas": ["resources/*"],
    },
)
EOF

    # Create pyproject.toml
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.0"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 88
target-version = ["py38"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "--cov=llama_canvas"
testpaths = ["tests"]
EOF

    # Create README.md
    cat > README.md << EOF
# 🦙 LlamaCanvas - Advanced AI-driven Multi-modal Generation Platform

![Python Version](https://img.shields.io/badge/python-$PYTHON_MIN_VERSION%2B-blue.svg)
![License](https://img.shields.io/badge/license-$LICENSE-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange.svg)

Advanced AI-driven multi-modal generation platform featuring Claude API integration, agent architecture, responsible AI tools, and sophisticated image/video generation capabilities.

## ✨ Features

- **Claude API Integration**: Leverage the power of Anthropic's Claude models for advanced text and multi-modal generation
- **Agent Architecture**: Modular, extensible system of specialized AI agents that work together to complete complex tasks
- **Responsible AI**: Built-in tools for content safety, bias mitigation, and ethical generation
- **Style Transfer**: Apply artistic styles to your images with state-of-the-art neural style transfer
- **Image Generation**: Create images from text descriptions using advanced AI models
- **Image Blending**: Seamlessly blend multiple images with various modes and effects
- **Inpainting**: Fill in missing parts of images with AI-generated content
- **Super-Resolution**: Enhance image quality and resolution
- **Animation**: Create smooth animations and transitions between images
- **Video Generation**: Generate and process videos with AI assistance
- **Interactive CLI**: User-friendly command-line interface with llama-themed design
- **Pre-trained Models**: Integration with popular generative models:
  - Claude (via API)
  - Stable Diffusion (v1.5, v2, XL)
  - DALL-E 2
  - VQGAN+CLIP

## 🚀 Installation

```bash
pip install llama-canvas
```

For development installation:

```bash
git clone https://github.com/$GITHUB_USERNAME/$PKG_NAME.git
cd $PKG_NAME
pip install -e ".[dev]"
```

## 🏁 Quick Start

### Python API

```python
from llama_canvas import Canvas, generators, claude

# Create a canvas
canvas = Canvas(width=512, height=512)

# Generate an image using Claude's multimodal capabilities
image = claude.generate_image(
    prompt="a colorful llama in a meadow at sunset, digital art style",
    model="claude-3-opus-20240229",
    width=1024,
    height=1024
)

# Apply style transfer
stylized = canvas.apply_style(
    image, 
    style="van_gogh", 
    strength=0.8
)

# Enhance with AI agents
enhanced = canvas.agents.enhance(
    stylized,
    prompt="Make it more vibrant and dreamy"
)

# Save the result
enhanced.save("colorful_llama.png")
```

### Command Line Interface

```bash
# Set your Anthropic API key (or place in .env file)
export ANTHROPIC_API_KEY=your_api_key_here

# Generate an image with Claude
llama-canvas claude-generate "a colorful llama in a meadow at sunset"

# Apply style transfer
llama-canvas style input.jpg --style van_gogh --output stylized.jpg

# Create an animation
llama-canvas animate "a llama walking through a field" --frames 30 --output animation.mp4

# Run the interactive UI
llama-canvas ui
```

## 📚 Documentation

For complete documentation, visit our [GitHub Pages](https://github.com/$GITHUB_USERNAME/$PKG_NAME).

## 🧪 Testing

Run tests with pytest:

```bash
pytest
```

## 📝 License

$LICENSE License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
EOF

    # Create LICENSE file
    cat > LICENSE << EOF
MIT License

Copyright (c) $(date +%Y) $GITHUB_USERNAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

    # Create .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Virtual environments
venv/
env/
ENV/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db

# Project specific
*.png
*.jpg
*.jpeg
*.mp4
*.gif
*.checkpoint

# Environment variables
.env
EOF

    # Create GitHub Actions workflow
    mkdir -p .github/workflows
    cat > .github/workflows/python-package.yml << 'EOF'
name: Python Package

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.8", "3.9", "3.10"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python -m pip install tox tox-gh-actions
    - name: Test with tox
      run: tox
EOF

    # Create tox.ini
    cat > tox.ini << 'EOF'
[tox]
envlist = py38, py39, py310, lint, type, coverage
isolated_build = True

[gh-actions]
python =
    3.8: py38
    3.9: py39
    3.10: py310, lint, type, coverage

[testenv]
deps = pytest
commands =
    pytest {posargs:tests}

[testenv:lint]
deps =
    black
    isort
    flake8
commands =
    black --check src tests
    isort --check src tests
    flake8 src tests

[testenv:type]
deps = mypy
commands =
    mypy src

[testenv:coverage]
deps =
    pytest
    pytest-cov
commands =
    pytest --cov=llama_canvas --cov-report=xml
EOF

    # Create environment template
    cat > .env.template << 'EOF'
# Anthropic API Key
ANTHROPIC_API_KEY=your_api_key_here

# Optional settings
# MODEL_CACHE_DIR=~/.cache/llama_canvas/models
# LOG_LEVEL=INFO
EOF

    # Create CONTRIBUTING.md
    cat > CONTRIBUTING.md << 'EOF'
# Contributing to LlamaCanvas

Thank you for considering contributing to LlamaCanvas! We welcome contributions from the community to help improve and expand the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Set up the development environment:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards

3. Run the tests to ensure everything works:
   ```bash
   pytest
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request on GitHub

## Coding Standards

- Follow PEP 8 and PEP 257 conventions
- Use type hints for all function parameters and return values
- Write docstrings for all public functions, classes, and methods
- Keep lines under 88 characters (Black's default)
- Use meaningful variable and function names

## Testing

- Add tests for all new features and bug fixes
- Ensure all tests pass before submitting a PR
- Aim for high test coverage

## Documentation

- Update documentation for any changed functionality
- Add documentation for new features

## Responsible AI

LlamaCanvas follows ethical AI principles. When contributing:

- Be mindful of potential bias in models and data
- Consider privacy implications of features
- Implement appropriate safety mechanisms for content generation

Thank you for your contributions!
EOF

    # Create CODE_OF_CONDUCT.md
    cat > CODE_OF_CONDUCT.md << 'EOF'
# Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or
  advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email
  address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement.
All complaints will be reviewed and investigated promptly and fairly.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.
EOF

    success "Package configuration files created"
}

# Create core module files
create_core_module() {
    progress "Creating core module files"
    
    # Main package __init__.py
    cat > "$SRC_DIR/$PKG_NAME/__init__.py" << 'EOF'
"""
LlamaCanvas - Advanced AI-driven multi-modal generation platform with Claude API integration.

This package provides tools for image and video generation, manipulation,
and enhancement using cutting-edge AI models and techniques.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("llama_canvas")
except PackageNotFoundError:
    __version__ = "unknown"

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.core.video import Video
from llama_canvas.core.agent_manager import AgentManager

__all__ = ["Canvas", "Image", "Video", "AgentManager"]
EOF

    # Core module __init__.py
    cat > "$SRC_DIR/$PKG_NAME/core/__init__.py" << 'EOF'
"""
Core classes for LlamaCanvas.
"""

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.core.video import Video
from llama_canvas.core.agent_manager import AgentManager

__all__ = ["Canvas", "Image", "Video", "AgentManager"]
EOF

    # Canvas class
    cat > "$SRC_DIR/$PKG_NAME/core/canvas.py" << 'EOF'
"""
Core Canvas class for LlamaCanvas.

The Canvas class represents the main workspace for image and video generation,
manipulation, and processing. It integrates with various generators, processors,
and the agent system to provide a unified interface for creative tasks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from PIL import Image as PILImage

from llama_canvas.core.image import Image
from llama_canvas.core.agent_manager import AgentManager
from llama_canvas.utils.logging import get_logger
from llama_canvas.utils.config import settings

logger = get_logger(__name__)


class Canvas:
    """
    Canvas represents the main workspace for multi-modal content generation and manipulation.
    
    It serves as the central coordinating class for the LlamaCanvas system,
    providing access to image generation, manipulation, agent dispatching,
    and integration with Claude API.
    """
    
    def __init__(
        self, 
        width: int = 512, 
        height: int = 512, 
        background_color: Tuple[int, int, int] = (255, 255, 255),
        use_agents: bool = True
    ):
        """
        Initialize a new Canvas.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            background_color: RGB color tuple for canvas background
            use_agents: Whether to initialize and use the agent system
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.layers: List[Image] = []
        
        # Initialize with blank canvas
        blank = np.ones((height, width, 3), dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
        self.base_image = Image(PILImage.fromarray(blank))
        
        # Initialize agent manager if enabled
        self.agents = AgentManager(self) if use_agents else None
        
        logger.info(f"Canvas initialized with dimensions {width}x{height}")
    
    def generate_from_text(
        self, 
        prompt: str,
        model: str = "stable-diffusion-v2",
        **kwargs
    ) -> Image:
        """
        Generate an image from text prompt and add it to the canvas.
        
        Args:
            prompt: Text description of the image to generate
            model: Model name to use for generation
            **kwargs: Additional parameters for the generator
            
        Returns:
            Generated image
        """
        # Import here to avoid circular imports
        from llama_canvas.generators import text_to_image
        
        logger.info(f"Generating image from text prompt using {model}")
        
        # If agents are enabled and model is not explicitly Claude, let the agent system handle it
        if self.agents and not model.startswith("claude-"):
            return self.agents.generate_image(prompt, model=model, **kwargs)
        
        # Otherwise use the direct generator
        image = text_to_image(
            prompt, 
            width=self.width, 
            height=self.height, 
            model=model, 
            **kwargs
        )
        
        self.layers.append(image)
        return image
    
    def apply_style(
        self,
        image: Image,
        style: Union[str, Image],
        strength: float = 0.8,
        use_claude: bool = False,
        **kwargs
    ) -> Image:
        """
        Apply a style to an image.
        
        Args:
            image: Source image to stylize
            style: Style name or reference image
            strength: Style transfer strength (0.0 to 1.0)
            use_claude: Whether to use Claude for style transfer
            **kwargs: Additional style transfer parameters
            
        Returns:
            Stylized image
        """
        # Import here to avoid circular imports
        from llama_canvas.processors import style_transfer
        
        logger.info(f"Applying style {'with Claude' if use_claude else ''}")
        
        # If agents are enabled and use_claude is True, let the agent system handle it
        if self.agents and use_claude:
            return self.agents.apply_style(image, style, strength=strength, **kwargs)
        
        # Otherwise use the direct processor
        styled_image = style_transfer(image, style, strength=strength, **kwargs)
        return styled_image
    
    def enhance_resolution(
        self,
        image: Image,
        scale: int = 2,
        **kwargs
    ) -> Image:
        """
        Enhance image resolution.
        
        Args:
            image: Image to enhance
            scale: Upscaling factor
            **kwargs: Additional parameters for super resolution
            
        Returns:
            Enhanced image
        """
        # Import here to avoid circular imports
        from llama_canvas.processors import super_resolution
        
        logger.info(f"Enhancing image resolution by {scale}x")
        
        # If agents are enabled, let the agent system handle it for better results
        if self.agents and kwargs.get("use_agents", True):
            return self.agents.enhance_resolution(image, scale=scale, **kwargs)
        
        # Otherwise use the direct processor
        enhanced = super_resolution(image, scale=scale, **kwargs)
        return enhanced
    
    def inpaint(
        self,
        image: Image,
        mask: Union[Image, np.ndarray],
        prompt: Optional[str] = None,
        use_claude: bool = False,
        **kwargs
    ) -> Image:
        """
        Perform inpainting on an image.
        
        Args:
            image: Image to inpaint
            mask: Mask of the area to inpaint (white=inpaint)
            prompt: Optional text prompt to guide inpainting
            use_claude: Whether to use Claude for inpainting
            **kwargs: Additional inpainting parameters
            
        Returns:
            Inpainted image
        """
        # Import here to avoid circular imports
        from llama_canvas.processors import inpainting
        
        logger.info(f"Performing inpainting {'with Claude' if use_claude else ''}")
        
        # If agents are enabled and use_claude is True, let the agent system handle it
        if self.agents and use_claude:
            return self.agents.inpaint(image, mask, prompt=prompt, **kwargs)
        
        # Otherwise use the direct processor
        result = inpainting(image, mask, prompt=prompt, **kwargs)
        return result
    
    def blend_images(
        self,
        image1: Image,
        image2: Image,
        alpha: float = 0.5,
        mode: str = "normal"
    ) -> Image:
        """
        Blend two images together.
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            mode: Blending mode (normal, multiply, screen, overlay, etc.)
            
        Returns:
            Blended image
        """
        logger.info(f"Blending images using {mode} mode with alpha={alpha}")
        
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
        
        if mode == "normal":
            blended = Image.blend(image1, image2, alpha)
        elif mode == "multiply":
            blended = Image.multiply(image1, image2)
        elif mode == "screen":
            blended = Image.screen(image1, image2)
        elif mode == "overlay":
            blended = Image.overlay(image1, image2)
        elif mode == "add":
            blended = Image.add(image1, image2, alpha)
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")
        
        return blended
    
    def create_animation(
        self,
        frames: List[Image],
        fps: int = 24,
        **kwargs
    ) -> 'Video':
        """
        Create an animation from a list of images.
        
        Args:
            frames: List of frames
            fps: Frames per second
            **kwargs: Additional animation parameters
            
        Returns:
            Video object containing the animation
        """
        # Import here to avoid circular imports
        from llama_canvas.core.video import Video
        
        logger.info(f"Creating animation with {len(frames)} frames at {fps} FPS")
        
        video = Video(frames, fps=fps)
        return video
    
    def save(self, path: Union[str, Path], format: Optional[str] = None) -> None:
        """
        Save the current canvas state to a file.
        
        Args:
            path: Output file path
            format: Optional file format (deduced from extension if not provided)
        """
        logger.info(f"Saving canvas to {path}")
        
        if not self.layers:
            self.base_image.save(path, format=format)
            return
        
        # Composite all layers
        composite = self.layers[0].copy()
        for layer in self.layers[1:]:
            composite = self.blend_images(composite, layer)
        
        composite.save(path, format=format)
    
    def clear(self) -> None:
        """Clear all layers from the canvas."""
        logger.info("Clearing canvas")
        
        self.layers = []
        
        # Reset to blank canvas
        blank = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)
        self.base_image = Image(PILImage.fromarray(blank))
    
    def resize_canvas(self, width: int, height: int, preserve_content: bool = True) -> None:
        """
        Resize the canvas to new dimensions.
        
        Args:
            width: New canvas width
            height: New canvas height
            preserve_content: Whether to preserve and scale existing content
        """
        logger.info(f"Resizing canvas to {width}x{height}")
        
        old_width, old_height = self.width, self.height
        self.width, self.height = width, height
        
        if not preserve_content:
            # Just create a new blank canvas
            self.clear()
            return
        
        # Scale all layers to new size
        if self.layers:
            scaled_layers = []
            for layer in self.layers:
                scaled_layers.append(layer.resize((width, height)))
            self.layers = scaled_layers
        
        # Scale base image
        blank = np.ones((height, width, 3), dtype=np.uint8) * np.array(self.background_color, dtype=np.uint8)
        self.base_image = Image(PILImage.fromarray(blank))
EOF

    # Image class
    cat > "$SRC_DIR/$PKG_NAME/core/image.py" << 'EOF'
"""
Image class for LlamaCanvas.

Provides an enhanced Image class with various manipulation and processing capabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from PIL import Image as PILImage, ImageEnhance, ImageFilter, ImageOps


class Image:
    """
    Enhanced image class with additional processing capabilities.
    
    This class provides a wrapper around PIL Image with additional methods for
    common image manipulation tasks and integration with the LlamaCanvas ecosystem.
    """
    
    def __init__(self, image: Union[PILImage.Image, np.ndarray, str, Path, 'Image']):
        """
        Initialize an image from various sources.
        
        Args:
            image: Image source (PIL Image, numpy array, file path, or another Image)
        """
        if isinstance(image, Image):
            self._image = image._image.copy()
        elif isinstance(image, PILImage.Image):
            self._image = image
        elif isinstance(image, np.ndarray):
            self._image = PILImage.fromarray(
                image.astype(np.uint8) if image.dtype != np.uint8 else image
            )
        elif isinstance(image, (str, Path)):
            self._image = PILImage.open(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if needed (unless it's RGBA which we want to preserve)
        if self._image.mode not in ["RGB", "RGBA"]:
            self._image = self._image.convert("RGB")
    
    @property
    def width(self) -> int:
        """Get image width."""
        return self._image.width
    
    @property
    def height(self) -> int:
        """Get image height."""
        return self._image.height
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape as (height, width, channels)."""
        return (self.height, self.width, 3 if self._image.mode == "RGB" else 4)
    
    @property
    def array(self) -> np.ndarray:
        """Get image as numpy array."""
        return np.array(self._image)
    
    @property
    def pil_image(self) -> PILImage.Image:
        """Get underlying PIL image."""
        return self._image
    
    @property
    def mode(self) -> str:
        """Get image mode (RGB, RGBA, etc.)."""
        return self._image.mode
    
    def copy(self) -> 'Image':
        """Create a copy of the image."""
        return Image(self._image.copy())
    
    def resize(
        self, 
        size: Union[Tuple[int, int], int],
        resample: int = PILImage.LANCZOS,
        maintain_aspect_ratio: bool = False
    ) -> 'Image':
        """
        Resize the image.
        
        Args:
            size: New size as (width, height) or scaling factor if int
            resample: Resampling filter
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            
        Returns:
            Resized image
        """
        if isinstance(size, int):
            # Size is a scaling factor
            new_width = int(self.width * size)
            new_height = int(self.height * size)
            size = (new_width, new_height)
        elif maintain_aspect_ratio:
            # Calculate new dimensions while maintaining aspect ratio
            target_width, target_height = size
            aspect = self.width / self.height
            
            if target_width / target_height > aspect:
                # Width is the limiting factor
                new_width = int(target_height * aspect)
                new_height = target_height
                size = (new_width, new_height)
            else:
                # Height is the limiting factor
                new_width = target_width
                new_height = int(target_width / aspect)
                size = (new_width, new_height)
        
        return Image(self._image.resize(size, resample=resample))
    
    def crop(self, box: Tuple[int, int, int, int]) -> 'Image':
        """
        Crop the image.
        
        Args:
            box: Crop box as (left, upper, right, lower)
            
        Returns:
            Cropped image
        """
        return Image(self._image.crop(box))
    
    def adjust_brightness(self, factor: float) -> 'Image':
        """
        Adjust image brightness.
        
        Args:
            factor: Brightness adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Brightness(self._image)
        return Image(enhancer.enhance(factor))
    
    def adjust_contrast(self, factor: float) -> 'Image':
        """
        Adjust image contrast.
        
        Args:
            factor: Contrast adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Contrast(self._image)
        return Image(enhancer.enhance(factor))
    
    def adjust_saturation(self, factor: float) -> 'Image':
        """
        Adjust image saturation.
        
        Args:
            factor: Saturation adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Color(self._image)
        return Image(enhancer.enhance(factor))
    
    def adjust_sharpness(self, factor: float) -> 'Image':
        """
        Adjust image sharpness.
        
        Args:
            factor: Sharpness adjustment factor (1.0 = original)
            
        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Sharpness(self._image)
        return Image(enhancer.enhance(factor))
    
    def apply_filter(self, filter_type: str) -> 'Image':
        """
        Apply a filter to the image.
        
        Args:
            filter_type: Filter name (blur, sharpen, contour, etc.)
            
        Returns:
            Filtered image
        """
        filters = {
            "blur": ImageFilter.BLUR,
            "sharpen": ImageFilter.SHARPEN,
            "contour": ImageFilter.CONTOUR,
            "detail": ImageFilter.DETAIL,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
            "emboss": ImageFilter.EMBOSS,
            "find_edges": ImageFilter.FIND_EDGES,
            "smooth": ImageFilter.SMOOTH,
            "smooth_more": ImageFilter.SMOOTH_MORE,
        }
        
        if filter_type not in filters:
            raise ValueError(f"Unsupported filter: {filter_type}")
        
        return Image(self._image.filter(filters[filter_type]))
    
    def apply_gaussian_blur(self, radius: float) -> 'Image':
        """
        Apply Gaussian blur to the image.
        
        Args:
            radius: Blur radius
            
        Returns:
            Blurred image
        """
        return Image(self._image.filter(ImageFilter.GaussianBlur(radius)))
    
    def rotate(self, angle: float, expand: bool = False) -> 'Image':
        """
        Rotate the image.
        
        Args:
            angle: Rotation angle in degrees
            expand: Whether to expand the output to fit the rotated image
            
        Returns:
            Rotated image
        """
        return Image(self._image.rotate(angle, expand=expand))
    
    def flip_horizontal(self) -> 'Image':
        """
        Flip the image horizontally.
        
        Returns:
            Flipped image
        """
        return Image(ImageOps.mirror(self._image))
    
    def flip_vertical(self) -> 'Image':
        """
        Flip the image vertically.
        
        Returns:
            Flipped image
        """
        return Image(ImageOps.flip(self._image))
    
    def convert(self, mode: str) -> 'Image':
        """
        Convert the image to a different mode (RGB, RGBA, L, etc.).
        
        Args:
            mode: Target mode
            
        Returns:
            Converted image
        """
        return Image(self._image.convert(mode))
    
    def add_alpha(self, alpha: Union[float, np.ndarray, 'Image']) -> 'Image':
        """
        Add or modify alpha channel.
        
        Args:
            alpha: Alpha value (0.0-1.0) or alpha mask
            
        Returns:
            RGBA image
        """
        if self._image.mode != "RGBA":
            rgba = self._image.convert("RGBA")
        else:
            rgba = self._image.copy()
        
        if isinstance(alpha, (int, float)):
            # Uniform alpha
            alpha_value = int(alpha * 255)
            r, g, b, a = rgba.split()
            a = PILImage.new('L', self._image.size, alpha_value)
            return Image(PILImage.merge('RGBA', (r, g, b, a)))
        elif isinstance(alpha, np.ndarray):
            # Alpha mask as numpy array
            if alpha.shape[:2] != (self.height, self.width):
                raise ValueError("Alpha mask dimensions must match image dimensions")
            alpha_img = PILImage.fromarray((alpha * 255).astype(np.uint8), "L")
            r, g, b, _ = rgba.split()
            return Image(PILImage.merge('RGBA', (r, g, b, alpha_img)))
        elif isinstance(alpha, Image):
            # Alpha mask as image
            if alpha.width != self.width or alpha.height != self.height:
                alpha = alpha.resize((self.width, self.height))
            alpha_img = alpha.convert("L").pil_image
            r, g, b, _ = rgba.split()
            return Image(PILImage.merge('RGBA', (r, g, b, alpha_img)))
        else:
            raise TypeError(f"Unsupported alpha type: {type(alpha)}")
    
    def compose(self, other: 'Image', position: Tuple[int, int] = (0, 0)) -> 'Image':
        """
        Compose this image with another image at specified position.
        
        Args:
            other: Image to compose with
            position: Position (x, y) to place the other image
            
        Returns:
            Composed image
        """
        if self._image.mode != "RGBA":
            base = self._image.convert("RGBA")
        else:
            base = self._image.copy()
        
        if other.mode != "RGBA":
            overlay = other.pil_image.convert("RGBA")
        else:
            overlay = other.pil_image
        
        result = base.copy()
        result.alpha_composite(overlay, position)
        return Image(result)
    
    def save(self, path: Union[str, Path], format: Optional[str] = None, **kwargs) -> None:
        """
        Save the image to a file.
        
        Args:
            path: Output file path
            format: Optional file format (deduced from extension if not provided)
            **kwargs: Additional save parameters
        """
        self._image.save(path, format=format, **kwargs)
    
    def to_bytes(self, format: str = "PNG") -> bytes:
        """
        Convert image to bytes.
        
        Args:
            format: Image format
            
        Returns:
            Image data as bytes
        """
        import io
        buf = io.BytesIO()
        self._image.save(buf, format=format)
        return buf.getvalue()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Image':
        """
        Create image from bytes.
        
        Args:
            data: Image data as bytes
            
        Returns:
            Image object
        """
        import io
        buf = io.BytesIO(data)
        return cls(PILImage.open(buf))
    
    @staticmethod
    def blend(image1: 'Image', image2: 'Image', alpha: float) -> 'Image':
        """
        Blend two images.
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        # Ensure both images are in RGB mode
        img1 = image1.pil_image
        img2 = image2.pil_image
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")
            
        return Image(PILImage.blend(img1, img2, alpha))
    
    @staticmethod
    def multiply(image1: 'Image', image2: 'Image') -> 'Image':
        """
        Multiply blend two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Multiplied image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        arr1 = np.array(image1.pil_image.convert("RGB"), dtype=float) / 255.0
        arr2 = np.array(image2.pil_image.convert("RGB"), dtype=float) / 255.0
        multiplied = arr1 * arr2
        result = (multiplied * 255).astype(np.uint8)
        return Image(result)
    
    @staticmethod
    def screen(image1: 'Image', image2: 'Image') -> 'Image':
        """
        Screen blend two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Screened image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        arr1 = np.array(image1.pil_image.convert("RGB"), dtype=float) / 255.0
        arr2 = np.array(image2.pil_image.convert("RGB"), dtype=float) / 255.0
        screened = 1.0 - (1.0 - arr1) * (1.0 - arr2)
        result = (screened * 255).astype(np.uint8)
        return Image(result)
    
    @staticmethod
    def overlay(image1: 'Image', image2: 'Image') -> 'Image':
        """
        Overlay blend two images.
        
        Args:
            image1: First image (base)
            image2: Second image (overlay)
            
        Returns:
            Overlaid image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        arr1 = np.array(image1.pil_image.convert("RGB"), dtype=float) / 255.0
        arr2 = np.array(image2.pil_image.convert("RGB"), dtype=float) / 255.0
        
        # Overlay blend formula
        mask = arr1 < 0.5
        result = np.zeros_like(arr1)
        result[mask] = 2 * arr1[mask] * arr2[mask]
        result[~mask] = 1 - 2 * (1 - arr1[~mask]) * (1 - arr2[~mask])
        
        return Image((result * 255).astype(np.uint8))
    
    @staticmethod
    def add(image1: 'Image', image2: 'Image', alpha: float = 1.0) -> 'Image':
        """
        Add blend two images with alpha.
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blend factor for the second image
            
        Returns:
            Added image
        """
        # Ensure images are the same size
        if image1.width != image2.width or image1.height != image2.height:
            image2 = image2.resize((image1.width, image1.height))
            
        arr1 = np.array(image1.pil_image.convert("RGB"), dtype=float)
        arr2 = np.array(image2.pil_image.convert("RGB"), dtype=float) * alpha
        added = np.clip(arr1 + arr2, 0, 255).astype(np.uint8)
        return Image(added)
EOF

    # Create AgentManager class
    cat > "$SRC_DIR/$PKG_NAME/core/agent_manager.py" << 'EOF'
"""
Agent Manager for LlamaCanvas.

Implements the agent architecture that coordinates AI agents for various tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import json
import uuid

from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger
from llama_canvas.utils.config import settings

logger = get_logger(__name__)


class Agent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, capabilities: List[str]):
        """
        Initialize an agent.
        
        Args:
            name: Agent name
            capabilities: List of agent capabilities
        """
        self.name = name
        self.capabilities = capabilities
        self.id = str(uuid.uuid4())
        self.state: Dict[str, Any] = {}
        logger.info(f"Agent {name} initialized with capabilities: {', '.join(capabilities)}")
    
    def can_handle(self, task: str) -> bool:
        """
        Check if this agent can handle a specific task.
        
        Args:
            task: Task to check
            
        Returns:
            Whether the agent can handle the task
        """
        return task in self.capabilities
    
    def handle_task(self, task: str, **kwargs) -> Any:
        """
        Handle a task.
        
        Args:
            task: Task to handle
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        method_name = f"handle_{task}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(**kwargs)
        else:
            raise NotImplementedError(f"Agent {self.name} does not implement task: {task}")
    
    def __repr__(self) -> str:
        return f"Agent({self.name}, {self.id})"


class ClaudeAgent(Agent):
    """Agent that integrates with Claude API for AI tasks."""
    
    def __init__(self):
        """Initialize the Claude agent."""
        super().__init__(
            name="Claude",
            capabilities=[
                "generate_image", 
                "enhance_image", 
                "apply_style", 
                "inpaint",
                "generate_variations", 
                "annotate_image",
                "describe_image",
                "suggest_improvements"
            ]
        )
        
        # Import here to avoid circular imports
        from llama_canvas.claude.client import ClaudeClient
        
        if not settings.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set. Claude agent will not be functional.")
            self.client = None
        else:
            self.client = ClaudeClient(settings.get("ANTHROPIC_API_KEY"))
    
    def handle_generate_image(self, prompt: str, **kwargs) -> Image:
        """
        Generate an image using Claude.
        
        Args:
            prompt: Text description of the image
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image
        """
        if not self.client:
            raise ValueError("Claude client not initialized (API key missing?)")
        
        logger.info(f"Claude generating image: {prompt}")
        
        # Set default parameters if not specified
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        model = kwargs.get("model", settings.get("DEFAULT_CLAUDE_MODEL", "claude-3-opus-20240229"))
        
        # Call Claude API
        response = self.client.generate_image(
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            **kwargs
        )
        
        return response
    
    def handle_enhance_image(self, image: Image, prompt: str, **kwargs) -> Image:
        """
        Enhance an image with Claude guidance.
        
        Args:
            image: Image to enhance
            prompt: Enhancement instructions
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        if not self.client:
            raise ValueError("Claude client not initialized (API key missing?)")
            
        logger.info(f"Claude enhancing image with prompt: {prompt}")
        
        # Call Claude API
        enhanced = self.client.enhance_image(
            image=image,
            prompt=prompt,
            **kwargs
        )
        
        return enhanced
    
    def handle_describe_image(self, image: Image, **kwargs) -> str:
        """
        Generate a description of an image.
        
        Args:
            image: Image to describe
            **kwargs: Additional parameters
            
        Returns:
            Image description
        """
        if not self.client:
            raise ValueError("Claude client not initialized (API key missing?)")
            
        logger.info("Claude describing image")
        
        # Call Claude API
        description = self.client.describe_image(
            image=image,
            **kwargs
        )
        
        return description


class StableDiffusionAgent(Agent):
    """Agent that handles Stable Diffusion based tasks."""
    
    def __init__(self):
        """Initialize the Stable Diffusion agent."""
        super().__init__(
            name="StableDiffusion",
            capabilities=[
                "generate_image", 
                "inpaint", 
                "apply_style",
                "generate_variations"
            ]
        )
    
    def handle_generate_image(self, prompt: str, **kwargs) -> Image:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text description of the image
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image
        """
        # Import here to avoid circular imports
        from llama_canvas.generators import stable_diffusion
        
        logger.info(f"StableDiffusion generating image: {prompt}")
        
        # Set default parameters if not specified
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        model = kwargs.get("model", "stable-diffusion-v2")
        
        # Generate the image
        image = stable_diffusion.generate(
            prompt=prompt,
            width=width,
            height=height,
            model_version=model.split("stable-diffusion-")[1] if "stable-diffusion-" in model else model,
            **kwargs
        )
        
        return image
    
    def handle_inpaint(self, image: Image, mask: Image, prompt: Optional[str] = None, **kwargs) -> Image:
        """
        Perform inpainting using Stable Diffusion.
        
        Args:
            image: Image to inpaint
            mask: Mask of area to inpaint
            prompt: Optional guidance prompt
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image
        """
        # Import here to avoid circular imports
        from llama_canvas.processors import inpainting
        
        logger.info(f"StableDiffusion inpainting image with prompt: {prompt}")
        
        # Perform inpainting
        result = inpainting.stable_diffusion_inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            **kwargs
        )
        
        return result


class AgentManager:
    """
    Manages the agent system for LlamaCanvas.
    
    This class coordinates the various AI agents, dispatching tasks to appropriate
    agents based on their capabilities and the task requirements.
    """
    
    def __init__(self, canvas):
        """
        Initialize the agent manager.
        
        Args:
            canvas: Parent canvas
        """
        self.canvas = canvas
        self.agents: Dict[str, Agent] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        # Register built-in agents
        self._register_builtin_agents()
        
        logger.info(f"AgentManager initialized with {len(self.agents)} agents")
    
    def _register_builtin_agents(self):
        """Register the built-in agents."""
        # Register Claude agent if API key is available
        self.register_agent(ClaudeAgent())
        
        # Register model-specific agents
        self.register_agent(StableDiffusionAgent())
        
        # More agents would be registered here
    
    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the manager.
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the manager.
        
        Args:
            agent_id: ID of agent to unregister
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent.name}")
    
    def get_agent_for_task(self, task: str, **kwargs) -> Optional[Agent]:
        """
        Find an appropriate agent for a specific task.
        
        Args:
            task: Task to handle
            **kwargs: Task parameters that might influence agent selection
            
        Returns:
            Selected agent or None if no suitable agent is found
        """
        # Special case for Claude-specific tasks
        if task.startswith("claude_"):
            for agent in self.agents.values():
                if agent.name == "Claude" and agent.can_handle(task[7:]):
                    return agent
        
        # Handle explicit model requests
        if "model" in kwargs:
            model = kwargs.get("model", "")
            if model.startswith("claude-"):
                for agent in self.agents.values():
                    if agent.name == "Claude":
                        return agent
            elif model.startswith("stable-diffusion"):
                for agent in self.agents.values():
                    if agent.name == "StableDiffusion":
                        return agent
        
        # Default agent selection logic
        candidates = [
            agent for agent in self.agents.values()
            if agent.can_handle(task)
        ]
        
        if not candidates:
            return None
        
        # For now, just return the first capable agent
        # In a more advanced version, this would use heuristics to select the best agent
        return candidates[0]
    
    def execute_task(self, task: str, **kwargs) -> Any:
        """
        Execute a task using an appropriate agent.
        
        Args:
            task: Task to execute
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        agent = self.get_agent_for_task(task, **kwargs)
        
        if not agent:
            raise ValueError(f"No agent available to handle task: {task}")
        
        logger.info(f"Executing task {task} with agent {agent.name}")
        
        # Record the task in history
        task_record = {
            "task": task,
            "agent": agent.name,
            "parameters": kwargs,
            "timestamp": self._get_timestamp()
        }
        self.task_history.append(task_record)
        
        # Execute the task
        result = agent.handle_task(task, **kwargs)
        
        # Record the completion
        task_record["completed"] = True
        
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Convenience methods for common tasks
    
    def generate_image(self, prompt: str, **kwargs) -> Image:
        """
        Generate an image using the most appropriate agent.
        
        Args:
            prompt: Text description of the image
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image
        """
        return self.execute_task("generate_image", prompt=prompt, **kwargs)
    
    def enhance_resolution(self, image: Image, scale: int = 2, **kwargs) -> Image:
        """
        Enhance image resolution using the most appropriate agent.
        
        Args:
            image: Image to enhance
            scale: Upscaling factor
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        return self.execute_task("enhance_resolution", image=image, scale=scale, **kwargs)
    
    def apply_style(self, image: Image, style: Union[str, Image], **kwargs) -> Image:
        """
        Apply a style to an image using the most appropriate agent.
        
        Args:
            image: Image to stylize
            style: Style name or reference image
            **kwargs: Additional parameters
            
        Returns:
            Stylized image
        """
        return self.execute_task("apply_style", image=image, style=style, **kwargs)
    
    def inpaint(self, image: Image, mask: Image, prompt: Optional[str] = None, **kwargs) -> Image:
        """
        Perform inpainting using the most appropriate agent.
        
        Args:
            image: Image to inpaint
            mask: Mask of area to inpaint
            prompt: Optional guidance prompt
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image
        """
        return self.execute_task("inpaint", image=image, mask=mask, prompt=prompt, **kwargs)
    
    def enhance(self, image: Image, prompt: str, **kwargs) -> Image:
        """
        Enhance an image using the Claude agent.
        
        Args:
            image: Image to enhance
            prompt: Enhancement instructions
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        return self.execute_task("enhance_image", image=image, prompt=prompt, **kwargs)
    
    def describe_image(self, image: Image, **kwargs) -> str:
        """
        Generate a description of an image using Claude.
        
        Args:
            image: Image to describe
            **kwargs: Additional parameters
            
        Returns:
            Image description
        """
        return self.execute_task("describe_image", image=image, **kwargs)
EOF

    # Create utility files
    mkdir -p "$SRC_DIR/$PKG_NAME/utils"
    
    # Create logging utility
    cat > "$SRC_DIR/$PKG_NAME/utils/logging.py" << 'EOF'
"""
Logging utilities for LlamaCanvas.
"""

import logging
import sys
from typing import Optional

from llama_canvas.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers have been added
    if not logger.handlers:
        # Set log level from settings
        log_level = getattr(logging, settings.get("LOG_LEVEL", "INFO"))
        logger.setLevel(log_level)
        
        # Create console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console)
        
        # Propagate to root logger
        logger.propagate = False
    
    return logger


def setup_file_logging(log_file: str) -> None:
    """
    Set up file logging for the application.
    
    Args:
        log_file: Path to log file
    """
    root_logger = logging.getLogger()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)


def log_exception(logger: logging.Logger, exception: Exception, message: Optional[str] = None) -> None:
    """
    Log an exception with additional context.
    
    Args:
        logger: Logger to use
        exception: Exception to log
        message: Optional additional message
    """
    if message:
        logger.error(f"{message}: {str(exception)}")
    else:
        logger.error(str(exception))
    
    logger.debug("Exception details:", exc_info=True)
EOF

    # Create configuration utility
    cat > "$SRC_DIR/$PKG_NAME/utils/config.py" << 'EOF'
"""
Configuration utilities for LlamaCanvas.
"""

import os
import json
from typing import Any, Dict, Optional
from pathlib import Path

from dotenv import load_dotenv


class Settings:
    """Class to manage application settings."""
    
    def __init__(self):
        """Initialize settings from environment and config file."""
        self._settings: Dict[str, Any] = {}
        self._load_environment()
        self._load_config_file()
    
    def _load_environment(self) -> None:
        """Load settings from environment variables."""
        # Load .env file if present
        load_dotenv()
        
        # Map environment variables to settings
        env_mapping = {
            "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            "LOG_LEVEL": "LOG_LEVEL",
            "MODEL_CACHE_DIR": "MODEL_CACHE_DIR",
            "DEFAULT_CLAUDE_MODEL": "DEFAULT_CLAUDE_MODEL",
        }
        
        for env_var, setting_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._settings[setting_name] = value
    
    def _load_config_file(self) -> None:
        """Load settings from config file if present."""
        config_paths = [
            Path.home() / ".llama_canvas" / "config.json",
            Path.home() / ".config" / "llama_canvas" / "config.json",
            Path("config.json")
        ]
        
        for path in config_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        config = json.load(f)
                        self._settings.update(config)
                    break
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error loading config from {path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
        """
        self._settings[key] = value
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save settings to config file.
        
        Args:
            path: Path to save to (default: ~/.llama_canvas/config.json)
        """
        if path is None:
            path = Path.home() / ".llama_canvas" / "config.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "w") as f:
                json.dump(self._settings, f, indent=2)
        except IOError as e:
            print(f"Error saving config to {path}: {e}")
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to settings."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style setting of values."""
        self.set(key, value)


# Create global settings instance
settings = Settings()
EOF

    success "Core module files created"
}

# Create Video class implementation 
create_video_class() {
    progress "Creating Video class implementation"
    
    # Create Video class file
    cat > "$SRC_DIR/$PKG_NAME/core/video.py" << 'EOF'
"""
Video class for LlamaCanvas.

Provides a Video class for creating, editing, and managing video content.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator

import numpy as np
from PIL import Image as PILImage

from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports - only required for full functionality
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import moviepy.editor as mpy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class Video:
    """
    Video class for creating, editing and managing video content.
    
    This class provides methods for working with video, including creating videos
    from image sequences, extracting frames, applying effects, and more.
    """
    
    def __init__(
        self, 
        source: Union[List[Image], str, Path, 'Video', None] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: float = 24.0,
        duration: Optional[float] = None
    ):
        """
        Initialize a video object.
        
        Args:
            source: Source for the video (image list, file path, or another Video)
            width: Video width (optional, determined from source if not provided)
            height: Video height (optional, determined from source if not provided)
            fps: Frames per second
            duration: Video duration in seconds (optional)
        """
        self._frames: List[Image] = []
        self._fps = fps
        self._duration = duration
        self._width = width
        self._height = height
        self._audio_path: Optional[str] = None
        
        # Handle different source types
        if source is None:
            # Empty video
            if width is None or height is None:
                raise ValueError("Width and height must be provided for empty videos")
            logger.info(f"Created empty video ({width}x{height}, {fps} fps)")
        
        elif isinstance(source, list):
            # List of images
            if not source:
                raise ValueError("Empty frame list provided")
            
            self._frames = list(source)  # Make a copy
            
            # Set dimensions from first frame if not specified
            if width is None:
                self._width = source[0].width
            if height is None:
                self._height = source[0].height
            
            # Set duration based on frame count if not specified
            if duration is None:
                self._duration = len(source) / fps
                
            logger.info(f"Created video from {len(source)} frames ({self._width}x{self._height}, {fps} fps)")
        
        elif isinstance(source, (str, Path)):
            # Video file
            if not MOVIEPY_AVAILABLE:
                raise ImportError("moviepy is required for loading video files")
            
            path = str(source)
            video = mpy.VideoFileClip(path)
            
            # Set properties from video file
            self._fps = video.fps if fps is None else fps
            self._duration = video.duration
            
            # Set dimensions
            if width is None or height is None:
                self._width = int(video.size[0])
                self._height = int(video.size[1])
            else:
                self._width = width
                self._height = height
            
            # Extract frames
            self._extract_frames_from_clip(video)
            
            # Extract audio if available
            if video.audio is not None:
                self._audio_path = tempfile.mktemp(suffix=".mp3")
                video.audio.write_audiofile(self._audio_path, verbose=False, logger=None)
            
            video.close()
            logger.info(f"Loaded video from {path} ({self._width}x{self._height}, {self._fps} fps)")
        
        elif isinstance(source, Video):
            # Another video object
            self._frames = [frame.copy() for frame in source._frames]
            self._fps = source._fps if fps is None else fps
            self._duration = source._duration
            self._width = source._width if width is None else width
            self._height = source._height if height is None else height
            
            # Copy audio if available
            if source._audio_path and os.path.exists(source._audio_path):
                self._audio_path = tempfile.mktemp(suffix=".mp3")
                import shutil
                shutil.copy(source._audio_path, self._audio_path)
                
            logger.info(f"Created video copy ({self._width}x{self._height}, {self._fps} fps)")
        
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")
        
        # Ensure all frames have consistent dimensions
        self._normalize_frames()
    
    @property
    def width(self) -> int:
        """Get video width."""
        return self._width
    
    @property
    def height(self) -> int:
        """Get video height."""
        return self._height
    
    @property
    def fps(self) -> float:
        """Get frames per second."""
        return self._fps
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self._duration is not None:
            return self._duration
        return len(self._frames) / self._fps if self._frames else 0
    
    @property
    def frame_count(self) -> int:
        """Get number of frames."""
        return len(self._frames)
    
    @property
    def has_audio(self) -> bool:
        """Check if video has audio."""
        return self._audio_path is not None and os.path.exists(self._audio_path)
    
    def _extract_frames_from_clip(self, clip: 'mpy.VideoClip') -> None:
        """
        Extract frames from a MoviePy video clip.
        
        Args:
            clip: MoviePy video clip
        """
        # Calculate the total number of frames
        total_frames = int(clip.fps * clip.duration)
        
        # Extract frames
        self._frames = []
        for i in range(total_frames):
            time = i / clip.fps
            frame = clip.get_frame(time)
            
            # Convert to Image
            pil_image = PILImage.fromarray(frame)
            
            # Resize if needed
            if self._width is not None and self._height is not None:
                if pil_image.width != self._width or pil_image.height != self._height:
                    pil_image = pil_image.resize((self._width, self._height))
            
            self._frames.append(Image(pil_image))
    
    def _normalize_frames(self) -> None:
        """Ensure all frames have consistent dimensions."""
        if not self._frames:
            return
            
        # Check if any frames need resizing
        for i, frame in enumerate(self._frames):
            if frame.width != self._width or frame.height != self._height:
                self._frames[i] = frame.resize((self._width, self._height))
    
    def get_frame(self, index: int) -> Image:
        """
        Get a specific frame by index.
        
        Args:
            index: Frame index
            
        Returns:
            Frame as Image
        """
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Frame index {index} out of range (0-{len(self._frames)-1})")
        
        return self._frames[index]
    
    def get_frame_at_time(self, time: float) -> Image:
        """
        Get a frame at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Frame as Image
        """
        if time < 0 or time > self.duration:
            raise ValueError(f"Time {time} out of range (0-{self.duration})")
        
        # Convert time to frame index
        index = int(time * self._fps)
        index = min(index, len(self._frames) - 1)  # Ensure we don't exceed frame count
        
        return self._frames[index]
    
    def add_frame(self, frame: Image) -> None:
        """
        Add a frame to the video.
        
        Args:
            frame: Frame to add
        """
        # Resize frame if needed
        if frame.width != self._width or frame.height != self._height:
            frame = frame.resize((self._width, self._height))
        
        self._frames.append(frame)
        
        # Update duration if it was explicitly set
        if self._duration is not None:
            self._duration = len(self._frames) / self._fps
    
    def save(self, path: Union[str, Path], codec: str = "libx264", audio: bool = True) -> None:
        """
        Save the video to a file.
        
        Args:
            path: Output file path
            codec: Video codec to use
            audio: Whether to include audio
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError("moviepy is required for saving videos")
        
        path = str(path)
        
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as images
            frame_paths = []
            for i, frame in enumerate(self._frames):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame.save(frame_path)
                frame_paths.append(frame_path)
            
            # Create a clip from frames
            clip = mpy.ImageSequenceClip(frame_paths, fps=self._fps)
            
            # Add audio if available and requested
            if audio and self.has_audio:
                audio_clip = mpy.AudioFileClip(self._audio_path)
                clip = clip.set_audio(audio_clip)
            
            # Set duration if explicitly set
            if self._duration is not None:
                clip = clip.set_duration(self._duration)
            
            # Write the video file
            clip.write_videofile(path, codec=codec, audio=audio and self.has_audio, logger=None)
            
            clip.close()
            
            logger.info(f"Saved video to {path} ({self._width}x{self._height}, {self._fps} fps)")
    
    def extract_frames(self, output_dir: Union[str, Path], prefix: str = "frame_", format: str = "png") -> List[Path]:
        """
        Extract all frames to image files.
        
        Args:
            output_dir: Directory to save frames
            prefix: Filename prefix
            format: Image format
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, frame in enumerate(self._frames):
            filename = f"{prefix}{i:06d}.{format.lower()}"
            path = output_dir / filename
            frame.save(path)
            paths.append(path)
        
        logger.info(f"Extracted {len(paths)} frames to {output_dir}")
        return paths
    
    def __len__(self) -> int:
        """Get number of frames."""
        return len(self._frames)
    
    def __getitem__(self, index: int) -> Image:
        """Get a frame by index."""
        return self.get_frame(index)
    
    def __iter__(self) -> Iterator[Image]:
        """Iterate over frames."""
        return iter(self._frames)
EOF

    success "Video class implementation created"
}

# Create CLI module
create_cli_module() {
    progress "Creating CLI module"
    
    # Create CLI main file
    cat > "$SRC_DIR/$PKG_NAME/cli.py" << 'EOF'
"""
Command-line interface for LlamaCanvas.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich import print as rprint

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger, setup_file_logging
from llama_canvas.utils.config import settings

# Create Typer app
app = typer.Typer(
    name="llama-canvas",
    help="Advanced AI-driven multi-modal generation platform with Claude API integration",
    add_completion=False,
)

# Create console for rich output
console = Console()

# Set up logger
logger = get_logger(__name__)

# ASCII art banner for CLI
BANNER = r"""
    🦙 [bold magenta]LLAMA CANVAS[/bold magenta] 🦙
    [cyan]Advanced AI-driven Multi-modal Generation[/cyan]
"""


def print_banner():
    """Print CLI banner."""
    console.print(Panel(BANNER, border_style="magenta"))


def check_api_key():
    """Check if Anthropic API key is available."""
    api_key = settings.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[yellow]⚠️ ANTHROPIC_API_KEY not found in environment or config.[/yellow]"
        )
        console.print(
            "Set your API key using:\n"
            "  [green]export ANTHROPIC_API_KEY=your_key_here[/green]\n"
            "Or add it to your .env file."
        )
        return False
    return True


@app.callback()
def callback():
    """LlamaCanvas CLI."""
    # Set up file logging if needed
    log_file = settings.get("LOG_FILE")
    if log_file:
        setup_file_logging(log_file)


@app.command()
def version():
    """Show version information."""
    from llama_canvas import __version__

    print_banner()
    console.print(f"[bold]LlamaCanvas[/bold] version: [cyan]{__version__}[/cyan]")
    
    # Check for Claude API access
    if check_api_key():
        console.print("[green]✓[/green] Claude API key found.")
    
    # Check for GPU support
    try:
        import torch
        console.print(f"PyTorch version: [cyan]{torch.__version__}[/cyan]")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            console.print(f"[green]✓[/green] CUDA available: [cyan]{device_name}[/cyan]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            console.print("[green]✓[/green] Apple MPS available")
        else:
            console.print("[yellow]⚠️ GPU acceleration not available. Using CPU only.[/yellow]")
    except ImportError:
        console.print("[yellow]⚠️ PyTorch not installed.[/yellow]")


@app.command("generate")
def generate_image(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    model: str = typer.Option("stable-diffusion-v2", help="Model to use for generation"),
    width: int = typer.Option(512, help="Image width"),
    height: int = typer.Option(512, help="Image height"),
    output: Path = typer.Option("output.png", help="Output file path"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    guidance_scale: float = typer.Option(7.5, help="Guidance scale for diffusion models"),
    num_inference_steps: int = typer.Option(50, help="Number of inference steps for diffusion models"),
    use_claude: bool = typer.Option(False, help="Use Claude for image generation"),
):
    """Generate an image from a text prompt."""
    print_banner()
    
    console.print(f"[bold]Generating image[/bold] from prompt:")
    console.print(f"[italic cyan]\"{prompt}\"[/italic cyan]")
    
    if use_claude:
        if not check_api_key():
            return
        model = settings.get("DEFAULT_CLAUDE_MODEL", "claude-3-opus-20240229")
    
    # Create canvas
    canvas = Canvas(width=width, height=height)
    
    # Show generation parameters
    console.print(f"Using model: [bold]{model}[/bold]")
    console.print(f"Output size: [bold]{width}x{height}[/bold]")
    
    # Generate image with progress bar
    with Progress() as progress:
        task = progress.add_task("Generating...", total=100)
        
        # Generate image
        if use_claude:
            image = canvas.agents.generate_image(prompt, model=model, width=width, height=height)
        else:
            image = canvas.generate_from_text(
                prompt,
                model=model,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        
        progress.update(task, completed=100)
    
    # Save image
    image.save(output)
    
    # Show success message
    console.print(f"[green]✓[/green] Image saved to [bold]{output}[/bold]")


@app.command("enhance")
def enhance_image(
    input_file: Path = typer.Argument(..., help="Input image file"),
    output: Path = typer.Option("enhanced.png", help="Output file path"),
    scale: int = typer.Option(2, help="Upscaling factor"),
    prompt: Optional[str] = typer.Option(None, help="Enhancement guidance prompt"),
    use_claude: bool = typer.Option(False, help="Use Claude for enhancement"),
):
    """Enhance an image (super-resolution or guided enhancement)."""
    print_banner()
    
    if use_claude and not check_api_key():
        return
    
    # Load input image
    console.print(f"Loading image: [bold]{input_file}[/bold]")
    image = Image(input_file)
    
    # Create canvas
    canvas = Canvas()
    
    # Enhance image with progress bar
    with Progress() as progress:
        task = progress.add_task("Enhancing image...", total=100)
        
        if use_claude and prompt:
            # Use Claude for guided enhancement
            console.print(f"Using Claude to enhance with prompt: [italic cyan]\"{prompt}\"[/italic cyan]")
            enhanced = canvas.agents.enhance(image, prompt=prompt)
        else:
            # Use super-resolution
            console.print(f"Enhancing resolution by {scale}x")
            enhanced = canvas.enhance_resolution(image, scale=scale)
        
        progress.update(task, completed=100)
    
    # Save image
    enhanced.save(output)
    
    # Show success message
    console.print(f"[green]✓[/green] Enhanced image saved to [bold]{output}[/bold]")


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
EOF

    success "CLI module created"
}

# Create Claude API client
create_claude_client() {
    progress "Creating Claude API client"
    
    # Ensure directory exists
    ensure_dir "$SRC_DIR/$PKG_NAME/claude"
    touch "$SRC_DIR/$PKG_NAME/claude/__init__.py"
    
    # Create Claude client file
    cat > "$SRC_DIR/$PKG_NAME/claude/client.py" << 'EOF'
"""
Claude API client for LlamaCanvas.

Handles communication with Anthropic's Claude API for various AI tasks.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO

import requests
from PIL import Image as PILImage

from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger

logger = get_logger(__name__)


class ClaudeClient:
    """Client for interacting with Claude API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
        """
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com"
        self.api_version = "2023-06-01"  # This would be updated as Anthropic updates their API
        
        # Default model settings
        self.default_model = "claude-3-opus-20240229"
        self.default_max_tokens = 1024
        
        logger.info("Claude API client initialized")
    
    def _create_headers(self) -> Dict[str, str]:
        """
        Create request headers for Claude API.
        
        Returns:
            Request headers
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }
    
    def _encode_image_to_base64(self, image: Union[Image, PILImage.Image, str, Path, BinaryIO]) -> str:
        """
        Encode image to base64 for API requests.
        
        Args:
            image: Image to encode
            
        Returns:
            Base64-encoded image
        """
        if isinstance(image, Image):
            pil_image = image.pil_image
        elif isinstance(image, PILImage.Image):
            pil_image = image
        elif isinstance(image, (str, Path)):
            pil_image = PILImage.open(image)
        elif hasattr(image, "read"):
            pil_image = PILImage.open(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        
        # Get base64 string
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return img_str
    
    def describe_image(
        self,
        image: Image,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a description of an image.
        
        Args:
            image: Image to describe
            model: Claude model to use
            **kwargs: Additional parameters
            
        Returns:
            Image description
        """
        model = model or self.default_model
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        
        # Encode image
        image_b64 = self._encode_image_to_base64(image)
        
        # Prepare request
        headers = self._create_headers()
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please describe this image in detail."
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        
        # Make API request
        try:
            response = requests.post(f"{self.api_url}/v1/messages", headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            # Extract text from the response
            description = response_data["content"][0]["text"]
            
            logger.info("Successfully generated image description")
            return description
            
        except (requests.RequestException, KeyError, IndexError) as e:
            error_msg = f"Error generating image description: {str(e)}"
            logger.error(error_msg)
            
            # Return a fallback message
            return f"Error: Could not generate description. {error_msg}"
EOF
    
    success "Claude API client created"
}

# Create API server implementation
create_api_server() {
    progress "Creating API server implementation"
    
    # Ensure directories exist
    ensure_dir "$SRC_DIR/$PKG_NAME/api"
    ensure_dir "$SRC_DIR/$PKG_NAME/api/templates"
    ensure_dir "$SRC_DIR/$PKG_NAME/api/static"
    ensure_dir "$SRC_DIR/$PKG_NAME/api/static/css"
    ensure_dir "$SRC_DIR/$PKG_NAME/api/static/js"
    
    # Create base __init__.py
    touch "$SRC_DIR/$PKG_NAME/api/__init__.py"
    
    # Create the FastAPI application
    cat > "$SRC_DIR/$PKG_NAME/api/app.py" << 'EOF'
"""
FastAPI application for LlamaCanvas.

Provides a web API and interface for using LlamaCanvas capabilities.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_canvas.core.canvas import Canvas
from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger
from llama_canvas.utils.config import settings

# Set up logger
logger = get_logger(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

# Templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="LlamaCanvas API",
        description="API for LlamaCanvas - Advanced AI-driven multi-modal generation platform",
        version="1.0.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    
    # Create temp directory for outputs
    output_dir = Path(tempfile.gettempdir()) / "llama_canvas_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Serve the web UI
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return templates.TemplateResponse("index.html", {"request": {}})
    
    # Model schemas
    class GenerateImageRequest(BaseModel):
        prompt: str
        model: str = "stable-diffusion-v2"
        width: int = 512
        height: int = 512
        guidance_scale: float = 7.5
        num_inference_steps: int = 50
        use_claude: bool = False
    
    class EnhanceImageRequest(BaseModel):
        prompt: Optional[str] = None
        scale: int = 2
        use_claude: bool = False
    
    # API endpoints
    @app.post("/api/generate")
    async def generate_image(req: GenerateImageRequest):
        try:
            # Create canvas
            canvas = Canvas(width=req.width, height=req.height)
            
            # Generate image
            if req.use_claude:
                if not settings.get("ANTHROPIC_API_KEY"):
                    raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")
                image = canvas.agents.generate_image(
                    req.prompt, 
                    model=settings.get("DEFAULT_CLAUDE_MODEL", "claude-3-opus-20240229"),
                    width=req.width,
                    height=req.height
                )
            else:
                image = canvas.generate_from_text(
                    req.prompt,
                    model=req.model,
                    width=req.width,
                    height=req.height,
                    guidance_scale=req.guidance_scale,
                    num_inference_steps=req.num_inference_steps
                )
            
            # Save generated image
            output_path = output_dir / f"generated_{os.urandom(8).hex()}.png"
            image.save(output_path)
            
            return {"success": True, "image_path": str(output_path)}
        
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/upload")
    async def upload_image(file: UploadFile = File(...)):
        try:
            # Create temp file for uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp:
                temp.write(await file.read())
                temp_path = temp.name
            
            # Load the image to verify it's valid
            try:
                Image(temp_path)
            except Exception as e:
                os.unlink(temp_path)
                raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
            
            return {"success": True, "image_path": temp_path}
        
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/enhance")
    async def enhance_image(
        req: EnhanceImageRequest,
        image_path: str = Form(...)
    ):
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Image not found")
            
            # Load image
            image = Image(image_path)
            
            # Create canvas
            canvas = Canvas()
            
            # Enhance image
            if req.use_claude and req.prompt:
                if not settings.get("ANTHROPIC_API_KEY"):
                    raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")
                enhanced = canvas.agents.enhance(image, prompt=req.prompt)
            else:
                enhanced = canvas.enhance_resolution(image, scale=req.scale)
            
            # Save enhanced image
            output_path = output_dir / f"enhanced_{os.urandom(8).hex()}.png"
            enhanced.save(output_path)
            
            return {"success": True, "image_path": str(output_path)}
        
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/image/{filename}")
    async def get_image(filename: str):
        # Serve image files
        image_path = output_dir / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(image_path)
    
    @app.get("/api/models")
    async def get_models():
        # Return available models
        models = {
            "text_to_image": [
                {"id": "stable-diffusion-v1-5", "name": "Stable Diffusion v1.5"},
                {"id": "stable-diffusion-v2", "name": "Stable Diffusion v2"},
                {"id": "stable-diffusion-xl", "name": "Stable Diffusion XL"},
            ],
            "claude": []
        }
        
        # Add Claude models if API key is set
        if settings.get("ANTHROPIC_API_KEY"):
            models["claude"] = [
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
            ]
        
        return models
    
    return app


def run_app(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the FastAPI application.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
    """
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_app()
EOF

    # Create a basic HTML template
    cat > "$SRC_DIR/$PKG_NAME/api/templates/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LlamaCanvas Web UI</title>
    <link rel="stylesheet" href="/static/css/styles.css">
    <style>
        :root {
            --primary-color: #6e4799;
            --secondary-color: #f0c6ff;
            --background-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #333333;
            --border-color: #dee2e6;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background-color: var(--primary-color);
            color: white;
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .logo {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }
        
        .tagline {
            font-style: italic;
            opacity: 0.8;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .card-title {
            margin-top: 0;
            color: var(--primary-color);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        input[type="text"], 
        input[type="number"],
        textarea,
        select {
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-size: 1rem;
        }
        
        textarea {
            height: 100px;
            resize: vertical;
        }
        
        button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1rem;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: #5a3b7d;
        }
        
        .result-container {
            text-align: center;
        }
        
        .result-image {
            max-width: 100%;
            max-height: 600px;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }
        
        .tab.active {
            border-bottom-color: var(--primary-color);
            font-weight: bold;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .loading::after {
            content: "...";
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: "."; }
            40% { content: ".."; }
            60%, 100% { content: "..."; }
        }
        
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.9rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="logo">🦙 LlamaCanvas</h1>
            <p class="tagline">Advanced AI-driven multi-modal generation platform</p>
        </div>
    </div>
    
    <div class="container">
        <div class="tabs">
            <div class="tab active" data-tab="generate">Generate Image</div>
            <div class="tab" data-tab="enhance">Enhance Image</div>
        </div>
        
        <div class="tab-content active" id="generate-tab">
            <div class="card">
                <h2 class="card-title">Generate Image from Text</h2>
                <form id="generate-form">
                    <div class="form-group">
                        <label for="prompt">Prompt</label>
                        <textarea id="prompt" name="prompt" required placeholder="Describe the image you want to generate..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="model">Model</label>
                        <select id="model" name="model">
                            <option value="stable-diffusion-v2">Stable Diffusion v2</option>
                            <option value="stable-diffusion-v1-5">Stable Diffusion v1.5</option>
                            <option value="stable-diffusion-xl">Stable Diffusion XL</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <div style="display: flex; gap: 20px;">
                            <div style="flex: 1;">
                                <label for="width">Width</label>
                                <input type="number" id="width" name="width" value="512" min="128" max="1024" step="64">
                            </div>
                            <div style="flex: 1;">
                                <label for="height">Height</label>
                                <input type="number" id="height" name="height" value="512" min="128" max="1024" step="64">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <div style="display: flex; align-items: center;">
                            <input type="checkbox" id="use-claude" name="use_claude" style="width: auto; margin-right: 10px;">
                            <label for="use-claude" style="display: inline;">Use Claude (requires API key)</label>
                        </div>
                    </div>
                    
                    <button type="submit">Generate</button>
                </form>
            </div>
            