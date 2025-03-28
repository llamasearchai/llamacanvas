"""
Prompt enhancement utilities for better image generation results.

This module provides tools to enhance text prompts for image generation,
adding details and improving the quality of the resulting images.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Pre-defined styles with their descriptions and keywords
STYLES = {
    "photorealistic": {
        "description": "Highly detailed photorealistic style",
        "keywords": ["photorealistic", "detailed", "high resolution", "realistic lighting", "high definition"]
    },
    "digital art": {
        "description": "Polished digital art style",
        "keywords": ["digital art", "vibrant", "detailed", "illustration", "artstation"]
    },
    "oil painting": {
        "description": "Traditional oil painting style",
        "keywords": ["oil painting", "textured", "brush strokes", "canvas", "artistic"]
    },
    "watercolor": {
        "description": "Soft watercolor painting style",
        "keywords": ["watercolor", "soft edges", "flowy", "painterly", "artistic"]
    },
    "sketch": {
        "description": "Hand-drawn sketch style",
        "keywords": ["sketch", "hand-drawn", "pencil", "line art", "sketchy"]
    },
    "anime": {
        "description": "Japanese anime style",
        "keywords": ["anime", "manga", "japanese style", "animated", "stylized"]
    },
    "sci-fi": {
        "description": "Futuristic sci-fi style",
        "keywords": ["sci-fi", "futuristic", "cyberpunk", "neon", "technological"]
    },
    "fantasy": {
        "description": "Magical fantasy style",
        "keywords": ["fantasy", "magical", "mystical", "dreamy", "ethereal"]
    }
}

# Quality and detail enhancers
DETAIL_LEVELS = {
    "low": ["simple", "minimal details"],
    "medium": ["detailed", "good quality"],
    "high": ["highly detailed", "intricate", "4K", "high resolution"],
    "ultra": ["extremely detailed", "intricate", "8K", "ultra high definition", "professional"]
}

# Lighting enhancers
LIGHTING_OPTIONS = {
    "soft": ["soft lighting", "gentle light", "diffused"],
    "dramatic": ["dramatic lighting", "high contrast", "rim light"],
    "natural": ["natural lighting", "golden hour", "sunlight"],
    "studio": ["studio lighting", "professional lighting", "three-point lighting"]
}

def enhance_prompt(
    prompt: str,
    style: Optional[str] = None,
    detail_level: str = "medium",
    lighting: Optional[str] = None,
    aspect: Optional[str] = None,
    negative_prompt: Optional[str] = None
) -> str:
    """
    Enhance a prompt with additional details for better image generation.
    
    Args:
        prompt: Base prompt to enhance
        style: Style to apply (see STYLES)
        detail_level: Level of detail to add (low, medium, high, ultra)
        lighting: Lighting style to add
        aspect: Aspect/perspective description (e.g., "overhead view", "wide angle")
        negative_prompt: Elements to avoid (returned separately)
        
    Returns:
        Enhanced prompt string
    """
    # Start with the original prompt
    enhanced = prompt.strip()
    
    # Check if the prompt already contains detailed specifications
    has_style = any(s in enhanced.lower() for s in STYLES.keys())
    has_detail = any(d in enhanced.lower() for level in DETAIL_LEVELS.values() for d in level)
    has_lighting = any(l in enhanced.lower() for level in LIGHTING_OPTIONS.values() for l in level)
    
    # Add style if specified and not already present
    if style and style in STYLES and not has_style:
        style_keywords = STYLES[style]["keywords"]
        enhanced = f"{enhanced}, {style_keywords[0]}"
        
        # Add some additional style keywords
        if len(style_keywords) > 1:
            additional = style_keywords[1]
            enhanced = f"{enhanced}, {additional}"
    
    # Add detail level if not already specified
    if detail_level in DETAIL_LEVELS and not has_detail:
        details = DETAIL_LEVELS[detail_level]
        enhanced = f"{enhanced}, {details[0]}"
        
        # Add more detail descriptors for high/ultra
        if detail_level in ["high", "ultra"] and len(details) > 2:
            enhanced = f"{enhanced}, {details[2]}"
    
    # Add lighting if specified and not already present
    if lighting and lighting in LIGHTING_OPTIONS and not has_lighting:
        light_desc = LIGHTING_OPTIONS[lighting][0]
        enhanced = f"{enhanced}, {light_desc}"
    
    # Add aspect/perspective if specified
    if aspect:
        enhanced = f"{enhanced}, {aspect}"
    
    # Clean up any double commas or spaces
    enhanced = re.sub(r',\s*,', ',', enhanced)
    enhanced = re.sub(r'\s+', ' ', enhanced)
    
    logger.debug(f"Enhanced prompt: {enhanced}")
    return enhanced.strip()

def analyze_prompt(prompt: str) -> Dict:
    """
    Analyze a prompt and suggest improvements.
    
    Args:
        prompt: Prompt to analyze
        
    Returns:
        Dictionary with analysis results and suggestions
    """
    words = len(prompt.split())
    specificity = "low" if words < 5 else "medium" if words < 15 else "high"
    
    suggestions = []
    
    # Check for style specification
    has_style = any(s in prompt.lower() for s in STYLES.keys())
    if not has_style:
        suggestions.append("Add a specific art style (e.g., photorealistic, digital art)")
    
    # Check for detail level
    has_detail = any(d in prompt.lower() for level in DETAIL_LEVELS.values() for d in level)
    if not has_detail:
        suggestions.append("Specify level of detail (e.g., highly detailed, 4K)")
    
    # Check for subject clarity
    if words < 3:
        suggestions.append("Describe the main subject more clearly")
    
    # Check for environmental/contextual details
    environment_keywords = ["background", "scene", "setting", "environment", "landscape"]
    has_environment = any(e in prompt.lower() for e in environment_keywords)
    if not has_environment and words > 3:  # Only suggest if prompt has some substance
        suggestions.append("Add environmental context or background description")
    
    # Check for lighting
    lighting_keywords = ["lighting", "light", "shadow", "illuminated", "bright", "dark"]
    has_lighting = any(l in prompt.lower() for l in lighting_keywords)
    if not has_lighting:
        suggestions.append("Add lighting description (e.g., soft lighting, dramatic shadows)")
    
    # Check for color information
    color_keywords = ["color", "red", "blue", "green", "yellow", "purple", "orange", 
                     "white", "black", "vibrant", "monochrome", "colorful"]
    has_color = any(c in prompt.lower() for c in color_keywords)
    if not has_color:
        suggestions.append("Include color information or palette")
    
    return {
        "word_count": words,
        "specificity": specificity,
        "has_style": has_style,
        "has_detail": has_detail,
        "has_environment": has_environment,
        "has_lighting": has_lighting,
        "has_color": has_color,
        "suggestions": suggestions
    }

def generate_variations(
    prompt: str,
    count: int = 3,
    variation_strength: float = 0.3
) -> List[str]:
    """
    Generate variations of a prompt for creating diverse images.
    
    Args:
        prompt: Base prompt
        count: Number of variations to generate
        variation_strength: How much to vary the prompt (0.0 to 1.0)
        
    Returns:
        List of prompt variations
    """
    base_enhanced = enhance_prompt(prompt, detail_level="high")
    variations = [base_enhanced]
    
    # Style variations
    available_styles = list(STYLES.keys())
    
    # Lighting variations
    available_lighting = list(LIGHTING_OPTIONS.keys())
    
    # Generate variations
    for i in range(1, count):
        # Pick a different style and lighting for variation
        style_idx = i % len(available_styles)
        light_idx = (i + 1) % len(available_lighting)
        
        # Create variation with different style and lighting
        variation = enhance_prompt(
            prompt,
            style=available_styles[style_idx],
            lighting=available_lighting[light_idx],
            detail_level="high"
        )
        
        variations.append(variation)
    
    return variations 