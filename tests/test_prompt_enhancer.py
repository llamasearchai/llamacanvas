"""Unit tests for the prompt enhancer utilities."""

import unittest
from llama_canvas.utils.prompt_enhancer import (
    enhance_prompt,
    analyze_prompt,
    generate_variations
)

class TestPromptEnhancer(unittest.TestCase):
    """Test the prompt enhancement utilities."""
    
    def test_enhance_prompt_basic(self):
        """Test basic prompt enhancement."""
        prompt = "a cat"
        enhanced = enhance_prompt(prompt, style="photorealistic", detail_level="high")
        
        # Check that the original prompt is preserved
        self.assertIn("a cat", enhanced)
        
        # Check that style and detail keywords were added
        self.assertIn("photorealistic", enhanced.lower())
        self.assertIn("detailed", enhanced.lower())
    
    def test_enhance_prompt_with_lighting(self):
        """Test prompt enhancement with lighting options."""
        prompt = "a mountain landscape"
        enhanced = enhance_prompt(
            prompt, 
            style="digital art", 
            lighting="dramatic",
            detail_level="high"
        )
        
        # Check that lighting was added
        self.assertIn("dramatic lighting", enhanced.lower())
        
        # Check that style was added
        self.assertIn("digital art", enhanced.lower())
    
    def test_enhance_prompt_with_aspect(self):
        """Test prompt enhancement with aspect specification."""
        prompt = "a city street"
        enhanced = enhance_prompt(
            prompt,
            aspect="aerial view",
            detail_level="medium"
        )
        
        # Check that aspect was added
        self.assertIn("aerial view", enhanced.lower())
    
    def test_enhance_prompt_no_duplicates(self):
        """Test that enhancement doesn't duplicate existing terms."""
        prompt = "a detailed photorealistic portrait with dramatic lighting"
        enhanced = enhance_prompt(
            prompt,
            style="photorealistic",
            lighting="dramatic",
            detail_level="high"
        )
        
        # Count occurrences to ensure no duplication
        self.assertEqual(enhanced.lower().count("photorealistic"), 1)
        self.assertEqual(enhanced.lower().count("dramatic lighting"), 1)
    
    def test_analyze_prompt_basic(self):
        """Test basic prompt analysis."""
        prompt = "cat"
        analysis = analyze_prompt(prompt)
        
        # Check analysis structure
        self.assertIn("word_count", analysis)
        self.assertIn("specificity", analysis)
        self.assertIn("suggestions", analysis)
        
        # Check that suggestions are provided for this simple prompt
        self.assertGreater(len(analysis["suggestions"]), 0)
    
    def test_analyze_prompt_detailed(self):
        """Test analysis of a detailed prompt."""
        prompt = "A photorealistic portrait of a cat with green eyes in a garden, soft lighting, detailed fur"
        analysis = analyze_prompt(prompt)
        
        # This prompt has style, lighting, detail, and environment
        self.assertTrue(analysis["has_style"])
        self.assertTrue(analysis["has_lighting"])
        self.assertTrue(analysis["has_detail"])
        self.assertTrue(analysis["has_environment"])
        self.assertTrue(analysis["has_color"])
        
        # Should have high specificity
        self.assertEqual(analysis["specificity"], "high")
        
        # Should have fewer suggestions
        self.assertLess(len(analysis["suggestions"]), 3)
    
    def test_generate_variations(self):
        """Test generation of prompt variations."""
        prompt = "a beach at sunset"
        variations = generate_variations(prompt, count=5)
        
        # Check that we got the right number of variations
        self.assertEqual(len(variations), 5)
        
        # Check that all variations are different
        self.assertEqual(len(set(variations)), 5)
        
        # Check that all variations contain the original prompt
        for var in variations:
            self.assertIn("a beach at sunset", var)
            
            # Each variation should have a style and lighting
            has_style = any(s in var.lower() for s in ["photorealistic", "digital art", "oil painting", 
                                                    "watercolor", "sketch", "anime", "sci-fi", "fantasy"])
            has_lighting = any(l in var.lower() for l in ["lighting", "light"])
            
            self.assertTrue(has_style or has_lighting)

if __name__ == "__main__":
    unittest.main() 