"""
Tests for the AgentManager class in LlamaCanvas.

This module contains tests for the agent system and AI integrations.
"""

import os
import tempfile
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from llama_canvas.core.image import Image
from llama_canvas.core.canvas import Canvas
from llama_canvas.core.agent_manager import AgentManager, Agent, ClaudeAgent


@pytest.fixture
def mock_canvas():
    """Create a mock canvas for testing."""
    return MagicMock(spec=Canvas)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    width, height = 100, 100
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = int(255 * y / height)
            g = int(255 * x / width)
            b = int(128)
            array[y, x] = [r, g, b]
    
    return Image(array)


@pytest.fixture
def agent_manager(mock_canvas):
    """Create an agent manager for testing."""
    with patch('llama_canvas.core.agent_manager.ClaudeAgent') as mock_claude:
        manager = AgentManager(mock_canvas)
        yield manager


class TestAgent:
    """Test the base Agent class."""
    
    def test_init(self, mock_canvas):
        """Test initializing an Agent."""
        agent = Agent("test_agent", mock_canvas)
        
        assert agent.name == "test_agent"
        assert agent.canvas is mock_canvas
        assert agent.is_initialized is False
    
    def test_initialize(self, mock_canvas):
        """Test initializing the agent."""
        agent = Agent("test_agent", mock_canvas)
        
        # Base agent init should set is_initialized to True
        agent.initialize()
        assert agent.is_initialized is True
    
    def test_generate_image_not_implemented(self, mock_canvas):
        """Test that generate_image raises NotImplementedError."""
        agent = Agent("test_agent", mock_canvas)
        
        with pytest.raises(NotImplementedError):
            agent.generate_image("test prompt")
    
    def test_style_image_not_implemented(self, mock_canvas, sample_image):
        """Test that style_image raises NotImplementedError."""
        agent = Agent("test_agent", mock_canvas)
        
        with pytest.raises(NotImplementedError):
            agent.style_image(sample_image, "test style")


class TestClaudeAgent:
    """Test the ClaudeAgent class."""
    
    @patch('llama_canvas.core.agent_manager.anthropic')
    def test_init(self, mock_anthropic, mock_canvas):
        """Test initializing a ClaudeAgent."""
        # Set environment variable for testing
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test_key"}):
            agent = ClaudeAgent(mock_canvas)
            
            assert agent.name == "claude"
            assert agent.canvas is mock_canvas
            assert agent.model == "claude-3-opus-20240229"  # Default model
    
    @patch('llama_canvas.core.agent_manager.anthropic')
    def test_initialize(self, mock_anthropic, mock_canvas):
        """Test initializing the Claude agent."""
        # Mock the anthropic client
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Set environment variable for testing
        with patch.dict(os.environ, {"CLAUDE_API_KEY": "test_key"}):
            agent = ClaudeAgent(mock_canvas)
            agent.initialize()
            
            assert agent.is_initialized is True
            assert agent.client is mock_client
            mock_anthropic.Anthropic.assert_called_once_with(api_key="test_key")
    
    @patch('llama_canvas.core.agent_manager.anthropic')
    def test_initialize_missing_api_key(self, mock_anthropic, mock_canvas):
        """Test initializing without API key."""
        # Ensure no environment variable
        with patch.dict(os.environ, {}, clear=True):
            agent = ClaudeAgent(mock_canvas)
            
            # Should raise ValueError due to missing API key
            with pytest.raises(ValueError):
                agent.initialize()
    
    @patch('llama_canvas.core.agent_manager.anthropic')
    def test_generate_image(self, mock_anthropic, mock_canvas, sample_image):
        """Test generating an image with Claude."""
        # Mock the anthropic client and response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="image", source=MagicMock(media_type="image/png", data=b"test_image_data"))]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock PIL.Image.open to return a sample image
        with patch('PIL.Image.open', return_value=sample_image._image), \
             patch.dict(os.environ, {"CLAUDE_API_KEY": "test_key"}), \
             patch('io.BytesIO'):
            
            agent = ClaudeAgent(mock_canvas)
            agent.initialize()
            
            result = agent.generate_image("Create a sunset over mountains")
            
            # Verify correct API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]
            assert call_args["model"] == "claude-3-opus-20240229"
            assert "Create a sunset over mountains" in call_args["messages"][0]["content"]
            
            # Verify result is an Image
            assert isinstance(result, Image)
    
    @patch('llama_canvas.core.agent_manager.anthropic')
    def test_style_image(self, mock_anthropic, mock_canvas, sample_image):
        """Test styling an image with Claude."""
        # Mock the anthropic client and response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(type="image", source=MagicMock(media_type="image/png", data=b"test_image_data"))]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock PIL.Image.open to return a sample image
        with patch('PIL.Image.open', return_value=sample_image._image), \
             patch.dict(os.environ, {"CLAUDE_API_KEY": "test_key"}), \
             patch('io.BytesIO'):
            
            agent = ClaudeAgent(mock_canvas)
            agent.initialize()
            
            result = agent.style_image(sample_image, "Van Gogh", strength=0.8)
            
            # Verify correct API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]
            assert call_args["model"] == "claude-3-opus-20240229"
            assert "Van Gogh" in str(call_args["messages"][0]["content"])
            
            # Verify result is an Image
            assert isinstance(result, Image)


class TestAgentManager:
    """Test the AgentManager class."""
    
    def test_init(self, mock_canvas):
        """Test initializing the AgentManager."""
        with patch('llama_canvas.core.agent_manager.ClaudeAgent') as mock_claude:
            manager = AgentManager(mock_canvas)
            
            assert manager.canvas is mock_canvas
            assert len(manager.agents) > 0
            assert "claude" in manager.agents
            assert manager.default_agent == "claude"
    
    def test_get_agent(self, agent_manager):
        """Test getting an agent by name."""
        # Mock agent
        mock_agent = MagicMock()
        agent_manager.agents["test_agent"] = mock_agent
        
        # Get existing agent
        agent = agent_manager.get_agent("test_agent")
        assert agent is mock_agent
        
        # Get non-existent agent
        with pytest.raises(ValueError):
            agent_manager.get_agent("nonexistent_agent")
    
    def test_generate_image(self, agent_manager, sample_image):
        """Test generating an image through the manager."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.generate_image.return_value = sample_image
        agent_manager.agents["test_agent"] = mock_agent
        
        # Generate with specific agent
        result = agent_manager.generate_image("test prompt", model="test_agent")
        
        # Verify agent called correctly
        mock_agent.generate_image.assert_called_once_with("test prompt")
        assert result is sample_image
    
    def test_apply_style(self, agent_manager, sample_image):
        """Test applying a style through the manager."""
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.style_image.return_value = sample_image
        agent_manager.agents["test_agent"] = mock_agent
        
        # Apply style with specific agent
        result = agent_manager.apply_style(sample_image, "test style", agent="test_agent")
        
        # Verify agent called correctly
        mock_agent.style_image.assert_called_once_with(sample_image, "test style", strength=0.8)
        assert result is sample_image
    
    def test_enhance_resolution(self, agent_manager, sample_image):
        """Test enhancing resolution through the manager."""
        # Create a larger version of the sample image
        larger_image = sample_image.resize(2)
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.enhance_resolution.return_value = larger_image
        agent_manager.agents["test_agent"] = mock_agent
        
        # Enhance with specific agent
        result = agent_manager.enhance_resolution(sample_image, scale=2, agent="test_agent")
        
        # Verify agent called correctly
        mock_agent.enhance_resolution.assert_called_once_with(sample_image, scale=2)
        assert result is larger_image
    
    def test_use_default_agent(self, agent_manager, sample_image):
        """Test using the default agent when none specified."""
        # Mock the default claude agent
        mock_claude = MagicMock()
        mock_claude.generate_image.return_value = sample_image
        agent_manager.agents["claude"] = mock_claude
        
        # Generate without specifying agent
        result = agent_manager.generate_image("test prompt")
        
        # Verify default agent called
        mock_claude.generate_image.assert_called_once_with("test prompt")
        assert result is sample_image
    
    def test_register_agent(self, agent_manager):
        """Test registering a new agent."""
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.name = "new_agent"
        
        # Register the agent
        agent_manager.register_agent(mock_agent)
        
        # Verify agent was added
        assert "new_agent" in agent_manager.agents
        assert agent_manager.agents["new_agent"] is mock_agent
    
    def test_set_default_agent(self, agent_manager):
        """Test setting the default agent."""
        # Mock agent
        mock_agent = MagicMock()
        agent_manager.agents["test_agent"] = mock_agent
        
        # Set as default
        agent_manager.set_default_agent("test_agent")
        assert agent_manager.default_agent == "test_agent"
        
        # Test with non-existent agent
        with pytest.raises(ValueError):
            agent_manager.set_default_agent("nonexistent_agent")


if __name__ == "__main__":
    pytest.main() 