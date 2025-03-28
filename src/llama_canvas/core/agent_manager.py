"""
Agent Manager for LlamaCanvas.

This module provides the AgentManager class and base Agent implementations
for handling tasks using different AI models and services.
"""

import abc
from typing import Any, Dict, List, Optional, Type, Union, Callable

from llama_canvas.core.image import Image
from llama_canvas.utils.logging import get_logger
from llama_canvas.utils.config import settings

logger = get_logger(__name__)


class Agent(abc.ABC):
    """
    Base class for all agents in the LlamaCanvas system.
    
    An agent is responsible for executing specific tasks using
    capabilities of an underlying AI model or service.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._capabilities = set()
        self._register_capabilities()
        
        logger.debug(f"Initialized agent: {name}")
    
    @abc.abstractmethod
    def _register_capabilities(self) -> None:
        """Register agent capabilities."""
        pass
    
    def has_capability(self, capability: str) -> bool:
        """
        Check if the agent has a specific capability.
        
        Args:
            capability: Capability to check
            
        Returns:
            True if the agent has the capability
        """
        return capability in self._capabilities
    
    def get_capabilities(self) -> List[str]:
        """
        Get all capabilities of the agent.
        
        Returns:
            List of capability strings
        """
        return list(self._capabilities)
    
    @abc.abstractmethod
    def execute(self, task: str, **kwargs) -> Any:
        """
        Execute a task.
        
        Args:
            task: Task name
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        pass


class ClaudeAgent(Agent):
    """
    Agent that uses Claude API for specific tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a Claude agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("claude", config)
        
        # Initialize Claude client if API key is available
        self.client = None
        api_key = self.config.get("api_key") or settings.get("claude_api_key")
        
        if api_key:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
                logger.debug("Claude client initialized")
            except ImportError:
                logger.warning("anthropic package not installed. Install with 'pip install anthropic'")
        else:
            logger.warning("Claude API key not found. Set CLAUDE_API_KEY env var or configure in settings.")
    
    def _register_capabilities(self) -> None:
        """Register Claude agent capabilities."""
        self._capabilities.update([
            "text_generation",
            "image_captioning",
            "image_description",
            "code_generation",
            "visual_qa"
        ])
        
        # Add vision capabilities if Claude client supports it
        if self.client:
            self._capabilities.add("image_generation")
            self._capabilities.add("image_editing")
    
    def execute(self, task: str, **kwargs) -> Any:
        """
        Execute a task using Claude.
        
        Args:
            task: Task name
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        if not self.client:
            logger.error("Claude client not initialized")
            return None
        
        if task == "generate_text":
            return self._generate_text(**kwargs)
        elif task == "generate_image":
            return self._generate_image(**kwargs)
        elif task == "describe_image":
            return self._describe_image(**kwargs)
        elif task == "edit_image":
            return self._edit_image(**kwargs)
        else:
            logger.warning(f"Unknown task: {task}")
            return None
    
    def _generate_text(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """
        Generate text using Claude.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        logger.debug(f"Generating text with Claude, prompt length: {len(prompt)}")
        
        try:
            response = self.client.messages.create(
                model=kwargs.get("model", "claude-3-opus-20240229"),
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Claude: {e}")
            return ""
    
    def _generate_image(self, prompt: str, **kwargs) -> Optional[Image]:
        """
        Generate an image using Claude vision.
        
        Args:
            prompt: Text prompt describing the image
            **kwargs: Additional parameters
            
        Returns:
            Generated image or None if generation failed
        """
        logger.debug(f"Generating image with Claude, prompt: {prompt}")
        
        try:
            # Note: Claude doesn't have native image generation yet
            # This is a placeholder for future functionality
            logger.warning("Claude image generation not supported yet")
            
            # Generate placeholder image for now
            import numpy as np
            from PIL import Image as PILImage, ImageDraw, ImageFont
            
            width = kwargs.get("width", 512)
            height = kwargs.get("height", 512)
            
            # Create a gradient background
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    r = int(255 * (x / width))
                    g = int(255 * (y / height))
                    b = int(128)
                    arr[y, x] = [r, g, b]
            
            img = PILImage.fromarray(arr)
            draw = ImageDraw.Draw(img)
            
            # Try to add text
            try:
                font = ImageFont.load_default()
                text = f"Claude: {prompt[:50]}..."
                draw.text((10, 10), text, fill=(255, 255, 255))
            except Exception:
                pass
                
            return Image(img)
            
        except Exception as e:
            logger.error(f"Error generating image with Claude: {e}")
            return None
    
    def _describe_image(self, image: Image, **kwargs) -> str:
        """
        Describe an image using Claude vision.
        
        Args:
            image: Image to describe
            **kwargs: Additional parameters
            
        Returns:
            Image description
        """
        logger.debug("Describing image with Claude")
        
        try:
            import base64
            from io import BytesIO
            
            # Convert image to bytes
            buffer = BytesIO()
            image.image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Create message with image
            response = self.client.messages.create(
                model=kwargs.get("model", "claude-3-opus-20240229"),
                max_tokens=kwargs.get("max_tokens", 500),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in detail."
                            }
                        ]
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error describing image with Claude: {e}")
            return "Failed to describe image."
    
    def _edit_image(self, image: Image, prompt: str, **kwargs) -> Optional[Image]:
        """
        Edit an image using Claude.
        
        Args:
            image: Image to edit
            prompt: Text prompt describing the edit
            **kwargs: Additional parameters
            
        Returns:
            Edited image or None if editing failed
        """
        logger.debug(f"Editing image with Claude, prompt: {prompt}")
        
        # Claude doesn't directly support image editing yet
        logger.warning("Claude image editing not fully supported yet")
        
        # For now, return the original image
        return image


class StableDiffusionAgent(Agent):
    """
    Agent that uses Stable Diffusion for image generation and editing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a Stable Diffusion agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("stable_diffusion", config)
        
        # Initialize Stable Diffusion
        self.model = None
        self.pipe = None
        
        # Try to load the model if diffusers is installed
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            model_id = self.config.get("model_id", "stabilityai/stable-diffusion-2-1")
            
            # Only initialize if explicitly enabled
            if self.config.get("initialize", False):
                logger.info(f"Loading Stable Diffusion model: {model_id}")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                    
                logger.debug("Stable Diffusion model loaded")
            else:
                logger.debug("Stable Diffusion initialization deferred (set initialize=True to load at startup)")
                
        except ImportError:
            logger.warning("diffusers package not installed. Install with 'pip install diffusers torch'")
    
    def _register_capabilities(self) -> None:
        """Register Stable Diffusion agent capabilities."""
        self._capabilities.update([
            "image_generation",
            "image_to_image",
            "inpainting",
        ])
    
    def _ensure_model_loaded(self, model_id: Optional[str] = None) -> bool:
        """
        Ensure the Stable Diffusion model is loaded.
        
        Args:
            model_id: Optional model ID to load
            
        Returns:
            True if model is loaded
        """
        if self.pipe is not None:
            return True
            
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            model_id = model_id or self.config.get("model_id", "stabilityai/stable-diffusion-2-1")
            
            logger.info(f"Loading Stable Diffusion model: {model_id}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                
            logger.debug("Stable Diffusion model loaded")
            return True
                
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion model: {e}")
            return False
    
    def execute(self, task: str, **kwargs) -> Any:
        """
        Execute a task using Stable Diffusion.
        
        Args:
            task: Task name
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        if task == "generate_image":
            return self._generate_image(**kwargs)
        elif task == "image_to_image":
            return self._image_to_image(**kwargs)
        elif task == "inpaint":
            return self._inpaint(**kwargs)
        else:
            logger.warning(f"Unknown task: {task}")
            return None
    
    def _generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "", 
        width: int = 512, 
        height: int = 512,
        **kwargs
    ) -> Optional[Image]:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            **kwargs: Additional parameters
            
        Returns:
            Generated image or None if generation failed
        """
        logger.debug(f"Generating image with Stable Diffusion, prompt: {prompt}")
        
        # If diffusers not available, create a placeholder
        if not self._ensure_model_loaded(kwargs.get("model_id")):
            logger.warning("Stable Diffusion not available, creating placeholder image")
            
            # Create a placeholder image
            import numpy as np
            from PIL import Image as PILImage, ImageDraw, ImageFont
            
            # Create a gradient background
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    r = int(200 * (x / width))
                    g = int(100 * (y / height))
                    b = int(200 * (1 - y / height))
                    arr[y, x] = [r, g, b]
            
            img = PILImage.fromarray(arr)
            draw = ImageDraw.Draw(img)
            
            # Try to add text
            try:
                font = ImageFont.load_default()
                text = f"SD: {prompt[:50]}..."
                draw.text((10, 10), text, fill=(255, 255, 255))
            except Exception:
                pass
                
            return Image(img)
        
        try:
            # Generate image
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=kwargs.get("steps", 30),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
            )
            
            # Check for safety issues
            if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected[0]:
                logger.warning("NSFW content detected, image generation filtered")
                return None
                
            # Convert to our Image type
            return Image(result.images[0])
            
        except Exception as e:
            logger.error(f"Error generating image with Stable Diffusion: {e}")
            return None
    
    def _image_to_image(
        self,
        image: Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.8,
        **kwargs
    ) -> Optional[Image]:
        """
        Generate an image based on another image.
        
        Args:
            image: Source image
            prompt: Text prompt
            negative_prompt: Negative prompt
            strength: Transformation strength (0.0 to 1.0)
            **kwargs: Additional parameters
            
        Returns:
            Generated image or None if generation failed
        """
        logger.debug(f"Performing image-to-image with Stable Diffusion, prompt: {prompt}")
        
        # For placeholder implementation
        if not self._ensure_model_loaded(kwargs.get("model_id")):
            logger.warning("Stable Diffusion not available for image-to-image, returning filtered image")
            return image.apply_filter("contour")
        
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            import torch
            
            # Get model ID
            model_id = kwargs.get("model_id", self.config.get("model_id", "stabilityai/stable-diffusion-2-1"))
            
            # Initialize img2img pipeline if needed
            if not hasattr(self, "img2img_pipe") or self.img2img_pipe is None:
                logger.debug(f"Loading Stable Diffusion Img2Img pipeline: {model_id}")
                self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                if torch.cuda.is_available():
                    self.img2img_pipe = self.img2img_pipe.to("cuda")
            
            # Generate image
            result = self.img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image.image,
                strength=strength,
                num_inference_steps=kwargs.get("steps", 30),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
            )
            
            # Convert to our Image type
            return Image(result.images[0])
            
        except Exception as e:
            logger.error(f"Error performing image-to-image: {e}")
            return image
    
    def _inpaint(
        self,
        image: Image,
        mask: Image,
        prompt: str,
        negative_prompt: str = "",
        **kwargs
    ) -> Optional[Image]:
        """
        Inpaint parts of an image.
        
        Args:
            image: Source image
            mask: Mask image (white=inpaint, black=keep)
            prompt: Text prompt
            negative_prompt: Negative prompt
            **kwargs: Additional parameters
            
        Returns:
            Inpainted image or None if inpainting failed
        """
        logger.debug(f"Inpainting with Stable Diffusion, prompt: {prompt}")
        
        # For placeholder implementation
        if not self._ensure_model_loaded(kwargs.get("model_id")):
            logger.warning("Stable Diffusion not available for inpainting, returning original image")
            return image
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch
            
            # Get model ID
            model_id = kwargs.get("model_id", self.config.get("model_id", "stabilityai/stable-diffusion-2-inpainting"))
            
            # Initialize inpaint pipeline if needed
            if not hasattr(self, "inpaint_pipe") or self.inpaint_pipe is None:
                logger.debug(f"Loading Stable Diffusion Inpaint pipeline: {model_id}")
                self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                if torch.cuda.is_available():
                    self.inpaint_pipe = self.inpaint_pipe.to("cuda")
            
            # Convert mask to proper format
            mask_image = mask.image.convert("L")
            
            # Generate image
            result = self.inpaint_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image.image,
                mask_image=mask_image,
                num_inference_steps=kwargs.get("steps", 30),
                guidance_scale=kwargs.get("guidance_scale", 7.5),
            )
            
            # Convert to our Image type
            return Image(result.images[0])
            
        except Exception as e:
            logger.error(f"Error performing inpainting: {e}")
            return image


class AgentManager:
    """
    Manages AI agents within the LlamaCanvas ecosystem.
    
    The AgentManager coordinates tasks and selects appropriate agents
    based on their capabilities.
    """
    
    def __init__(self, canvas=None):
        """
        Initialize the AgentManager.
        
        Args:
            canvas: Canvas instance for context
        """
        self.canvas = canvas
        self.agents = {}
        self._registered_agent_classes = {}
        
        # Register built-in agents
        self._register_agent_class("claude", ClaudeAgent)
        self._register_agent_class("stable_diffusion", StableDiffusionAgent)
        
        # Initialize default agents
        self._init_default_agents()
        
        logger.debug(f"AgentManager initialized with {len(self.agents)} agents")
    
    def _register_agent_class(self, agent_type: str, agent_class: Type[Agent]) -> None:
        """
        Register an agent class.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class
        """
        self._registered_agent_classes[agent_type] = agent_class
        logger.debug(f"Registered agent class: {agent_type}")
    
    def _init_default_agents(self) -> None:
        """Initialize default agents based on configuration."""
        # Initialize Claude agent if API key is available
        if settings.get("claude_api_key"):
            self.register_agent("claude", ClaudeAgent({"api_key": settings.get("claude_api_key")}))
        
        # Initialize Stable Diffusion with deferred loading
        self.register_agent("stable_diffusion", StableDiffusionAgent({"initialize": False}))
    
    def register_agent(self, name: str, agent: Agent) -> None:
        """
        Register an agent.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self.agents[name] = agent
        logger.debug(f"Registered agent: {name}")
    
    def unregister_agent(self, name: str) -> None:
        """
        Unregister an agent.
        
        Args:
            name: Agent name
        """
        if name in self.agents:
            del self.agents[name]
            logger.debug(f"Unregistered agent: {name}")
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """
        Get an agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name)
    
    def get_agents_with_capability(self, capability: str) -> List[Agent]:
        """
        Get all agents with a specific capability.
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List of agents with the capability
        """
        return [
            agent for agent in self.agents.values()
            if agent.has_capability(capability)
        ]
    
    def create_agent(self, agent_type: str, config: Optional[Dict[str, Any]] = None) -> Optional[Agent]:
        """
        Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            
        Returns:
            New agent instance or None if agent type is not registered
        """
        if agent_type not in self._registered_agent_classes:
            logger.error(f"Unknown agent type: {agent_type}")
            return None
            
        agent_class = self._registered_agent_classes[agent_type]
        return agent_class(config)
    
    def execute_task(
        self, 
        task: str, 
        agent_name: Optional[str] = None, 
        capability: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute a task using an agent.
        
        Args:
            task: Task name
            agent_name: Name of agent to use (optional)
            capability: Capability required for the task (used if agent_name not specified)
            **kwargs: Task parameters
            
        Returns:
            Task result
        """
        if agent_name:
            # Use specific agent
            agent = self.get_agent(agent_name)
            if not agent:
                logger.error(f"Agent not found: {agent_name}")
                return None
                
            return agent.execute(task, **kwargs)
            
        elif capability:
            # Find agents with capability
            capable_agents = self.get_agents_with_capability(capability)
            if not capable_agents:
                logger.error(f"No agents with capability: {capability}")
                return None
                
            # Use first available agent
            return capable_agents[0].execute(task, **kwargs)
            
        else:
            logger.error("Either agent_name or capability must be specified")
            return None
    
    def generate_image(self, prompt: str, model: str = "stable-diffusion", **kwargs) -> Optional[Image]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt
            model: Model identifier (stable-diffusion, claude, etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated image or None if generation failed
        """
        logger.debug(f"Generating image with model: {model}")
        
        if model.startswith("stable-diffusion"):
            agent = self.get_agent("stable_diffusion")
            if not agent:
                logger.warning("Stable Diffusion agent not available")
                # Fallback to Claude if available
                agent = self.get_agent("claude")
                if agent:
                    return agent.execute("generate_image", prompt=prompt, **kwargs)
                return None
                
            return agent.execute("generate_image", prompt=prompt, **kwargs)
            
        elif model.startswith("claude"):
            agent = self.get_agent("claude")
            if not agent:
                logger.warning("Claude agent not available")
                # Fallback to Stable Diffusion
                agent = self.get_agent("stable_diffusion")
                if agent:
                    return agent.execute("generate_image", prompt=prompt, **kwargs)
                return None
                
            return agent.execute("generate_image", prompt=prompt, **kwargs)
            
        else:
            # Try to find any agent with image_generation capability
            return self.execute_task("generate_image", capability="image_generation", prompt=prompt, **kwargs)
    
    def enhance_image(self, image: Image, method: str = "super_resolution", **kwargs) -> Image:
        """
        Enhance an image using AI models.
        
        Args:
            image: Image to enhance
            method: Enhancement method
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image
        """
        logger.debug(f"Enhancing image with method: {method}")
        
        if method == "super_resolution":
            # Try stable diffusion first
            agent = self.get_agent("stable_diffusion")
            if agent:
                # Use img2img with the same image as input
                result = agent.execute(
                    "image_to_image",
                    image=image,
                    prompt="High resolution, detailed, sharp",
                    negative_prompt="blurry, low quality",
                    strength=0.4,
                    **kwargs
                )
                if result:
                    return result
            
            # Fallback to basic enhancement
            return image.resize((image.width * 2, image.height * 2)).apply_filter("sharpen")
            
        elif method == "enhance_details":
            # Try stable diffusion
            agent = self.get_agent("stable_diffusion")
            if agent:
                result = agent.execute(
                    "image_to_image",
                    image=image,
                    prompt="Highly detailed, precise, sharp focus",
                    strength=0.3,
                    **kwargs
                )
                if result:
                    return result
            
            # Fallback to basic enhancement
            return image.apply_filter("sharpen").adjust_contrast(1.2)
            
        else:
            logger.warning(f"Unknown enhancement method: {method}")
            return image
    
    def apply_style(self, image: Image, style: Union[str, Image], **kwargs) -> Image:
        """
        Apply a style to an image.
        
        Args:
            image: Source image
            style: Style name or reference image
            **kwargs: Additional parameters
            
        Returns:
            Styled image
        """
        logger.debug(f"Applying style: {style if isinstance(style, str) else 'reference image'}")
        
        # Use stable diffusion for style transfer
        agent = self.get_agent("stable_diffusion")
        if agent:
            prompt = f"Image in the style of {style}" if isinstance(style, str) else "Transfer the style of the reference image"
            
            result = agent.execute(
                "image_to_image",
                image=image,
                prompt=prompt,
                strength=kwargs.get("strength", 0.7),
                **kwargs
            )
            
            if result:
                return result
        
        # Fallback to basic filtering
        logger.warning("No style transfer agent available, applying basic filter")
        filtered = image.apply_filter(kwargs.get("filter", "contour"))
        return filtered 