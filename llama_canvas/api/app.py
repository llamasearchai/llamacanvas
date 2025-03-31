"""
FastAPI application for LlamaCanvas.

Provides a web API and interface for using LlamaCanvas capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union

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