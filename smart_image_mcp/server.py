import os
import logging
from io import BytesIO
from datetime import datetime
import PIL.Image
from mcp.server.fastmcp import FastMCP
from .providers import get_aws_models, get_openai_models, get_gemini_models
from .providers import AWSProvider, OpenAIProvider, GeminiProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("smart-image-mcp")

def is_enabled(provider: str) -> bool:
    return os.getenv(f"ENABLE_{provider.upper()}", "false").lower() == "true"

def get_provider(model_id: str):
    """Get provider instance for the given model_id"""
    if model_id.startswith("amazon.") or model_id.startswith("stability."):
        if not is_enabled("aws"):
            raise ValueError("AWS provider not enabled")
        return AWSProvider(model_id)
    elif "gpt" in model_id.lower() or "dall" in model_id.lower():
        if not is_enabled("openai"):
            raise ValueError("OpenAI provider not enabled")
        return OpenAIProvider(model_id)
    elif "gemini" in model_id.lower():
        if not is_enabled("gemini"):
            raise ValueError("Gemini provider not enabled")
        return GeminiProvider(model_id)
    raise ValueError(f"Unknown model: {model_id}")

@mcp.tool()
def list_models() -> str:
    """List available image generation models from enabled providers"""
    results = []
    
    if is_enabled("aws"):
        try:
            results.append("AWS Bedrock:")
            for m in get_aws_models():
                results.append(f"  {m['id']}")
                results.append(f"    Name: {m['name']} | Provider: {m['provider']} | Status: {m['status']}")
                results.append(f"    Input: {', '.join(m['input'])}")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    if is_enabled("openai"):
        try:
            results.append("OpenAI:")
            for m in get_openai_models():
                created = datetime.fromtimestamp(m['created']).strftime('%Y-%m-%d')
                results.append(f"  {m['id']}")
                results.append(f"    Released: {created} | Owner: {m['owned_by']}")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    if is_enabled("gemini"):
        try:
            results.append("Google Gemini:")
            for m in get_gemini_models():
                results.append(f"  {m['id']}")
                results.append(f"    Name: {m['name']}")
                if m.get('description'):
                    results.append(f"    {m['description'][:80]}...")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    return "\n".join(results) if results else "No providers enabled"

@mcp.tool()
def generate_image(prompt: str, model_id: str, output_path: str, reference_image: str = None, width: int = 1024, height: int = 1024) -> str:
    """Generate an image from text prompt
    
    Args:
        prompt: Text description of the image to generate
        model_id: Model ID from list_models()
        output_path: Path to save the generated image
        reference_image: Optional path to reference image for style/content guidance
        width: Image width (default 1024)
        height: Image height (default 1024)
    """
    try:
        provider = get_provider(model_id)
        
        ref_img = None
        if reference_image:
            reference_image = os.path.expanduser(reference_image)
            if not os.path.exists(reference_image):
                return f"Error: Reference image not found at {reference_image}"
            ref_img = PIL.Image.open(reference_image)
        
        image_data = provider.generate(prompt, ref_img, width, height)
        
        image = PIL.Image.open(BytesIO(image_data))
        output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        
        return f"Image saved to {output_path}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {e}"

@mcp.tool()
def transform_image(image_path: str, prompt: str, model_id: str, output_path: str) -> str:
    """Transform an existing image based on text prompt
    
    Args:
        image_path: Path to source image
        prompt: Text description of transformation
        model_id: Model ID from list_models()
        output_path: Path to save the transformed image
    """
    try:
        image_path = os.path.expanduser(image_path)
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"
        
        source = PIL.Image.open(image_path)
        provider = get_provider(model_id)
        image_data = provider.transform(source, prompt)
        
        image = PIL.Image.open(BytesIO(image_data))
        output_path = os.path.expanduser(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        image.save(output_path)
        
        return f"Image saved to {output_path}"
    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {e}"

def main():
    logger.info("Starting Smart Image MCP server...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
