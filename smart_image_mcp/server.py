"""Smart Image MCP Server"""
import os
import logging
from io import BytesIO
from datetime import datetime
from typing import Optional
import PIL.Image
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("smart-image-mcp")

def is_enabled(provider: str) -> bool:
    return os.getenv(f"ENABLE_{provider.upper()}", "false").lower() == "true"

def get_provider(model_id: str):
    """Get provider instance for the given model_id. Imports are lazy to avoid loading disabled providers."""
    from .providers import AWSProvider, OpenAIProvider, GeminiProvider
    
    if model_id.startswith("amazon.") or model_id.startswith("stability."):
        if not is_enabled("aws"):
            raise ValueError("AWS provider not enabled. Set ENABLE_AWS=true")
        return AWSProvider(model_id)
    elif "gpt" in model_id.lower() or "dall" in model_id.lower() or "chatgpt" in model_id.lower():
        if not is_enabled("openai"):
            raise ValueError("OpenAI provider not enabled. Set ENABLE_OPENAI=true")
        return OpenAIProvider(model_id)
    elif "gemini" in model_id.lower() or "imagen" in model_id.lower():
        if not is_enabled("gemini"):
            raise ValueError("Gemini provider not enabled. Set ENABLE_GEMINI=true")
        return GeminiProvider(model_id)
    raise ValueError(f"Unknown model: {model_id}. Use list_models() to see available models.")


@mcp.tool()
def list_models() -> str:
    """List available image generation models from all enabled providers.
    
    Returns a formatted list of model IDs that can be used with generate_image and transform_image.
    Models are fetched dynamically from each provider's API.
    """
    results = []
    
    if is_enabled("aws"):
        try:
            from .providers import get_aws_models
            results.append("AWS Bedrock:")
            for m in get_aws_models():
                results.append(f"  {m['id']}")
                results.append(f"    Name: {m['name']} | Provider: {m['provider']} | Status: {m['status']}")
                results.append(f"    Input: {', '.join(m['input'])}")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    if is_enabled("openai"):
        try:
            from .providers import get_openai_models
            results.append("OpenAI:")
            for m in get_openai_models():
                created = datetime.fromtimestamp(m['created']).strftime('%Y-%m-%d')
                results.append(f"  {m['id']}")
                results.append(f"    Released: {created} | Owner: {m['owned_by']}")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    if is_enabled("gemini"):
        try:
            from .providers import get_gemini_models
            results.append("Google Gemini:")
            for m in get_gemini_models():
                results.append(f"  {m['id']}")
                results.append(f"    Name: {m['name']}")
                if m.get('description'):
                    results.append(f"    {m['description'][:80]}...")
        except Exception as e:
            results.append(f"  Error: {e}")
    
    if not results:
        return "No providers enabled. Set ENABLE_AWS=true, ENABLE_OPENAI=true, or ENABLE_GEMINI=true"
    
    return "\n".join(results)


@mcp.tool()
def generate_image(
    prompt: str,
    model_id: str,
    output_path: str,
    reference_image: Optional[str] = None,
    width: Optional[int] = 1024,
    height: Optional[int] = 1024
) -> str:
    """Generate an image from a text prompt using the specified model.
    
    Args:
        prompt: Detailed text description of the image to generate. Be specific about subject, 
                style, lighting, colors, composition, and mood. Example: "A fluffy orange cat 
                sitting on a windowsill, golden hour lighting, watercolor style"
        model_id: Model identifier from list_models(). Examples: "amazon.nova-canvas-v1:0", 
                  "gpt-image-1.5", "models/gemini-2.5-flash-image"
        output_path: Absolute or relative file path where the generated image will be saved. 
                     Supports PNG, JPEG formats. Parent directories are created automatically.
        reference_image: Optional. Path to an existing image to use as style/content reference.
                         The model will generate a new image influenced by this reference.
        width: Optional. Image width in pixels. Default: 1024. Common values: 512, 768, 1024, 1280.
               Note: Some models only support specific sizes.
        height: Optional. Image height in pixels. Default: 1024. Common values: 512, 768, 1024, 1280.
                Note: Some models only support specific sizes.
    
    Returns:
        Success message with output path, or error description.
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
        logger.error(f"generate_image error: {e}")
        return f"Error: {e}"


@mcp.tool()
def transform_image(
    image_path: str,
    prompt: str,
    model_id: str,
    output_path: str
) -> str:
    """Transform an existing image based on a text prompt.
    
    Args:
        image_path: Path to the source image to transform. Supports common formats (PNG, JPEG, etc.)
        prompt: Text description of the desired transformation. Examples: "Make it black and white",
                "Add a rainbow in the sky", "Convert to watercolor painting style"
        model_id: Model identifier from list_models(). Examples: "amazon.nova-canvas-v1:0",
                  "gpt-image-1.5", "models/gemini-2.5-flash-image"
        output_path: File path where the transformed image will be saved.
                     Parent directories are created automatically.
    
    Returns:
        Success message with output path, or error description.
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
        logger.error(f"transform_image error: {e}")
        return f"Error: {e}"


@mcp.tool()
def prompt_guide() -> str:
    """Get best practices and examples for writing effective image generation prompts.
    
    Returns guidelines for crafting detailed prompts that produce better results.
    """
    return """# Image Prompt Best Practices

## Structure
A good prompt includes: Subject + Details + Style + Lighting + Mood + Composition

## Key Elements

**Subject**: Be specific, not generic
- ❌ "a cat"
- ✅ "a fluffy orange tabby cat with bright green eyes"

**Setting/Environment**: Ground your subject
- ❌ "a dog"
- ✅ "a golden retriever playing in a sunlit meadow with wildflowers"

**Style**: Specify artistic style
- "photorealistic", "watercolor painting", "oil painting", "digital art"
- "anime style", "pixel art", "3D render", "pencil sketch"

**Lighting**: Sets mood and atmosphere
- "golden hour lighting", "dramatic shadows", "soft diffused light"
- "neon-lit", "candlelight", "overcast day", "studio lighting"

**Mood/Atmosphere**: Emotional tone
- "serene", "mysterious", "vibrant", "melancholic", "whimsical"

**Composition**: Camera/viewing angle
- "close-up portrait", "wide angle", "bird's eye view", "macro shot"
- "centered composition", "rule of thirds", "symmetrical"

## Examples

**Portrait**:
"A wise elderly woman with silver hair and deep wrinkles, warm smile, 
wearing a hand-knitted sweater, soft window light, photorealistic portrait"

**Landscape**:
"Misty mountain valley at sunrise, pine forests, a winding river reflecting 
pink and orange sky, atmospheric perspective, landscape photography style"

**Product**:
"Minimalist ceramic coffee mug on marble surface, steam rising, 
soft shadows, clean white background, commercial product photography"

**Fantasy**:
"Ancient library with floating books and glowing orbs, dust particles 
in light beams, magical atmosphere, detailed digital art style"

## Tips
1. More detail = better results (but stay coherent)
2. Avoid contradictions ("dark bright room")
3. Put important elements first
4. Use commas to separate concepts
5. Reference specific artists/styles for consistent aesthetics
"""


def main():
    mcp.run()

if __name__ == "__main__":
    main()
