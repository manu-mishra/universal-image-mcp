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

REPO_URL = "https://github.com/manu-mishra/smart-Image-mcp"

mcp = FastMCP(
    "smart-image-mcp",
    instructions=f"""Multi-provider image generation server supporting AWS Bedrock (Nova Canvas), OpenAI/ChatGPT (GPT Image), and Google Gemini (Nano Banana, Imagen).

Use list_models() to see available models, generate_image() to create images, transform_image() to edit existing images, and prompt_guide() for tips on writing effective prompts.

⭐ Star the repo: {REPO_URL}
🐛 Report issues: {REPO_URL}/issues"""
)

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

## The Formula
**Subject + Environment + Style + Lighting + Mood + Technical Details**

## Building Blocks

### Subject (WHO/WHAT)
❌ "a woman" → ✅ "a 30-year-old Japanese woman with shoulder-length black hair, wearing a vintage 1970s denim jacket, confident expression"

❌ "a car" → ✅ "a weathered 1967 Ford Mustang Fastback in faded cherry red, chrome details catching light, slight rust on wheel wells"

### Environment (WHERE)
❌ "in a city" → ✅ "on a rain-slicked Tokyo street at 2am, neon signs reflecting in puddles, steam rising from a ramen stall, few pedestrians with umbrellas"

❌ "in nature" → ✅ "in an ancient redwood forest, morning fog weaving between massive trunks, ferns covering the forest floor, single beam of golden light breaking through canopy"

### Style References
- **Photographic**: "shot on Hasselblad medium format, Kodak Portra 400 film, shallow depth of field f/1.4"
- **Cinematic**: "cinematography by Roger Deakins, anamorphic lens flare, teal and orange color grade"
- **Artistic**: "in the style of Studio Ghibli, soft watercolor backgrounds, whimsical details"
- **Digital Art**: "trending on ArtStation, octane render, volumetric lighting, 8K detail"

### Lighting
- "Rembrandt lighting with dramatic shadows on one side of face"
- "backlit silhouette against golden hour sun, lens flare"
- "soft overcast diffused light, no harsh shadows"
- "three-point studio lighting, clean commercial look"
- "bioluminescent glow emanating from within, dark surroundings"

### Mood/Atmosphere
- "melancholic and introspective, muted desaturated colors"
- "electric and energetic, high contrast, vivid saturated hues"
- "dreamlike and ethereal, soft focus edges, pastel palette"
- "gritty and raw, high grain, documentary feel"

## Sophisticated Examples

**Editorial Portrait**:
"Close-up portrait of a weathered Icelandic fisherman in his 60s, deep-set blue eyes telling stories of the sea, salt-and-pepper beard with ice crystals, wearing a thick wool sweater, standing on a fishing boat deck, overcast North Atlantic light, shot on medium format film, shallow depth of field, muted Nordic color palette"

**Architectural**:
"Interior of a brutalist concrete library at golden hour, dramatic shafts of warm light streaming through geometric skylights, dust particles floating in light beams, a single person reading in a leather chair, long shadows across polished concrete floor, shot with tilt-shift lens, architectural photography style"

**Product/Commercial**:
"Luxury perfume bottle on black obsidian surface, bottle is hand-blown glass with amber liquid, single orchid petal beside it, dramatic side lighting creating long shadow, water droplets on surface suggesting freshness, reflection visible in glossy surface, high-end cosmetics advertising style, 8K product photography"

**Concept Art**:
"Massive ancient tree city built into a 1000-year-old baobab, multiple levels connected by rope bridges and spiral staircases carved into bark, warm lantern light glowing from windows, tiny figures going about evening activities, flying creatures circling the canopy, painted in the style of Craig Mullins, epic fantasy concept art, atmospheric perspective showing scale"

**Street Photography**:
"Candid moment in a Havana street, elderly Cuban man playing dominos with friends at a weathered wooden table, cigar smoke curling upward, crumbling pastel colonial building behind them, vintage American car passing in background, harsh midday Caribbean sun creating strong shadows, Leica street photography aesthetic, authentic documentary feel"

**Surreal/Abstract**:
"A grand piano melting like Dalí's clocks, dripping onto an infinite mirror floor that reflects an impossible sky filled with floating musical notes that transform into birds, the scene transitions from warm sepia tones on the left to cool cyan on the right, hyperrealistic rendering of surreal concept, inspired by Magritte and Dalí"

## Pro Tips
1. **Layer details**: Start broad, then add specific details that make it unique
2. **Use sensory language**: Textures, temperatures, sounds implied visually
3. **Reference real cameras/film**: "shot on 35mm Kodak Ektar" gives specific look
4. **Name specific artists**: "lighting like Gregory Crewdson" is more precise than "dramatic"
5. **Include imperfections**: "slight motion blur", "film grain", "lens dust" add realism
6. **Specify what to avoid**: "no text, no watermarks, no extra limbs"
7. **Consider negative space**: "minimalist composition with breathing room"

---
⭐ Found this helpful? Star the repo: https://github.com/manu-mishra/smart-Image-mcp
🐛 Issues or feature requests: https://github.com/manu-mishra/smart-Image-mcp/issues
"""


def main():
    mcp.run()

if __name__ == "__main__":
    main()
