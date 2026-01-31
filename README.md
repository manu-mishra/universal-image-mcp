# Smart Image MCP

A generic MCP server for image generation supporting multiple providers: AWS Bedrock, OpenAI, and Google Gemini.

## Features

- **Multi-provider support**: AWS Bedrock, OpenAI, Google Gemini
- **Dynamic model discovery**: Models fetched from provider APIs with deprecated models filtered
- **Lazy initialization**: Provider clients only created when needed
- **Reference image support**: Generate new images based on existing images
- **Configurable dimensions**: Custom width/height for supported models
- **Prompt guide**: Built-in best practices for writing effective prompts

## Installation

```bash
pip install smart-image-mcp
```

## MCP Configuration

Add to your MCP config (e.g., `~/.kiro/settings/mcp.json` or Claude Desktop config):

```json
{
  "mcpServers": {
    "smart-image-mcp": {
      "command": "uvx",
      "args": ["smart-image-mcp@latest"],
      "env": {
        "ENABLE_AWS": "true",
        "AWS_PROFILE": "default",
        "AWS_REGION": "us-east-1",
        
        "ENABLE_OPENAI": "true",
        "OPENAI_API_KEY": "sk-...",
        
        "ENABLE_GEMINI": "true",
        "GEMINI_API_KEY": "..."
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ENABLE_AWS` | No | Enable AWS Bedrock provider (`true`/`false`, default: `false`) |
| `AWS_PROFILE` | No | AWS profile name (default, SSO, or named profile) |
| `AWS_REGION` | No | AWS region (default: `us-east-1`) |
| `ENABLE_OPENAI` | No | Enable OpenAI provider (`true`/`false`, default: `false`) |
| `OPENAI_API_KEY` | If OpenAI enabled | OpenAI API key |
| `ENABLE_GEMINI` | No | Enable Google Gemini provider (`true`/`false`, default: `false`) |
| `GEMINI_API_KEY` | If Gemini enabled | Google Gemini API key |

## Tools

### list_models()

List available image generation models from all enabled providers. Models are fetched dynamically from each provider's API with older/deprecated models filtered out.

### generate_image(prompt, model_id, output_path, reference_image?, width?, height?)

Generate an image from a text prompt.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `prompt` | Yes | - | Detailed text description of the image. Be specific about subject, style, lighting, colors, composition, and mood. |
| `model_id` | Yes | - | Model ID from `list_models()`. Examples: `amazon.nova-canvas-v1:0`, `gpt-image-1.5`, `models/gemini-2.5-flash-image` |
| `output_path` | Yes | - | File path to save the generated image. Parent directories created automatically. |
| `reference_image` | No | None | Path to reference image for style/content guidance |
| `width` | No | 1024 | Image width in pixels. Note: Some models only support specific sizes. |
| `height` | No | 1024 | Image height in pixels. Note: Some models only support specific sizes. |

### transform_image(image_path, prompt, model_id, output_path)

Transform an existing image based on a text prompt.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `image_path` | Yes | Path to source image to transform |
| `prompt` | Yes | Text description of transformation (e.g., "Make it black and white", "Add a rainbow") |
| `model_id` | Yes | Model ID from `list_models()` |
| `output_path` | Yes | File path to save the transformed image |

### prompt_guide()

Get best practices and examples for writing effective image generation prompts. Returns guidelines covering:
- Prompt structure (Subject + Details + Style + Lighting + Mood + Composition)
- Specific vs generic descriptions
- Style, lighting, and mood keywords
- Example prompts for different use cases

## Supported Models

Models are discovered dynamically. Use `list_models()` to see current options.

### AWS Bedrock
- Amazon Nova Canvas (`amazon.nova-canvas-v1:0`)

### OpenAI
- GPT Image 1.5 (`gpt-image-1.5`)
- ChatGPT Image Latest (`chatgpt-image-latest`)

### Google Gemini
- Gemini 2.5 Flash Image (`models/gemini-2.5-flash-image`)
- Gemini 3 Pro Image (`models/gemini-3-pro-image-preview`)
- Imagen 4 (`models/imagen-4.0-generate-001`)

## License

MIT
