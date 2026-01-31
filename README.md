# Smart Image MCP

A generic MCP server for image generation supporting multiple providers: AWS Bedrock, OpenAI, and Google Gemini.

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

| Variable | Description |
|----------|-------------|
| `ENABLE_AWS` | Enable AWS Bedrock provider (`true`/`false`) |
| `AWS_PROFILE` | AWS profile name (default, SSO, or named profile) |
| `AWS_REGION` | AWS region (e.g., `us-east-1`) |
| `ENABLE_OPENAI` | Enable OpenAI provider (`true`/`false`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ENABLE_GEMINI` | Enable Google Gemini provider (`true`/`false`) |
| `GEMINI_API_KEY` | Google Gemini API key |

## Tools

### list_models()

List available image generation models from enabled providers. Models are fetched dynamically from each provider's API with older/deprecated models filtered out.

### generate_image(prompt, model_id, output_path, reference_image?, width?, height?)

Generate an image from text prompt.

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `prompt` | Yes | - | Text description of the image |
| `model_id` | Yes | - | Model ID from `list_models()` |
| `output_path` | Yes | - | Path to save the generated image |
| `reference_image` | No | None | Path to reference image for style/content guidance |
| `width` | No | 1024 | Image width in pixels |
| `height` | No | 1024 | Image height in pixels |

### transform_image(image_path, prompt, model_id, output_path)

Transform an existing image based on text prompt.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `image_path` | Yes | Path to source image |
| `prompt` | Yes | Text description of transformation |
| `model_id` | Yes | Model ID from `list_models()` |
| `output_path` | Yes | Path to save the transformed image |

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
