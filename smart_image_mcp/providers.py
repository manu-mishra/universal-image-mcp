"""Image generation providers"""
import os
import json
import base64
from io import BytesIO
import PIL.Image
import boto3
from openai import OpenAI
from google import genai
from google.genai import types

# --- Model listing ---

def get_aws_models():
    region = os.getenv("AWS_REGION", "us-east-1")
    profile = os.getenv("AWS_PROFILE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    client = session.client("bedrock", region_name=region)
    response = client.list_foundation_models(byOutputModality="IMAGE")
    
    excluded = EXCLUDED_MODELS.get("aws", [])
    return [{"id": m["modelId"], "name": m["modelName"], "provider": m["providerName"],
             "input": m["inputModalities"], "status": m["modelLifecycle"]["status"]} 
            for m in response.get("modelSummaries", []) if m["modelId"] not in excluded]

def get_openai_models():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.models.list()
    
    excluded = EXCLUDED_MODELS.get("openai", [])
    return [{"id": m.id, "created": m.created, "owned_by": m.owned_by} 
            for m in response.data 
            if any(x in m.id.lower() for x in ["image", "dall", "gpt-image"]) and m.id not in excluded]

# Models to exclude from listing (older/experimental/specialized versions)
EXCLUDED_MODELS = {
    "gemini": [
        "models/gemini-2.0-flash-exp-image-generation",  # Experimental, use 2.5 instead
        "models/imagen-4.0-generate-preview-06-06",      # Preview, use GA version
        "models/imagen-4.0-ultra-generate-preview-06-06", # Preview, use GA version
    ],
    "aws": [
        # Older generation
        "amazon.titan-image-generator-v2:0",
        # Stability AI specialized editing tools (not text-to-image generators)
        "stability.stable-creative-upscale-v1:0",
        "stability.stable-conservative-upscale-v1:0",
        "stability.stable-fast-upscale-v1:0",
        "stability.stable-image-remove-background-v1:0",
        "stability.stable-image-control-sketch-v1:0",
        "stability.stable-image-control-structure-v1:0",
        "stability.stable-image-search-recolor-v1:0",
        "stability.stable-image-search-replace-v1:0",
        "stability.stable-image-erase-object-v1:0",
        "stability.stable-image-style-guide-v1:0",
        "stability.stable-style-transfer-v1:0",
        "stability.stable-outpaint-v1:0",
        "stability.stable-image-inpaint-v1:0",
    ],
    "openai": [
        "dall-e-2",  # Deprecated
        "dall-e-3",  # Legacy, replaced by GPT Image
        "gpt-image-1",  # Older, use 1.5
        "gpt-image-1-mini",  # Older, use 1.5
    ],
}

def get_gemini_models():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.list()
    
    excluded = EXCLUDED_MODELS.get("gemini", [])
    return [{"id": m.name, "name": m.display_name, "description": m.description} 
            for m in response if "image" in m.name.lower() and m.name not in excluded]

# --- Providers ---

class AWSProvider:
    def __init__(self, model: str):
        self.model = model
        region = os.getenv("AWS_REGION", "us-east-1")
        profile = os.getenv("AWS_PROFILE")
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.client = session.client("bedrock-runtime", region_name=region)
    
    def generate(self, prompt: str, reference: PIL.Image.Image = None, width: int = 1024, height: int = 1024) -> bytes:
        if reference:
            return self.transform(reference, prompt)
        
        if "nova-canvas" in self.model:
            body = json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {"numberOfImages": 1, "width": width, "height": height}
            })
        else:
            body = json.dumps({"text_prompts": [{"text": prompt}], "cfg_scale": 10, "steps": 50, "width": width, "height": height})
        
        response = self.client.invoke_model(modelId=self.model, body=body)
        result = json.loads(response["body"].read())
        
        if "nova-canvas" in self.model:
            return base64.b64decode(result["images"][0])
        return base64.b64decode(result["artifacts"][0]["base64"])
    
    def transform(self, image: PIL.Image.Image, prompt: str) -> bytes:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        init_image = base64.b64encode(buffer.getvalue()).decode()
        
        if "nova-canvas" in self.model:
            body = json.dumps({
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {"text": prompt, "images": [init_image]},
                "imageGenerationConfig": {"numberOfImages": 1, "width": 1024, "height": 1024}
            })
        else:
            body = json.dumps({"text_prompts": [{"text": prompt}], "init_image": init_image, "cfg_scale": 10, "steps": 50})
        
        response = self.client.invoke_model(modelId=self.model, body=body)
        result = json.loads(response["body"].read())
        
        if "nova-canvas" in self.model:
            return base64.b64decode(result["images"][0])
        return base64.b64decode(result["artifacts"][0]["base64"])


class OpenAIProvider:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, reference: PIL.Image.Image = None, width: int = 1024, height: int = 1024) -> bytes:
        if reference:
            return self.transform(reference, prompt)
        response = self.client.images.generate(model=self.model, prompt=prompt, n=1, size=f"{width}x{height}")
        return base64.b64decode(response.data[0].b64_json)
    
    def transform(self, image: PIL.Image.Image, prompt: str) -> bytes:
        # Use images.edit with image as reference for generation
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        response = self.client.images.edit(
            model=self.model,
            image=[("image.png", buffer, "image/png")],
            prompt=prompt
        )
        return base64.b64decode(response.data[0].b64_json)


class GeminiProvider:
    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def generate(self, prompt: str, reference: PIL.Image.Image = None, width: int = 1024, height: int = 1024) -> bytes:
        if reference:
            return self.transform(reference, prompt)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        raise ValueError("No image in response")
    
    def transform(self, image: PIL.Image.Image, prompt: str) -> bytes:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[f"Transform this image: {prompt}", image],
            config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        raise ValueError("No image in response")
