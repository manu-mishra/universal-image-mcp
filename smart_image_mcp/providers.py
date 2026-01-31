"""Image generation providers - lazy initialization"""
import os
import json
import base64
from io import BytesIO
import PIL.Image

# Models to exclude from listing (older/experimental/specialized versions)
EXCLUDED_MODELS = {
    "gemini": [
        "models/gemini-2.0-flash-exp-image-generation",
        "models/imagen-4.0-generate-preview-06-06",
        "models/imagen-4.0-ultra-generate-preview-06-06",
    ],
    "aws": [
        "amazon.titan-image-generator-v2:0",
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
        "dall-e-2",
        "dall-e-3",
        "gpt-image-1",
        "gpt-image-1-mini",
    ],
}

# --- Model listing (lazy imports) ---

def get_aws_models():
    import boto3
    from botocore.exceptions import TokenRetrievalError, NoCredentialsError, ClientError
    try:
        region = os.getenv("AWS_REGION", "us-east-1")
        profile = os.getenv("AWS_PROFILE")
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        client = session.client("bedrock", region_name=region)
        response = client.list_foundation_models(byOutputModality="IMAGE")
        
        excluded = EXCLUDED_MODELS.get("aws", [])
        return [{"id": m["modelId"], "name": m["modelName"], "provider": m["providerName"],
                 "input": m["inputModalities"], "status": m["modelLifecycle"]["status"]} 
                for m in response.get("modelSummaries", []) if m["modelId"] not in excluded]
    except TokenRetrievalError:
        raise ValueError("AWS SSO session expired. Run 'aws sso login' to refresh.")
    except NoCredentialsError:
        raise ValueError("AWS credentials not found. Configure AWS_PROFILE or credentials.")
    except ClientError as e:
        raise ValueError(f"AWS API error: {e.response['Error']['Message']}")

def get_openai_models():
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.models.list()
    
    excluded = EXCLUDED_MODELS.get("openai", [])
    return [{"id": m.id, "created": m.created, "owned_by": m.owned_by} 
            for m in response.data 
            if any(x in m.id.lower() for x in ["image", "dall", "gpt-image"]) and m.id not in excluded]

def get_gemini_models():
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.list()
    
    excluded = EXCLUDED_MODELS.get("gemini", [])
    return [{"id": m.name, "name": m.display_name, "description": m.description} 
            for m in response if "image" in m.name.lower() and m.name not in excluded]

# --- Providers (lazy client creation) ---

class AWSProvider:
    def __init__(self, model: str):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            import boto3
            region = os.getenv("AWS_REGION", "us-east-1")
            profile = os.getenv("AWS_PROFILE")
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self._client = session.client("bedrock-runtime", region_name=region)
        return self._client
    
    def _call_model(self, body: str) -> dict:
        from botocore.exceptions import TokenRetrievalError, NoCredentialsError, ClientError
        try:
            response = self.client.invoke_model(modelId=self.model, body=body)
            return json.loads(response["body"].read())
        except TokenRetrievalError:
            raise ValueError("AWS SSO session expired. Run 'aws sso login' to refresh.")
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Configure AWS_PROFILE or credentials.")
        except ClientError as e:
            raise ValueError(f"AWS API error: {e.response['Error']['Message']}")
    
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
        
        result = self._call_model(body)
        
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
        
        result = self._call_model(body)
        
        if "nova-canvas" in self.model:
            return base64.b64decode(result["images"][0])
        return base64.b64decode(result["artifacts"][0]["base64"])


class OpenAIProvider:
    def __init__(self, model: str):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def generate(self, prompt: str, reference: PIL.Image.Image = None, width: int = 1024, height: int = 1024) -> bytes:
        from openai import AuthenticationError, APIError
        if reference:
            return self.transform(reference, prompt)
        try:
            response = self.client.images.generate(model=self.model, prompt=prompt, n=1, size=f"{width}x{height}")
            return base64.b64decode(response.data[0].b64_json)
        except AuthenticationError:
            raise ValueError("OpenAI API key invalid or expired. Check OPENAI_API_KEY.")
        except APIError as e:
            raise ValueError(f"OpenAI API error: {e.message}")
    
    def transform(self, image: PIL.Image.Image, prompt: str) -> bytes:
        from openai import AuthenticationError, APIError
        try:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            response = self.client.images.edit(
                model=self.model,
                image=[("image.png", buffer, "image/png")],
                prompt=prompt
            )
            return base64.b64decode(response.data[0].b64_json)
        except AuthenticationError:
            raise ValueError("OpenAI API key invalid or expired. Check OPENAI_API_KEY.")
        except APIError as e:
            raise ValueError(f"OpenAI API error: {e.message}")


class GeminiProvider:
    def __init__(self, model: str):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client
    
    def _generate_content(self, contents, config):
        from google.api_core.exceptions import Unauthenticated, PermissionDenied, GoogleAPIError
        try:
            return self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except Unauthenticated:
            raise ValueError("Gemini API key invalid. Check GEMINI_API_KEY.")
        except PermissionDenied:
            raise ValueError("Gemini API key lacks permission for this model.")
        except GoogleAPIError as e:
            raise ValueError(f"Gemini API error: {e.message}")
    
    def generate(self, prompt: str, reference: PIL.Image.Image = None, width: int = 1024, height: int = 1024) -> bytes:
        if reference:
            return self.transform(reference, prompt)
        from google.genai import types
        response = self._generate_content([prompt], types.GenerateContentConfig(response_modalities=['Text', 'Image']))
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        raise ValueError("No image in response")
    
    def transform(self, image: PIL.Image.Image, prompt: str) -> bytes:
        from google.genai import types
        response = self._generate_content([image, prompt], types.GenerateContentConfig(response_modalities=['Text', 'Image']))
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return part.inline_data.data
        raise ValueError("No image in response")
