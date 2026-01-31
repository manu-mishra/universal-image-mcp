import os
import pytest
from unittest.mock import Mock
from smart_image_mcp.server import list_models, generate_image, transform_image, get_provider, is_enabled, prompt_guide

FAKE_PNG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\n\x00\x00\x00\n\x08\x02\x00\x00\x00\x02PX\xea\x00\x00\x00\x13IDATx\x9cc\xfc\xcf\x80\x0f0\xe1\x95e\x18\xa9\xd2\x00A,\x01\x13y\xed\xba&\x00\x00\x00\x00IEND\xaeB`\x82'

# Sophisticated test prompts from prompt_guide
TEST_PROMPTS = {
    "editorial_portrait": "Close-up portrait of a weathered Icelandic fisherman in his 60s, deep-set blue eyes telling stories of the sea, salt-and-pepper beard with ice crystals, wearing a thick wool sweater, standing on a fishing boat deck, overcast North Atlantic light, shot on medium format film, shallow depth of field, muted Nordic color palette",
    "architectural": "Interior of a brutalist concrete library at golden hour, dramatic shafts of warm light streaming through geometric skylights, dust particles floating in light beams, a single person reading in a leather chair, long shadows across polished concrete floor, shot with tilt-shift lens, architectural photography style",
    "product": "Luxury perfume bottle on black obsidian surface, bottle is hand-blown glass with amber liquid, single orchid petal beside it, dramatic side lighting creating long shadow, water droplets on surface suggesting freshness, reflection visible in glossy surface, high-end cosmetics advertising style, 8K product photography",
    "concept_art": "Massive ancient tree city built into a 1000-year-old baobab, multiple levels connected by rope bridges and spiral staircases carved into bark, warm lantern light glowing from windows, tiny figures going about evening activities, flying creatures circling the canopy, painted in the style of Craig Mullins, epic fantasy concept art",
    "street": "Candid moment in a Havana street, elderly Cuban man playing dominos with friends at a weathered wooden table, cigar smoke curling upward, crumbling pastel colonial building behind them, vintage American car passing in background, harsh midday Caribbean sun creating strong shadows, Leica street photography aesthetic",
    "surreal": "A grand piano melting like Dalí's clocks, dripping onto an infinite mirror floor that reflects an impossible sky filled with floating musical notes that transform into birds, the scene transitions from warm sepia tones on the left to cool cyan on the right, hyperrealistic rendering of surreal concept",
}


class TestIsEnabled:
    def test_enabled_true(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "true"})
        assert is_enabled("aws") is True

    def test_enabled_false(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "false"})
        assert is_enabled("aws") is False

    def test_enabled_missing(self, mocker):
        mocker.patch.dict(os.environ, {}, clear=True)
        assert is_enabled("aws") is False


class TestGetProvider:
    def test_aws_model(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "true", "AWS_REGION": "us-east-1"})
        provider = get_provider("amazon.nova-canvas-v1:0")
        assert provider.model == "amazon.nova-canvas-v1:0"

    def test_openai_model(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_OPENAI": "true", "OPENAI_API_KEY": "fake"})
        provider = get_provider("gpt-image-1.5")
        assert provider.model == "gpt-image-1.5"

    def test_gemini_model(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true", "GEMINI_API_KEY": "fake"})
        provider = get_provider("models/gemini-2.5-flash-image")
        assert provider.model == "models/gemini-2.5-flash-image"

    def test_imagen_model(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true", "GEMINI_API_KEY": "fake"})
        provider = get_provider("models/imagen-4.0-generate-001")
        assert provider.model == "models/imagen-4.0-generate-001"

    def test_disabled_provider(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "false"})
        with pytest.raises(ValueError, match="not enabled"):
            get_provider("amazon.nova-canvas-v1:0")

    def test_unknown_model(self, mocker):
        mocker.patch.dict(os.environ, {})
        with pytest.raises(ValueError, match="Unknown model"):
            get_provider("unknown-model-xyz")


class TestGenerateImage:
    def test_success_with_sophisticated_prompt(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true"})
        mock_provider = Mock()
        mock_provider.generate.return_value = FAKE_PNG
        mocker.patch('smart_image_mcp.server.get_provider', return_value=mock_provider)
        
        output = tmp_path / "test.png"
        result = generate_image(TEST_PROMPTS["editorial_portrait"], "models/gemini-2.5-flash-image", str(output))
        
        assert "saved" in result.lower()
        assert output.exists()
        mock_provider.generate.assert_called_once()
        # Verify the full prompt was passed
        call_args = mock_provider.generate.call_args
        assert "Icelandic fisherman" in call_args[0][0]

    def test_with_custom_dimensions(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "true"})
        mock_provider = Mock()
        mock_provider.generate.return_value = FAKE_PNG
        mocker.patch('smart_image_mcp.server.get_provider', return_value=mock_provider)
        
        output = tmp_path / "wide.png"
        result = generate_image(TEST_PROMPTS["architectural"], "amazon.nova-canvas-v1:0", str(output), width=1280, height=720)
        
        assert "saved" in result.lower()
        mock_provider.generate.assert_called_with(TEST_PROMPTS["architectural"], None, 1280, 720)

    def test_missing_image_path(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_OPENAI": "true"})
        mock_provider = Mock()
        mock_provider.generate.return_value = FAKE_PNG
        mocker.patch('smart_image_mcp.server.get_provider', return_value=mock_provider)
        
        result = generate_image(TEST_PROMPTS["product"], "gpt-image-1.5", str(tmp_path / "output.png"), reference_image="/nonexistent/image.png")
        assert "not found" in result.lower()


class TestTransformImage:
    def test_success(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "true"})
        mock_provider = Mock()
        mock_provider.transform.return_value = FAKE_PNG
        mocker.patch('smart_image_mcp.server.get_provider', return_value=mock_provider)
        
        # Create a source image
        source = tmp_path / "source.png"
        source.write_bytes(FAKE_PNG)
        output = tmp_path / "transformed.png"
        
        result = transform_image(str(source), "Convert to watercolor painting style with soft edges", "amazon.nova-canvas-v1:0", str(output))
        
        assert "saved" in result.lower()
        assert output.exists()

    def test_source_not_found(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true"})
        result = transform_image("/nonexistent/source.png", "Make it black and white", "models/gemini-2.5-flash-image", str(tmp_path / "out.png"))
        assert "not found" in result.lower()


class TestPromptGuide:
    def test_returns_content(self):
        guide = prompt_guide()
        assert "Subject" in guide
        assert "Environment" in guide
        assert "Lighting" in guide
        
    def test_contains_examples(self):
        guide = prompt_guide()
        assert "Editorial Portrait" in guide
        assert "Architectural" in guide
        assert "Product" in guide
        assert "Concept Art" in guide
        assert "Street Photography" in guide
        assert "Surreal" in guide

    def test_contains_pro_tips(self):
        guide = prompt_guide()
        assert "Pro Tips" in guide
        assert "Layer details" in guide


class TestListModels:
    def test_no_providers_enabled(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "false", "ENABLE_OPENAI": "false", "ENABLE_GEMINI": "false"}, clear=True)
        result = list_models()
        assert "No providers enabled" in result


# Integration tests - require real credentials
@pytest.mark.integration
class TestIntegrationAWS:
    def test_list_aws_models(self):
        from smart_image_mcp.providers import get_aws_models
        models = get_aws_models()
        assert len(models) > 0
        assert any("nova-canvas" in m["id"] for m in models)

    def test_generate_editorial_portrait(self, tmp_path):
        os.environ["ENABLE_AWS"] = "true"
        output = tmp_path / "editorial_portrait.png"
        result = generate_image(TEST_PROMPTS["editorial_portrait"], "amazon.nova-canvas-v1:0", str(output))
        assert "saved" in result.lower()
        assert output.exists()
        assert output.stat().st_size > 1000  # Real image should be > 1KB

    def test_generate_concept_art(self, tmp_path):
        os.environ["ENABLE_AWS"] = "true"
        output = tmp_path / "concept_art.png"
        result = generate_image(TEST_PROMPTS["concept_art"], "amazon.nova-canvas-v1:0", str(output))
        assert "saved" in result.lower()
        assert output.exists()


@pytest.mark.integration
class TestIntegrationOpenAI:
    def test_list_openai_models(self):
        from smart_image_mcp.providers import get_openai_models
        models = get_openai_models()
        assert isinstance(models, list)
        assert any("gpt-image" in m["id"] for m in models)

    def test_generate_product_shot(self, tmp_path):
        os.environ["ENABLE_OPENAI"] = "true"
        output = tmp_path / "product.png"
        result = generate_image(TEST_PROMPTS["product"], "gpt-image-1.5", str(output))
        assert "saved" in result.lower()
        assert output.exists()


@pytest.mark.integration
class TestIntegrationGemini:
    def test_list_gemini_models(self):
        from smart_image_mcp.providers import get_gemini_models
        models = get_gemini_models()
        assert len(models) > 0

    def test_generate_surreal(self, tmp_path):
        os.environ["ENABLE_GEMINI"] = "true"
        output = tmp_path / "surreal.png"
        result = generate_image(TEST_PROMPTS["surreal"], "models/gemini-2.5-flash-image", str(output))
        assert "saved" in result.lower()
        assert output.exists()

    def test_generate_street_photography(self, tmp_path):
        os.environ["ENABLE_GEMINI"] = "true"
        output = tmp_path / "street.png"
        result = generate_image(TEST_PROMPTS["street"], "models/gemini-2.5-flash-image", str(output))
        assert "saved" in result.lower()
        assert output.exists()
