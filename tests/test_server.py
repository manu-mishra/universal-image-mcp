import os
import pytest
from unittest.mock import Mock
from smart_image_mcp.server import list_models, generate_image, get_provider, is_enabled

FAKE_PNG = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'


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

    def test_gemini_model(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true", "GEMINI_API_KEY": "fake"})
        provider = get_provider("gemini-2.5-flash-image")
        assert provider.model == "gemini-2.5-flash-image"

    def test_disabled_provider(self, mocker):
        mocker.patch.dict(os.environ, {"ENABLE_AWS": "false"})
        with pytest.raises(ValueError, match="not enabled"):
            get_provider("amazon.nova-canvas-v1:0")


class TestGenerateImage:
    def test_success(self, mocker, tmp_path):
        mocker.patch.dict(os.environ, {"ENABLE_GEMINI": "true"})
        mock_provider = Mock()
        mock_provider.generate.return_value = FAKE_PNG
        mocker.patch('smart_image_mcp.server.get_provider', return_value=mock_provider)
        
        output = tmp_path / "test.png"
        result = generate_image("a cat", "gemini-2.5-flash-image", str(output))
        
        assert "saved" in result.lower()
        assert output.exists()


# Integration tests
@pytest.mark.integration
class TestIntegrationAWS:
    def test_list_aws_models(self):
        from smart_image_mcp.providers import get_aws_models
        models = get_aws_models()
        assert len(models) > 0


@pytest.mark.integration
class TestIntegrationGemini:
    def test_list_gemini_models(self):
        from smart_image_mcp.providers import get_gemini_models
        models = get_gemini_models()
        assert len(models) > 0


@pytest.mark.integration
class TestIntegrationOpenAI:
    def test_list_openai_models(self):
        from smart_image_mcp.providers import get_openai_models
        models = get_openai_models()
        assert isinstance(models, list)
