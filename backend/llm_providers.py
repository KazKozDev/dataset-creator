"""
LLM Provider integrations for LLM Dataset Creator
Supports Ollama, OpenAI (ChatGPT), and Anthropic (Claude)
Enhanced with registry pattern and configuration management
"""

import os
import json
import requests
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from config import get_config, get_provider_config

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available"""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get available models from the provider"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gpt-oss:20b", timeout: int = 300):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Ollama API"""
        try:
            # Prepare the request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Make the API request to Ollama
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            
            if response.status_code != 200:
                print(f"Ollama API error: {response.status_code}, {response.text}")
                return ""
            
            result = response.json().get("response", "").strip()
            return result
            
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text response"""
        try:
            # Extract JSON from the response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Find JSON object bounds
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                clean_json = text[start_idx:end_idx]
                return json.loads(clean_json)
            else:
                print("Could not find valid JSON in response")
                return None
                
        except Exception as e:
            print(f"Error extracting JSON: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the specified model exists"""
        try:
            # Check Ollama API availability
            response = requests.get(f"{self.base_url}/api/version")
            if response.status_code != 200:
                print(f"Warning: Could not connect to Ollama API at {self.base_url}: {response.status_code}")
                return False
            
            # Check if the model is available
            models_response = requests.get(f"{self.base_url}/api/tags")
            if models_response.status_code == 200:
                available_models = [model["name"] for model in models_response.json().get("models", [])]
                if self.model not in available_models:
                    print(f"Warning: Model '{self.model}' not found in available models: {available_models}")
                    print(f"You may need to pull the model: ollama pull {self.model}")
                    return False
                return True
            return False
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from Ollama"""
        try:
            models_response = requests.get(f"{self.base_url}/api/tags")
            if models_response.status_code == 200:
                return [model["name"] for model in models_response.json().get("models", [])]
            return []
        except Exception:
            return []

class OpenAIProvider(LLMProvider):
    """OpenAI (ChatGPT) provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", timeout: int = 120):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.timeout = timeout
        
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using OpenAI API"""
        try:
            if not self.api_key:
                print("OpenAI API key not found")
                return ""
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"OpenAI API error: {response.status_code}, {response.text}")
                return ""
            
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(f"{self.base_url}/models", headers=headers)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from OpenAI"""
        if not self.api_key:
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(f"{self.base_url}/models", headers=headers)
            if response.status_code == 200:
                models = response.json().get("data", [])
                # Filter to only include GPT models
                gpt_models = [model["id"] for model in models if "gpt" in model["id"].lower()]
                return gpt_models
            return []
        except Exception:
            return []

class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219", timeout: int = 120):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = timeout
        
    def generate_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Anthropic API"""
        try:
            if not self.api_key:
                print("Anthropic API key not found")
                return ""
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                print(f"Anthropic API error: {response.status_code}, {response.text}")
                return ""
            
            return response.json()["content"][0]["text"].strip()
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Simple request to check API availability
            response = requests.get(f"{self.base_url}/models", headers=headers)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self) -> List[str]:
        """Get available models from Anthropic"""
        if not self.api_key:
            return []
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.get(f"{self.base_url}/models", headers=headers)
            if response.status_code == 200:
                return [model["id"] for model in response.json().get("models", [])]
            return []
        except Exception:
            return []

def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to create an LLM provider instance"""
    if provider_type.lower() == "ollama":
        return OllamaProvider(
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            model=kwargs.get("model", "gemma3:27b")
        )
    elif provider_type.lower() == "openai":
        return OpenAIProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gpt-4")
        )
    elif provider_type.lower() == "anthropic":
        return AnthropicProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "claude-3-7-sonnet-20250219")
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

def get_provider_types() -> List[str]:
    """Get list of supported provider types"""
    return ["ollama", "openai", "anthropic"]


# ====================
# Provider Registry
# ====================

class ProviderRegistry:
    """Registry for managing LLM providers"""

    _providers: Dict[str, Type[LLMProvider]] = {}
    _instances: Dict[str, LLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider class"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_provider_class(cls, name: str) -> Optional[Type[LLMProvider]]:
        """Get a provider class by name"""
        return cls._providers.get(name.lower())

    @classmethod
    def create_provider(cls, name: str, **kwargs) -> Optional[LLMProvider]:
        """Create a provider instance with given parameters"""
        provider_class = cls.get_provider_class(name)
        if provider_class:
            return provider_class(**kwargs)
        return None

    @classmethod
    def create_from_config(cls, provider_name: Optional[str] = None) -> Optional[LLMProvider]:
        """Create a provider instance from configuration"""
        config = get_config()

        # Get provider name
        if provider_name is None:
            provider_name = config.get_default_provider()

        # Get provider configuration
        provider_config = config.get_provider_config(provider_name)

        if not provider_config:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")

        if not provider_config.get('enabled', False):
            raise ValueError(f"Provider '{provider_name}' is not enabled in configuration")

        # Get provider class
        provider_class = cls.get_provider_class(provider_name)
        if not provider_class:
            raise ValueError(f"Provider '{provider_name}' is not registered")

        # Prepare initialization parameters
        init_params = {}

        if provider_name == "ollama":
            init_params = {
                "base_url": provider_config.get("base_url", "http://localhost:11434"),
                "model": provider_config.get("default_model", "gpt-oss:20b"),
                "timeout": provider_config.get("timeout", 300)
            }
        elif provider_name == "openai":
            init_params = {
                "api_key": provider_config.get("api_key"),
                "model": provider_config.get("default_model", "gpt-4-turbo-preview"),
                "timeout": provider_config.get("timeout", 120)
            }
        elif provider_name == "anthropic":
            init_params = {
                "api_key": provider_config.get("api_key"),
                "model": provider_config.get("default_model", "claude-3-7-sonnet-20250219"),
                "timeout": provider_config.get("timeout", 120)
            }

        # Create instance
        return provider_class(**init_params)

    @classmethod
    def get_or_create_provider(cls, provider_name: Optional[str] = None, **override_params) -> LLMProvider:
        """Get cached provider instance or create new one"""
        config = get_config()

        # Get provider name
        if provider_name is None:
            provider_name = config.get_default_provider()

        # Create cache key
        cache_key = f"{provider_name}:{json.dumps(override_params, sort_keys=True)}"

        # Check cache
        if cache_key not in cls._instances:
            # Create new instance
            if override_params:
                # Create with override parameters
                cls._instances[cache_key] = cls.create_provider(provider_name, **override_params)
            else:
                # Create from config
                cls._instances[cache_key] = cls.create_from_config(provider_name)

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached provider instances"""
        cls._instances.clear()

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers"""
        return list(cls._providers.keys())

    @classmethod
    def get_available_providers(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available and enabled providers from configuration"""
        config = get_config()
        return config.get_enabled_providers()


# Register built-in providers
ProviderRegistry.register("ollama", OllamaProvider)
ProviderRegistry.register("openai", OpenAIProvider)
ProviderRegistry.register("anthropic", AnthropicProvider)


# ====================
# Convenience Functions
# ====================

def get_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Get or create a provider instance

    Args:
        provider_name: Name of the provider (ollama, openai, anthropic)
        **kwargs: Override parameters for provider initialization

    Returns:
        LLMProvider instance
    """
    return ProviderRegistry.get_or_create_provider(provider_name, **kwargs)


def get_default_provider() -> LLMProvider:
    """Get the default provider from configuration"""
    return ProviderRegistry.create_from_config()


def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """List all available providers from configuration"""
    return ProviderRegistry.get_available_providers()


def get_provider_models(provider_name: Optional[str] = None) -> List[str]:
    """Get available models for a provider"""
    try:
        provider = get_provider(provider_name)
        return provider.get_models()
    except Exception as e:
        print(f"Error getting models for provider '{provider_name}': {e}")
        return []


def test_provider_availability(provider_name: Optional[str] = None) -> bool:
    """Test if a provider is available"""
    try:
        provider = get_provider(provider_name)
        return provider.is_available()
    except Exception as e:
        print(f"Error testing provider '{provider_name}': {e}")
        return False