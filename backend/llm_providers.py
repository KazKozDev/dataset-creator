"""
LLM Provider integrations for LLM Dataset Creator
Supports Ollama, OpenAI (ChatGPT), and Anthropic (Claude)
"""

import os
import json
import requests
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

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
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:27b"):
        self.base_url = base_url
        self.model = model
        
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
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
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
                json=payload
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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        
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
                json=payload
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