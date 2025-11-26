"""
Model configuration manager
Loads available models from models.json
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class ModelManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to models.json in project root (parent of backend dir)
            backend_dir = Path(__file__).parent
            config_path = backend_dir.parent / "models.json"
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load models configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            "providers": {
                "ollama": {
                    "name": "Ollama (Local)",
                    "type": "local",
                    "requires_api_key": False,
                    "models": []
                }
            },
            "default_provider": "ollama",
            "default_model": {"ollama": "llama3.2"}
        }
    
    def get_providers(self) -> List[Dict[str, Any]]:
        """Get list of all available providers"""
        providers = []
        for provider_id, provider_data in self.config.get("providers", {}).items():
            # Check if API key is configured for paid providers
            available = True
            if provider_data.get("requires_api_key"):
                api_key_env = provider_data.get("api_key_env")
                available = bool(os.getenv(api_key_env))
            
            providers.append({
                "id": provider_id,
                "name": provider_data.get("name", provider_id),
                "type": provider_data.get("type", "unknown"),
                "requires_api_key": provider_data.get("requires_api_key", False),
                "available": available,
                "model_count": len(provider_data.get("models", []))
            })
        
        return providers
    
    def get_models(self, provider: str) -> List[Dict[str, Any]]:
        """Get list of models for a specific provider"""
        provider_data = self.config.get("providers", {}).get(provider, {})
        
        # For Ollama, dynamically fetch installed models
        if provider == "ollama":
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    ollama_models = response.json().get("models", [])
                    return [
                        {
                            "id": model["name"],
                            "name": model["name"],
                            "description": f"Size: {model.get('size', 'unknown')}",
                            "cost_per_1k_tokens": 0.0,
                            "context_window": 4096
                        }
                        for model in ollama_models
                    ]
            except Exception as e:
                print(f"Failed to fetch Ollama models: {e}")
                return []
        
        return provider_data.get("models", [])
    
    def get_model_info(self, provider: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        models = self.get_models(provider)
        for model in models:
            if model.get("id") == model_id:
                return model
        return None
    
    def get_default_model(self, provider: str) -> Optional[str]:
        """Get default model for a provider"""
        return self.config.get("default_model", {}).get(provider)
    
    def estimate_cost(self, provider: str, model_id: str, tokens: int) -> float:
        """Estimate cost for a given number of tokens"""
        model_info = self.get_model_info(provider, model_id)
        if not model_info:
            return 0.0
        
        cost_per_1k = model_info.get("cost_per_1k_tokens", 0.0)
        return (tokens / 1000) * cost_per_1k
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()

# Global instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get or create global ModelManager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
