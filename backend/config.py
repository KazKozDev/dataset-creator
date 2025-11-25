"""
Configuration management module for LLM Dataset Creator
Handles loading and managing configuration from YAML files
"""

import os
import yaml
import re
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration manager for the application"""

    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: str = ""

    def __new__(cls):
        """Singleton pattern to ensure only one config instance"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the configuration"""
        if not self._config:
            self.load_config()

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path is None:
            # Try multiple locations
            possible_paths = [
                "config.yaml",
                "backend/config.yaml",
                os.path.join(os.path.dirname(__file__), "config.yaml"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break

            if config_path is None:
                raise FileNotFoundError("Configuration file not found. Please create config.yaml")

        self._config_path = config_path

        with open(config_path, 'r') as f:
            config_content = f.read()

            # Replace environment variables
            config_content = self._replace_env_vars(config_content)

            # Load YAML
            self._config = yaml.safe_load(config_content)

        return self._config

    def _replace_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        pattern = re.compile(r'\$\{([^}]+)\}')

        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return pattern.sub(replace, content)

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """Save current configuration to YAML file"""
        try:
            path = config_path or self._config_path

            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider"""
        return self.get(f'providers.{provider_name}')

    def get_default_provider(self) -> str:
        """Get the default provider name"""
        return self.get('default_provider', 'ollama')

    def set_default_provider(self, provider_name: str) -> None:
        """Set the default provider"""
        self.set('default_provider', provider_name)

    def get_enabled_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get all enabled providers"""
        providers = self.get('providers', {})
        return {
            name: config
            for name, config in providers.items()
            if config.get('enabled', False)
        }

    def update_provider_config(self, provider_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific provider"""
        current_config = self.get_provider_config(provider_name) or {}
        current_config.update(config)
        self.set(f'providers.{provider_name}', current_config)

    def add_custom_provider(self, provider_name: str, config: Dict[str, Any]) -> None:
        """Add a new custom provider configuration"""
        providers = self.get('providers', {})
        providers[provider_name] = config
        self.set('providers', providers)

    def get_generation_settings(self) -> Dict[str, Any]:
        """Get generation settings"""
        return self.get('generation', {})

    def get_quality_settings(self) -> Dict[str, Any]:
        """Get quality control settings"""
        return self.get('quality', {})

    def get_system_settings(self) -> Dict[str, Any]:
        """Get system settings"""
        return self.get('system', {})

    def get_database_settings(self) -> Dict[str, Any]:
        """Get database settings"""
        return self.get('database', {})

    def get_api_settings(self) -> Dict[str, Any]:
        """Get API settings"""
        return self.get('api', {})

    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file"""
        return self.load_config(self._config_path)

    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary"""
        return self._config.copy()

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist"""
        system = self.get_system_settings()

        directories = [
            system.get('data_directory', './data'),
            system.get('datasets_directory', './data/datasets'),
            system.get('cache_directory', './data/cache'),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config_instance = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def reload_config() -> Config:
    """Reload the global configuration"""
    global _config_instance
    _config_instance = Config()
    _config_instance.reload()
    return _config_instance


# Convenience functions
def get_provider_config(provider_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific provider"""
    return get_config().get_provider_config(provider_name)

def get_default_provider() -> str:
    """Get the default provider name"""
    return get_config().get_default_provider()

def get_enabled_providers() -> Dict[str, Dict[str, Any]]:
    """Get all enabled providers"""
    return get_config().get_enabled_providers()

def get_generation_settings() -> Dict[str, Any]:
    """Get generation settings"""
    return get_config().get_generation_settings()

def get_quality_settings() -> Dict[str, Any]:
    """Get quality control settings"""
    return get_config().get_quality_settings()

def get_system_settings() -> Dict[str, Any]:
    """Get system settings"""
    return get_config().get_system_settings()
