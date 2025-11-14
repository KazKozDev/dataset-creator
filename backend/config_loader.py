"""
Configuration Loader Module
Load and validate dataset generation configurations from YAML/JSON files
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator


class DataSourceConfig(BaseModel):
    """Configuration for data source"""
    type: str = Field(..., description="Type of data source: 'web', 'api', 'file', 'synthetic'")
    enabled: bool = Field(True, description="Whether this data source is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Source-specific configuration")


class WebScrapingConfig(BaseModel):
    """Configuration for web scraping"""
    urls: List[str] = Field(default_factory=list, description="List of URLs to scrape")
    start_url: Optional[str] = Field(None, description="Starting URL for crawling")
    max_pages: int = Field(100, description="Maximum pages to crawl")
    link_pattern: Optional[str] = Field(None, description="Regex pattern for filtering links")
    content_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    max_concurrent: int = Field(5, description="Maximum concurrent requests")


class APIConfig(BaseModel):
    """Configuration for API data collection"""
    api_type: str = Field(..., description="Type of API: 'github', 'stackoverflow', 'rss'")
    api_key: Optional[str] = Field(None, description="API key (can be env variable)")
    params: Dict[str, Any] = Field(default_factory=dict, description="API-specific parameters")


class FileParsingConfig(BaseModel):
    """Configuration for file parsing"""
    file_paths: List[str] = Field(default_factory=list, description="List of file paths to parse")
    file_pattern: Optional[str] = Field(None, description="Glob pattern for files")
    file_types: List[str] = Field(default_factory=list, description="File types to parse")
    chunk_size: int = Field(1000, description="Chunk size for text splitting")
    overlap: int = Field(100, description="Overlap between chunks")


class SyntheticConfig(BaseModel):
    """Configuration for synthetic data generation"""
    domain: str = Field(..., description="Domain for generation")
    subdomain: Optional[str] = Field(None, description="Subdomain for generation")
    format: str = Field("chat", description="Output format: 'chat' or 'instruction'")
    language: str = Field("en", description="Language for generation")
    count: int = Field(100, description="Number of examples to generate")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperature for generation")


class DeduplicationConfig(BaseModel):
    """Configuration for deduplication"""
    enabled: bool = Field(True, description="Enable deduplication")
    method: str = Field("hash", description="Deduplication method: 'exact', 'hash', 'fuzzy'")
    similarity_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Similarity threshold for fuzzy dedup")
    keep_first: bool = Field(True, description="Keep first occurrence of duplicate")


class CleaningConfig(BaseModel):
    """Configuration for data cleaning"""
    enabled: bool = Field(True, description="Enable data cleaning")
    min_length: int = Field(10, description="Minimum content length")
    max_length: int = Field(10000, description="Maximum content length")
    remove_toxic: bool = Field(True, description="Remove toxic content")
    toxic_patterns: List[str] = Field(default_factory=list, description="Toxic content patterns")
    remove_extra_whitespace: bool = Field(True, description="Remove extra whitespace")
    fix_encoding: bool = Field(True, description="Fix encoding issues")


class QualityControlConfig(BaseModel):
    """Configuration for quality control"""
    enabled: bool = Field(False, description="Enable quality control")
    batch_size: int = Field(10, description="Batch size for quality evaluation")
    threshold: float = Field(7.0, ge=0.0, le=10.0, description="Quality score threshold")
    auto_fix: bool = Field(False, description="Automatically fix low-quality examples")
    auto_remove: bool = Field(False, description="Automatically remove low-quality examples")


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider"""
    provider: str = Field("ollama", description="Provider type: 'ollama', 'openai', 'anthropic'")
    model: str = Field("gemma3:27b", description="Model name")
    api_key: Optional[str] = Field(None, description="API key (can be env variable)")
    base_url: Optional[str] = Field(None, description="Base URL for provider")


class OutputConfig(BaseModel):
    """Configuration for output"""
    dataset_name: Optional[str] = Field(None, description="Output dataset name")
    output_dir: str = Field("data/datasets", description="Output directory")
    format: str = Field("jsonl", description="Output format: 'jsonl', 'json', 'csv'")
    save_intermediate: bool = Field(False, description="Save intermediate results")


class DatasetPipelineConfig(BaseModel):
    """Complete configuration for dataset generation pipeline"""
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")

    # LLM Provider
    llm_provider: LLMProviderConfig = Field(default_factory=LLMProviderConfig)

    # Data Sources
    data_sources: List[DataSourceConfig] = Field(default_factory=list, description="List of data sources")

    # Processing
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    quality_control: QualityControlConfig = Field(default_factory=QualityControlConfig)

    # Output
    output: OutputConfig = Field(default_factory=OutputConfig)

    @validator('data_sources')
    def validate_data_sources(cls, v):
        if not v:
            raise ValueError("At least one data source must be configured")
        return v


class ConfigLoader:
    """Load and validate pipeline configurations"""

    def __init__(self):
        self.configs_dir = Path("configs")
        self.configs_dir.mkdir(exist_ok=True)

    def load_yaml(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            raise ValueError(f"Error loading YAML config: {e}")

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load JSON configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading JSON config: {e}")

    def load_config(self, file_path: str) -> DatasetPipelineConfig:
        """Load configuration from file (auto-detect format)"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        # Determine file format
        if file_path.suffix in ['.yaml', '.yml']:
            config_dict = self.load_yaml(str(file_path))
        elif file_path.suffix == '.json':
            config_dict = self.load_json(str(file_path))
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")

        # Resolve environment variables
        config_dict = self.resolve_env_vars(config_dict)

        # Validate and parse config
        try:
            config = DatasetPipelineConfig(**config_dict)
            return config
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")

    def resolve_env_vars(self, config: Any) -> Any:
        """Recursively resolve environment variables in config"""
        if isinstance(config, dict):
            return {k: self.resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self.resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Check if value is an environment variable reference
            if config.startswith('${') and config.endswith('}'):
                env_var = config[2:-1]
                # Support default values: ${VAR:default}
                if ':' in env_var:
                    var_name, default = env_var.split(':', 1)
                    return os.environ.get(var_name, default)
                else:
                    return os.environ.get(env_var, config)
        return config

    def save_config(self, config: DatasetPipelineConfig, file_path: str, format: str = 'yaml'):
        """Save configuration to file"""
        file_path = Path(file_path)
        config_dict = config.dict()

        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def list_configs(self) -> List[Dict[str, str]]:
        """List all available configurations"""
        configs = []

        for file_path in self.configs_dir.glob('*.yaml'):
            configs.append({
                'name': file_path.stem,
                'path': str(file_path),
                'format': 'yaml'
            })

        for file_path in self.configs_dir.glob('*.yml'):
            configs.append({
                'name': file_path.stem,
                'path': str(file_path),
                'format': 'yaml'
            })

        for file_path in self.configs_dir.glob('*.json'):
            configs.append({
                'name': file_path.stem,
                'path': str(file_path),
                'format': 'json'
            })

        return configs

    def create_example_config(self, output_path: str, template: str = 'basic'):
        """Create example configuration file"""
        if template == 'basic':
            config = DatasetPipelineConfig(
                name="example_pipeline",
                description="Example dataset generation pipeline",
                llm_provider=LLMProviderConfig(
                    provider="ollama",
                    model="gemma3:27b",
                    base_url="http://localhost:11434"
                ),
                data_sources=[
                    DataSourceConfig(
                        type="synthetic",
                        enabled=True,
                        config={
                            "domain": "support",
                            "subdomain": "technical",
                            "format": "chat",
                            "language": "en",
                            "count": 100,
                            "temperature": 0.7
                        }
                    )
                ],
                deduplication=DeduplicationConfig(
                    enabled=True,
                    method="hash",
                    similarity_threshold=0.9
                ),
                cleaning=CleaningConfig(
                    enabled=True,
                    min_length=10,
                    max_length=10000
                ),
                output=OutputConfig(
                    dataset_name="example_dataset",
                    output_dir="data/datasets",
                    format="jsonl"
                )
            )

        elif template == 'web_scraping':
            config = DatasetPipelineConfig(
                name="web_scraping_pipeline",
                description="Web scraping dataset generation pipeline",
                llm_provider=LLMProviderConfig(
                    provider="openai",
                    model="gpt-4",
                    api_key="${OPENAI_API_KEY}"
                ),
                data_sources=[
                    DataSourceConfig(
                        type="web",
                        enabled=True,
                        config={
                            "start_url": "https://example.com/docs",
                            "max_pages": 50,
                            "content_selector": "article.documentation"
                        }
                    )
                ],
                output=OutputConfig(
                    dataset_name="web_scraped_dataset",
                    format="jsonl"
                )
            )

        elif template == 'github_issues':
            config = DatasetPipelineConfig(
                name="github_issues_pipeline",
                description="GitHub issues dataset generation pipeline",
                llm_provider=LLMProviderConfig(
                    provider="anthropic",
                    model="claude-3-7-sonnet-20250219",
                    api_key="${ANTHROPIC_API_KEY}"
                ),
                data_sources=[
                    DataSourceConfig(
                        type="api",
                        enabled=True,
                        config={
                            "api_type": "github",
                            "api_key": "${GITHUB_TOKEN}",
                            "params": {
                                "repo": "owner/repository",
                                "state": "all",
                                "max_issues": 200
                            }
                        }
                    )
                ],
                output=OutputConfig(
                    dataset_name="github_issues_dataset",
                    format="jsonl"
                )
            )

        else:
            raise ValueError(f"Unknown template: {template}")

        # Save config
        self.save_config(config, output_path, format='yaml')
        return config


# Singleton instance
config_loader = ConfigLoader()


def load_pipeline_config(config_path: str) -> DatasetPipelineConfig:
    """Convenience function to load pipeline config"""
    return config_loader.load_config(config_path)


def create_example_configs():
    """Create example configuration files"""
    configs_dir = Path("configs/examples")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Create basic example
    config_loader.create_example_config(
        str(configs_dir / "basic_pipeline.yaml"),
        template='basic'
    )

    # Create web scraping example
    config_loader.create_example_config(
        str(configs_dir / "web_scraping_pipeline.yaml"),
        template='web_scraping'
    )

    # Create GitHub issues example
    config_loader.create_example_config(
        str(configs_dir / "github_issues_pipeline.yaml"),
        template='github_issues'
    )

    print("Example configurations created in configs/examples/")
