"""
Prompt Template Schema Definitions

Defines the structure for YAML-based prompt templates
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class VariableType(str, Enum):
    """Types of variables that can be used in prompts"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    ARRAY = "array"


class PromptVariable(BaseModel):
    """Definition of a variable used in prompt template"""
    name: str = Field(..., description="Variable name")
    type: VariableType = Field(VariableType.STRING, description="Variable type")
    description: str = Field("", description="Variable description")
    required: bool = Field(True, description="Whether variable is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum_values: Optional[List[str]] = Field(None, description="Allowed values for enum type")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric types")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric types")

    @validator('enum_values')
    def validate_enum_values(cls, v, values):
        if values.get('type') == VariableType.ENUM and not v:
            raise ValueError("enum_values must be provided for enum type")
        return v


class PromptMetadata(BaseModel):
    """Metadata about the prompt template"""
    name: str = Field(..., description="Template name")
    version: str = Field("1.0.0", description="Template version (semver)")
    author: str = Field("", description="Template author")
    description: str = Field("", description="Template description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    domain: Optional[str] = Field(None, description="Associated domain")
    subdomain: Optional[str] = Field(None, description="Associated subdomain")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class PromptTemplate(BaseModel):
    """Complete prompt template definition"""
    metadata: PromptMetadata = Field(..., description="Template metadata")
    variables: List[PromptVariable] = Field(default_factory=list, description="Template variables")
    system_prompt: Optional[str] = Field(None, description="System prompt (for chat models)")
    user_prompt: str = Field(..., description="Main prompt template")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Few-shot examples")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Default temperature")
    max_tokens: int = Field(2000, ge=1, le=32000, description="Max tokens to generate")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "name": "tech_support_troubleshooting",
                    "version": "1.0.0",
                    "author": "AI Team",
                    "description": "Template for generating technical support conversations",
                    "tags": ["support", "technical", "troubleshooting"],
                    "domain": "support",
                    "subdomain": "tech_support"
                },
                "variables": [
                    {
                        "name": "product_type",
                        "type": "enum",
                        "description": "Type of product",
                        "required": True,
                        "enum_values": ["software", "hardware", "service"]
                    },
                    {
                        "name": "complexity",
                        "type": "enum",
                        "description": "Issue complexity",
                        "required": False,
                        "default": "medium",
                        "enum_values": ["simple", "medium", "complex"]
                    }
                ],
                "system_prompt": "You are a helpful technical support agent.",
                "user_prompt": "Generate a technical support conversation about {{product_type}} with {{complexity}} complexity.",
                "temperature": 0.7,
                "max_tokens": 2000
            }
        }

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the prompt template with given variables"""
        prompt = self.user_prompt

        # Simple variable substitution using {{variable_name}} syntax
        for var in self.variables:
            value = variables.get(var.name, var.default)

            if value is None and var.required:
                raise ValueError(f"Required variable '{var.name}' not provided")

            if value is not None:
                prompt = prompt.replace(f"{{{{{var.name}}}}}", str(value))

        return prompt

    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """Validate provided variables against template definition"""
        for var in self.variables:
            value = variables.get(var.name)

            # Check required
            if var.required and value is None:
                raise ValueError(f"Required variable '{var.name}' not provided")

            if value is None:
                continue

            # Check type
            if var.type == VariableType.INTEGER and not isinstance(value, int):
                raise TypeError(f"Variable '{var.name}' must be integer")
            elif var.type == VariableType.FLOAT and not isinstance(value, (int, float)):
                raise TypeError(f"Variable '{var.name}' must be numeric")
            elif var.type == VariableType.BOOLEAN and not isinstance(value, bool):
                raise TypeError(f"Variable '{var.name}' must be boolean")
            elif var.type == VariableType.ARRAY and not isinstance(value, list):
                raise TypeError(f"Variable '{var.name}' must be array")

            # Check enum values
            if var.type == VariableType.ENUM and value not in var.enum_values:
                raise ValueError(f"Variable '{var.name}' must be one of {var.enum_values}")

            # Check numeric bounds
            if var.type in [VariableType.INTEGER, VariableType.FLOAT]:
                if var.min_value is not None and value < var.min_value:
                    raise ValueError(f"Variable '{var.name}' must be >= {var.min_value}")
                if var.max_value is not None and value > var.max_value:
                    raise ValueError(f"Variable '{var.name}' must be <= {var.max_value}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary"""
        return cls(**data)
