"""
Base Exporter Class

Defines the interface for dataset exporters
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class BaseExporter(ABC):
    """Base class for dataset exporters"""

    def __init__(self, output_dir: str = "./exports"):
        """
        Initialize the exporter

        Args:
            output_dir: Directory where exported files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def export(self, examples: List[Dict[str, Any]], output_filename: str, **kwargs) -> str:
        """
        Export examples to the target format

        Args:
            examples: List of conversation examples
            output_filename: Name of the output file
            **kwargs: Additional format-specific parameters

        Returns:
            Path to the exported file
        """
        pass

    @abstractmethod
    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate that an example matches the expected format

        Args:
            example: Example to validate

        Returns:
            True if valid
        """
        pass

    def get_output_path(self, filename: str) -> Path:
        """
        Get full output path for a filename

        Args:
            filename: Output filename

        Returns:
            Full path to output file
        """
        return self.output_dir / filename

    def validate_examples(self, examples: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate all examples

        Args:
            examples: List of examples to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for idx, example in enumerate(examples):
            try:
                if not self.validate_example(example):
                    errors.append(f"Example {idx}: Invalid format")
            except Exception as e:
                errors.append(f"Example {idx}: {str(e)}")

        return (len(errors) == 0, errors)

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load examples from JSONL file

        Args:
            file_path: Path to JSONL file

        Returns:
            List of examples
        """
        examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        example = json.loads(line)
                        examples.append(example)
                    except json.JSONDecodeError:
                        continue

        return examples

    def save_jsonl(self, examples: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Save examples to JSONL format

        Args:
            examples: List of examples
            output_path: Path to save to
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

    def save_json(self, data: Any, output_path: Path, indent: int = 2) -> None:
        """
        Save data to JSON format

        Args:
            data: Data to save
            output_path: Path to save to
            indent: JSON indentation
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
