"""
HuggingFace Exporter

Exports datasets to HuggingFace datasets format
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseExporter


class HuggingFaceExporter(BaseExporter):
    """Exporter for HuggingFace datasets format"""

    def export(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str,
        dataset_name: str = "custom_dataset",
        split: str = "train",
        features_schema: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """
        Export examples to HuggingFace format

        Args:
            examples: List of conversation examples
            output_filename: Name of the output file (without extension)
            dataset_name: Name of the dataset
            split: Dataset split (train, validation, test)
            features_schema: Optional custom schema for features
            **kwargs: Additional parameters

        Returns:
            Path to the exported directory
        """
        # Validate examples
        is_valid, errors = self.validate_examples(examples)
        if not is_valid:
            raise ValueError(f"Invalid examples: {'; '.join(errors)}")

        # Create dataset directory
        dataset_dir = self.output_dir / output_filename
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Convert examples to HuggingFace format
        hf_examples = []
        for example in examples:
            hf_example = self._convert_to_hf_format(example)
            hf_examples.append(hf_example)

        # Save data
        data_file = dataset_dir / f"{split}.jsonl"
        self.save_jsonl(hf_examples, data_file)

        # Create dataset_info.json
        dataset_info = self._create_dataset_info(
            dataset_name=dataset_name,
            num_examples=len(hf_examples),
            features_schema=features_schema or self._infer_features_schema(hf_examples[0])
        )

        info_file = dataset_dir / "dataset_info.json"
        self.save_json(dataset_info, info_file)

        # Create README.md
        readme_content = self._create_readme(
            dataset_name=dataset_name,
            num_examples=len(hf_examples),
            split=split
        )

        readme_file = dataset_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        return str(dataset_dir)

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate example format

        Args:
            example: Example to validate

        Returns:
            True if valid
        """
        # Check if it's a conversation or instruction format
        if "conversation" in example or "messages" in example:
            return True
        elif "instruction" in example or "prompt" in example:
            return True
        elif "input" in example and "output" in example:
            return True
        else:
            # Check for text field
            return "text" in example or "content" in example

    def _convert_to_hf_format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert example to HuggingFace format

        Args:
            example: Original example

        Returns:
            HuggingFace formatted example
        """
        hf_example = {}

        # Handle conversation format
        if "conversation" in example:
            hf_example["messages"] = example["conversation"]
        elif "messages" in example:
            hf_example["messages"] = example["messages"]

        # Handle instruction format
        elif "instruction" in example:
            hf_example["instruction"] = example["instruction"]
            hf_example["input"] = example.get("input", "")
            hf_example["output"] = example.get("output", "")

        # Handle prompt-completion format
        elif "prompt" in example:
            hf_example["prompt"] = example["prompt"]
            hf_example["completion"] = example.get("completion", example.get("response", ""))

        # Handle simple input-output format
        elif "input" in example and "output" in example:
            hf_example["input"] = example["input"]
            hf_example["output"] = example["output"]

        # Handle text format
        elif "text" in example:
            hf_example["text"] = example["text"]

        # Copy metadata fields
        for key in ["id", "metadata", "domain", "subdomain", "language"]:
            if key in example:
                hf_example[key] = example[key]

        return hf_example

    def _infer_features_schema(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Infer feature schema from an example

        Args:
            example: Example to infer schema from

        Returns:
            Features schema dictionary
        """
        schema = {}

        for key, value in example.items():
            if isinstance(value, str):
                schema[key] = "string"
            elif isinstance(value, int):
                schema[key] = "int32"
            elif isinstance(value, float):
                schema[key] = "float32"
            elif isinstance(value, bool):
                schema[key] = "bool"
            elif isinstance(value, list):
                schema[key] = "list"
            elif isinstance(value, dict):
                schema[key] = "dict"
            else:
                schema[key] = "string"

        return schema

    def _create_dataset_info(
        self,
        dataset_name: str,
        num_examples: int,
        features_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create dataset_info.json content

        Args:
            dataset_name: Name of the dataset
            num_examples: Number of examples
            features_schema: Features schema

        Returns:
            Dataset info dictionary
        """
        return {
            "dataset_name": dataset_name,
            "description": f"Synthetic dataset created with LLM Dataset Creator",
            "version": "1.0.0",
            "features": features_schema,
            "splits": {
                "train": {
                    "num_examples": num_examples
                }
            },
            "download_size": 0,
            "dataset_size": 0,
            "configs": [
                {
                    "config_name": "default",
                    "data_files": {
                        "train": "train.jsonl"
                    }
                }
            ]
        }

    def _create_readme(
        self,
        dataset_name: str,
        num_examples: int,
        split: str
    ) -> str:
        """
        Create README.md content

        Args:
            dataset_name: Name of the dataset
            num_examples: Number of examples
            split: Dataset split

        Returns:
            README content
        """
        return f"""# {dataset_name}

## Dataset Description

This dataset was automatically generated using the LLM Dataset Creator.

### Dataset Summary

- **Number of examples**: {num_examples}
- **Split**: {split}
- **Format**: JSONL (JSON Lines)

## Usage

### Loading with HuggingFace datasets

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='{split}.jsonl')
```

### Loading with pandas

```python
import pandas as pd

df = pd.read_json('{split}.jsonl', lines=True)
```

## Dataset Structure

See `dataset_info.json` for the complete feature schema.

## Citation

If you use this dataset, please cite:

```
@misc{{{dataset_name},
  title={{{dataset_name}}},
  author={{LLM Dataset Creator}},
  year={{2025}},
  howpublished={{\\url{{https://github.com/yourusername/dataset-creator}}}}
}}
```

## License

[Specify your license here]
"""

    def create_push_script(
        self,
        dataset_dir: str,
        repo_name: str,
        organization: Optional[str] = None
    ) -> str:
        """
        Create a script to push the dataset to HuggingFace Hub

        Args:
            dataset_dir: Path to the dataset directory
            repo_name: Repository name on HuggingFace Hub
            organization: Optional organization name

        Returns:
            Path to the push script
        """
        full_repo = f"{organization}/{repo_name}" if organization else repo_name

        script_content = f"""#!/usr/bin/env python3
\"\"\"
Script to push dataset to HuggingFace Hub
\"\"\"

from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# Create repository (if it doesn't exist)
try:
    api.create_repo(
        repo_id="{full_repo}",
        repo_type="dataset",
        private=False
    )
    print(f"Created repository: {full_repo}")
except Exception as e:
    print(f"Repository might already exist: {{e}}")

# Upload files
api.upload_folder(
    folder_path="{dataset_dir}",
    repo_id="{full_repo}",
    repo_type="dataset"
)

print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{full_repo}")
"""

        script_path = Path(dataset_dir) / "push_to_hub.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Make script executable
        script_path.chmod(0o755)

        return str(script_path)
