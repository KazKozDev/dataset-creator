"""
Alpaca/ShareGPT Exporter

Exports datasets to Alpaca and ShareGPT formats for instruction tuning
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseExporter


class AlpacaExporter(BaseExporter):
    """Exporter for Alpaca instruction format"""

    def export(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str,
        format_type: str = "alpaca",
        **kwargs
    ) -> str:
        """
        Export examples to Alpaca or ShareGPT format

        Args:
            examples: List of conversation examples
            output_filename: Name of the output file
            format_type: "alpaca" or "sharegpt"
            **kwargs: Additional parameters

        Returns:
            Path to the exported file
        """
        # Validate examples
        is_valid, errors = self.validate_examples(examples)
        if not is_valid:
            raise ValueError(f"Invalid examples: {'; '.join(errors)}")

        # Convert based on format type
        if format_type.lower() == "alpaca":
            converted = self._convert_to_alpaca(examples)
        elif format_type.lower() == "sharegpt":
            converted = self._convert_to_sharegpt(examples)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Use 'alpaca' or 'sharegpt'")

        # Save to JSON
        output_path = self.get_output_path(output_filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')

        self.save_json(converted, output_path, indent=2)

        return str(output_path)

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate example format"""
        # Check if it has convertible data
        if any(key in example for key in ["conversation", "messages", "instruction", "prompt", "input", "output"]):
            return True
        return False

    def _convert_to_alpaca(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert examples to Alpaca format

        Alpaca format:
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        }
        """
        alpaca_examples = []

        for example in examples:
            # If already in Alpaca format
            if "instruction" in example and "output" in example:
                alpaca_examples.append({
                    "instruction": example["instruction"],
                    "input": example.get("input", ""),
                    "output": example["output"]
                })

            # Convert from conversation/messages format
            elif "conversation" in example or "messages" in example:
                messages = example.get("conversation", example.get("messages", []))

                # Extract system message as instruction if present
                instruction = ""
                user_input = ""
                assistant_output = ""

                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if role == "system":
                        instruction = content
                    elif role in ["user", "human"]:
                        if user_input:
                            user_input += "\n" + content
                        else:
                            user_input = content
                    elif role in ["assistant", "bot", "ai"]:
                        if assistant_output:
                            assistant_output += "\n" + content
                        else:
                            assistant_output = content

                # If no instruction, create generic one
                if not instruction:
                    instruction = "Respond to the following conversation."

                alpaca_examples.append({
                    "instruction": instruction,
                    "input": user_input,
                    "output": assistant_output
                })

            # Convert from prompt-completion format
            elif "prompt" in example:
                alpaca_examples.append({
                    "instruction": "Complete the following prompt.",
                    "input": example["prompt"],
                    "output": example.get("completion", example.get("response", ""))
                })

            # Convert from simple input-output
            elif "input" in example and "output" in example:
                alpaca_examples.append({
                    "instruction": example.get("instruction", "Respond to the input."),
                    "input": example["input"],
                    "output": example["output"]
                })

        return alpaca_examples

    def _convert_to_sharegpt(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert examples to ShareGPT format

        ShareGPT format:
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
        """
        sharegpt_examples = []

        for example in examples:
            conversations = []

            # If already in conversation/messages format
            if "conversation" in example or "messages" in example:
                messages = example.get("conversation", example.get("messages", []))

                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    # Map roles to ShareGPT format
                    if role == "system":
                        # ShareGPT doesn't have system role, prepend to first human message
                        conversations.append({
                            "from": "system",
                            "value": content
                        })
                    elif role in ["user", "human"]:
                        conversations.append({
                            "from": "human",
                            "value": content
                        })
                    elif role in ["assistant", "bot", "ai"]:
                        conversations.append({
                            "from": "gpt",
                            "value": content
                        })

            # Convert from instruction format
            elif "instruction" in example:
                instruction = example["instruction"]
                user_input = example.get("input", "")
                output = example.get("output", "")

                # Combine instruction and input
                human_msg = instruction
                if user_input:
                    human_msg += "\n\n" + user_input

                conversations.append({"from": "human", "value": human_msg})
                conversations.append({"from": "gpt", "value": output})

            # Convert from prompt-completion format
            elif "prompt" in example:
                conversations.append({"from": "human", "value": example["prompt"]})
                conversations.append({
                    "from": "gpt",
                    "value": example.get("completion", example.get("response", ""))
                })

            # Convert from simple input-output
            elif "input" in example and "output" in example:
                conversations.append({"from": "human", "value": example["input"]})
                conversations.append({"from": "gpt", "value": example["output"]})

            if conversations:
                sharegpt_examples.append({"conversations": conversations})

        return sharegpt_examples

    def export_both_formats(
        self,
        examples: List[Dict[str, Any]],
        output_prefix: str
    ) -> tuple[str, str]:
        """
        Export to both Alpaca and ShareGPT formats

        Args:
            examples: List of examples
            output_prefix: Prefix for output filenames

        Returns:
            Tuple of (alpaca_path, sharegpt_path)
        """
        alpaca_path = self.export(examples, f"{output_prefix}_alpaca", format_type="alpaca")
        sharegpt_path = self.export(examples, f"{output_prefix}_sharegpt", format_type="sharegpt")

        return (alpaca_path, sharegpt_path)

    def create_dataset_card(
        self,
        output_dir: str,
        dataset_name: str,
        num_examples: int,
        format_type: str
    ) -> str:
        """
        Create a dataset card README

        Args:
            output_dir: Directory containing the dataset
            dataset_name: Name of the dataset
            num_examples: Number of examples
            format_type: Format type (alpaca or sharegpt)

        Returns:
            Path to README file
        """
        if format_type == "alpaca":
            usage_example = '''```python
import json

# Load the dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Format: [{"instruction": "...", "input": "...", "output": "..."}]
for example in data:
    print(f"Instruction: {example['instruction']}")
    print(f"Input: {example['input']}")
    print(f"Output: {example['output']}")
    print()
```'''
        else:  # sharegpt
            usage_example = '''```python
import json

# Load the dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Format: [{"conversations": [{"from": "human/gpt", "value": "..."}]}]
for example in data:
    for turn in example['conversations']:
        print(f"{turn['from']}: {turn['value']}")
    print()
```'''

        readme_content = f"""# {dataset_name}

## Dataset Description

This dataset is in **{format_type.upper()}** format, suitable for instruction-tuning language models.

### Dataset Summary

- **Total Examples**: {num_examples}
- **Format**: {format_type.upper()}
- **Generated**: Synthetic dataset created with LLM Dataset Creator

## Dataset Structure

### Format: {format_type.upper()}

"""

        if format_type == "alpaca":
            readme_content += """Each example contains three fields:
- `instruction`: The task instruction
- `input`: Additional context or input (can be empty)
- `output`: The expected response

"""
        else:
            readme_content += """Each example contains a `conversations` array with alternating turns:
- `from`: Either "human" or "gpt"
- `value`: The message content

"""

        readme_content += f"""## Usage

{usage_example}

### Training with FastChat

```bash
# For Alpaca format
fastchat train --model-path <model> --data-path dataset.json --data-format alpaca

# For ShareGPT format
fastchat train --model-path <model> --data-path dataset.json --data-format sharegpt
```

### Training with Axolotl

```yaml
datasets:
  - path: dataset.json
    type: {"alpaca" if format_type == "alpaca" else "sharegpt"}
```

## Citation

```bibtex
@misc{{{dataset_name.lower().replace(' ', '_')},
  title={{{dataset_name}}},
  author={{LLM Dataset Creator}},
  year={{2025}}
}}
```

## License

[Specify your license here]
"""

        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        return str(readme_path)
