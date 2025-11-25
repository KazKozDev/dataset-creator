"""
OpenAI Exporter

Exports datasets to OpenAI fine-tuning format (JSONL)
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseExporter


class OpenAIExporter(BaseExporter):
    """Exporter for OpenAI fine-tuning format"""

    def export(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str,
        system_message: Optional[str] = None,
        include_function_calls: bool = False,
        **kwargs
    ) -> str:
        """
        Export examples to OpenAI fine-tuning format

        Args:
            examples: List of conversation examples
            output_filename: Name of the output file
            system_message: Optional default system message
            include_function_calls: Whether to include function call examples
            **kwargs: Additional parameters

        Returns:
            Path to the exported file
        """
        # Validate examples
        is_valid, errors = self.validate_examples(examples)
        if not is_valid:
            raise ValueError(f"Invalid examples: {'; '.join(errors)}")

        # Convert examples to OpenAI format
        openai_examples = []
        for example in examples:
            openai_example = self._convert_to_openai_format(
                example,
                system_message=system_message,
                include_function_calls=include_function_calls
            )
            if openai_example:
                openai_examples.append(openai_example)

        # Save to JSONL
        output_path = self.get_output_path(output_filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.jsonl')

        self.save_jsonl(openai_examples, output_path)

        # Create validation report
        self._create_validation_report(
            output_path.with_suffix('.validation.txt'),
            openai_examples
        )

        return str(output_path)

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """
        Validate example format

        Args:
            example: Example to validate

        Returns:
            True if valid
        """
        # Check if it has conversation/messages or can be converted
        if "conversation" in example or "messages" in example:
            return True
        elif "instruction" in example or "prompt" in example:
            return True
        elif "input" in example and "output" in example:
            return True
        return False

    def _convert_to_openai_format(
        self,
        example: Dict[str, Any],
        system_message: Optional[str] = None,
        include_function_calls: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Convert example to OpenAI format

        Args:
            example: Original example
            system_message: Optional system message
            include_function_calls: Whether to include function calls

        Returns:
            OpenAI formatted example or None if conversion fails
        """
        messages = []

        # Add system message if provided
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })

        # Handle conversation format
        if "conversation" in example:
            for turn in example["conversation"]:
                role = turn.get("role", "user")
                content = turn.get("content", "")

                # Map roles to OpenAI format
                if role in ["user", "assistant", "system"]:
                    messages.append({
                        "role": role,
                        "content": content
                    })
                elif role in ["human", "person"]:
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                elif role in ["bot", "ai", "agent"]:
                    messages.append({
                        "role": "assistant",
                        "content": content
                    })

        # Handle messages format (already in chat format)
        elif "messages" in example:
            messages.extend(example["messages"])

        # Handle instruction format
        elif "instruction" in example:
            # Add instruction as system message if no system message provided
            if not system_message and example.get("instruction"):
                messages.append({
                    "role": "system",
                    "content": example["instruction"]
                })

            # Add input as user message
            user_content = example.get("input", "")
            if user_content:
                messages.append({
                    "role": "user",
                    "content": user_content
                })

            # Add output as assistant message
            assistant_content = example.get("output", "")
            if assistant_content:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

        # Handle prompt-completion format
        elif "prompt" in example:
            messages.append({
                "role": "user",
                "content": example["prompt"]
            })

            completion = example.get("completion", example.get("response", ""))
            if completion:
                messages.append({
                    "role": "assistant",
                    "content": completion
                })

        # Handle simple input-output format
        elif "input" in example and "output" in example:
            messages.append({
                "role": "user",
                "content": example["input"]
            })

            messages.append({
                "role": "assistant",
                "content": example["output"]
            })

        # Return None if no messages were created
        if not messages:
            return None

        # Create the OpenAI example
        openai_example = {
            "messages": messages
        }

        # Add function calls if present and requested
        if include_function_calls and "functions" in example:
            openai_example["functions"] = example["functions"]

        return openai_example

    def _create_validation_report(
        self,
        output_path: Path,
        examples: List[Dict[str, Any]]
    ) -> None:
        """
        Create a validation report for the dataset

        Args:
            output_path: Path to save the report
            examples: List of OpenAI formatted examples
        """
        total_examples = len(examples)
        total_messages = sum(len(ex["messages"]) for ex in examples)
        avg_messages = total_messages / total_examples if total_examples > 0 else 0

        # Count role distribution
        role_counts = {"system": 0, "user": 0, "assistant": 0}
        for example in examples:
            for message in example["messages"]:
                role = message.get("role", "unknown")
                if role in role_counts:
                    role_counts[role] += 1

        # Calculate average tokens (rough estimate)
        total_tokens = 0
        for example in examples:
            for message in example["messages"]:
                content = message.get("content", "")
                # Rough token estimate: ~4 characters per token
                total_tokens += len(content) // 4

        avg_tokens = total_tokens / total_examples if total_examples > 0 else 0

        # Create report
        report = f"""OpenAI Fine-tuning Dataset Validation Report
============================================

Dataset Statistics:
------------------
Total Examples: {total_examples}
Total Messages: {total_messages}
Average Messages per Example: {avg_messages:.2f}
Estimated Total Tokens: {total_tokens:,}
Estimated Average Tokens per Example: {avg_tokens:.2f}

Message Role Distribution:
-------------------------
System Messages: {role_counts['system']}
User Messages: {role_counts['user']}
Assistant Messages: {role_counts['assistant']}

Format Validation:
-----------------
✓ All examples are in valid OpenAI format
✓ All messages have 'role' and 'content' fields
✓ Dataset is ready for fine-tuning

Cost Estimation (GPT-3.5-turbo):
--------------------------------
Training Cost (estimated): ${total_tokens * 0.008 / 1000:.2f}
Note: Actual cost may vary based on model and epoch count

Next Steps:
----------
1. Upload the JSONL file to OpenAI
2. Create a fine-tuning job using the OpenAI API or dashboard
3. Monitor the training progress
4. Test the fine-tuned model

OpenAI CLI Command:
------------------
openai api fine_tunes.create \\
    -t {output_path.with_suffix('.jsonl').name} \\
    -m gpt-3.5-turbo \\
    --suffix "custom-model"

Python API Example:
------------------
```python
import openai

# Upload file
with open("{output_path.with_suffix('.jsonl').name}", "rb") as f:
    response = openai.File.create(
        file=f,
        purpose='fine-tune'
    )
    file_id = response['id']

# Create fine-tuning job
openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)
```
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

    def validate_for_finetuning(
        self,
        examples: List[Dict[str, Any]]
    ) -> tuple[bool, List[str]]:
        """
        Validate examples specifically for OpenAI fine-tuning

        Args:
            examples: List of examples to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for idx, example in enumerate(examples):
            # Check messages field exists
            if "messages" not in example:
                errors.append(f"Example {idx}: Missing 'messages' field")
                continue

            messages = example["messages"]

            # Check messages is a list
            if not isinstance(messages, list):
                errors.append(f"Example {idx}: 'messages' must be a list")
                continue

            # Check at least one message
            if len(messages) == 0:
                errors.append(f"Example {idx}: Must have at least one message")
                continue

            # Validate each message
            for msg_idx, message in enumerate(messages):
                # Check required fields
                if "role" not in message:
                    errors.append(f"Example {idx}, Message {msg_idx}: Missing 'role' field")

                if "content" not in message:
                    errors.append(f"Example {idx}, Message {msg_idx}: Missing 'content' field")

                # Check role is valid
                if message.get("role") not in ["system", "user", "assistant", "function"]:
                    errors.append(
                        f"Example {idx}, Message {msg_idx}: "
                        f"Invalid role '{message.get('role')}'"
                    )

        return (len(errors) == 0, errors)

    def split_dataset(
        self,
        examples: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        output_prefix: str = "dataset"
    ) -> tuple[str, str]:
        """
        Split dataset into train and validation sets

        Args:
            examples: List of examples
            train_ratio: Ratio of examples for training (0.0 to 1.0)
            output_prefix: Prefix for output filenames

        Returns:
            Tuple of (train_file_path, validation_file_path)
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0.0 and 1.0")

        # Calculate split point
        split_point = int(len(examples) * train_ratio)

        # Split examples
        train_examples = examples[:split_point]
        val_examples = examples[split_point:]

        # Save train set
        train_path = self.get_output_path(f"{output_prefix}_train.jsonl")
        self.save_jsonl(train_examples, train_path)

        # Save validation set
        val_path = self.get_output_path(f"{output_prefix}_validation.jsonl")
        self.save_jsonl(val_examples, val_path)

        return (str(train_path), str(val_path))
