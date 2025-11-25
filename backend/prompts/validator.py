"""
Prompt Template Validator

Validates prompt templates and their outputs
"""

from typing import Dict, List, Any, Optional, Tuple
from .schema import PromptTemplate, PromptVariable, VariableType
import re
import json


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class PromptValidator:
    """Validator for prompt templates and their outputs"""

    @staticmethod
    def validate_template_structure(template_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a template dictionary

        Args:
            template_data: Template data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields
        if 'metadata' not in template_data:
            errors.append("Missing 'metadata' field")
        else:
            metadata = template_data['metadata']

            if 'name' not in metadata:
                errors.append("Missing 'metadata.name' field")
            if 'version' not in metadata:
                errors.append("Missing 'metadata.version' field")

        if 'user_prompt' not in template_data:
            errors.append("Missing 'user_prompt' field")

        # Check variables if present
        if 'variables' in template_data:
            if not isinstance(template_data['variables'], list):
                errors.append("'variables' must be a list")
            else:
                for idx, var in enumerate(template_data['variables']):
                    if not isinstance(var, dict):
                        errors.append(f"Variable at index {idx} must be a dictionary")
                        continue

                    if 'name' not in var:
                        errors.append(f"Variable at index {idx} missing 'name' field")

                    if 'type' in var:
                        if var['type'] not in ['string', 'integer', 'float', 'boolean', 'enum', 'array']:
                            errors.append(f"Variable '{var.get('name', idx)}' has invalid type '{var['type']}'")

                        # Enum must have enum_values
                        if var['type'] == 'enum' and 'enum_values' not in var:
                            errors.append(f"Enum variable '{var.get('name', idx)}' missing 'enum_values'")

        # Check temperature range
        if 'temperature' in template_data:
            temp = template_data['temperature']
            if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                errors.append(f"'temperature' must be a number between 0.0 and 2.0, got {temp}")

        # Check max_tokens range
        if 'max_tokens' in template_data:
            max_tokens = template_data['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 32000:
                errors.append(f"'max_tokens' must be an integer between 1 and 32000, got {max_tokens}")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_variable_references(template: PromptTemplate) -> Tuple[bool, List[str]]:
        """
        Validate that all variable references in the prompt exist in the variables list

        Args:
            template: Template to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Extract variable names from template
        defined_vars = {var.name for var in template.variables}

        # Find all {{variable}} references in user_prompt
        user_prompt_vars = set(re.findall(r'\{\{(\w+)\}\}', template.user_prompt))

        # Find all {{variable}} references in system_prompt if present
        system_prompt_vars = set()
        if template.system_prompt:
            system_prompt_vars = set(re.findall(r'\{\{(\w+)\}\}', template.system_prompt))

        # Check for undefined variables
        all_referenced_vars = user_prompt_vars | system_prompt_vars
        undefined_vars = all_referenced_vars - defined_vars

        if undefined_vars:
            errors.append(f"Undefined variables referenced: {', '.join(sorted(undefined_vars))}")

        # Check for unused variables (warning, not error)
        unused_vars = defined_vars - all_referenced_vars
        if unused_vars:
            errors.append(f"Warning: Defined but unused variables: {', '.join(sorted(unused_vars))}")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_variable_values(template: PromptTemplate, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that provided variable values match the template definition

        Args:
            template: Template with variable definitions
            variables: Variable values to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            template.validate_variables(variables)
        except (ValueError, TypeError) as e:
            errors.append(str(e))

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_output_format(output: str, expected_format: str = "text") -> Tuple[bool, List[str]]:
        """
        Validate that the generated output matches the expected format

        Args:
            output: Generated output text
            expected_format: Expected format (text, json, markdown, etc.)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not output or not output.strip():
            errors.append("Output is empty")
            return (False, errors)

        if expected_format == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON format: {e}")

        elif expected_format == "markdown":
            # Basic markdown validation (check for common elements)
            if not any(marker in output for marker in ['#', '*', '-', '```', '**']):
                errors.append("Output doesn't appear to be valid markdown (no common markdown elements found)")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_output_length(output: str, min_length: Optional[int] = None, max_length: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Validate output length constraints

        Args:
            output: Generated output text
            min_length: Minimum required length (characters)
            max_length: Maximum allowed length (characters)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        output_length = len(output)

        if min_length and output_length < min_length:
            errors.append(f"Output too short: {output_length} characters (minimum: {min_length})")

        if max_length and output_length > max_length:
            errors.append(f"Output too long: {output_length} characters (maximum: {max_length})")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_output_content(output: str, forbidden_words: Optional[List[str]] = None, required_words: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate output content for specific words or phrases

        Args:
            output: Generated output text
            forbidden_words: List of words/phrases that should not appear
            required_words: List of words/phrases that must appear

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        output_lower = output.lower()

        if forbidden_words:
            found_forbidden = [word for word in forbidden_words if word.lower() in output_lower]
            if found_forbidden:
                errors.append(f"Output contains forbidden words: {', '.join(found_forbidden)}")

        if required_words:
            missing_required = [word for word in required_words if word.lower() not in output_lower]
            if missing_required:
                errors.append(f"Output missing required words: {', '.join(missing_required)}")

        return (len(errors) == 0, errors)

    @staticmethod
    def check_prompt_injection(user_input: str) -> Tuple[bool, List[str]]:
        """
        Check for potential prompt injection attacks in user input

        Args:
            user_input: User-provided input to check

        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []

        # Common prompt injection patterns
        injection_patterns = [
            (r'ignore\s+(previous|above|prior)\s+instructions', "Potential instruction override attempt"),
            (r'disregard\s+(previous|above|prior)', "Potential instruction override attempt"),
            (r'system\s*:\s*', "Potential system message injection"),
            (r'<\|im_start\|>', "Potential chat format injection"),
            (r'<\|im_end\|>', "Potential chat format injection"),
            (r'\[INST\]', "Potential instruction injection"),
            (r'\[/INST\]', "Potential instruction injection"),
            (r'{{.*}}', "Potential template variable injection"),
        ]

        for pattern, message in injection_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                warnings.append(message)

        return (len(warnings) == 0, warnings)

    @staticmethod
    def validate_examples(examples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate few-shot examples format

        Args:
            examples: List of example dictionaries

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not isinstance(examples, list):
            errors.append("Examples must be a list")
            return (False, errors)

        for idx, example in enumerate(examples):
            if not isinstance(example, dict):
                errors.append(f"Example at index {idx} must be a dictionary")
                continue

            # Check for common fields
            if 'input' not in example and 'output' not in example:
                errors.append(f"Example at index {idx} should have 'input' and/or 'output' fields")

        return (len(errors) == 0, errors)

    @classmethod
    def validate_full_template(cls, template: PromptTemplate) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Run all validation checks on a template

        Args:
            template: Template to validate

        Returns:
            Tuple of (is_valid, dict_of_errors_by_category)
        """
        all_errors = {}

        # Validate variable references
        is_valid, errors = cls.validate_variable_references(template)
        if errors:
            all_errors['variable_references'] = errors

        # Validate examples
        if template.examples:
            is_valid, errors = cls.validate_examples(template.examples)
            if errors:
                all_errors['examples'] = errors

        is_overall_valid = len(all_errors) == 0

        return (is_overall_valid, all_errors)


# Convenience functions
def validate_template(template: PromptTemplate) -> bool:
    """
    Validate a template (convenience function)

    Args:
        template: Template to validate

    Returns:
        True if valid

    Raises:
        ValidationError if invalid
    """
    validator = PromptValidator()
    is_valid, errors = validator.validate_full_template(template)

    if not is_valid:
        error_messages = []
        for category, category_errors in errors.items():
            error_messages.append(f"{category}: {'; '.join(category_errors)}")

        raise ValidationError("\n".join(error_messages))

    return True


def validate_template_dict(template_data: Dict[str, Any]) -> bool:
    """
    Validate a template dictionary (convenience function)

    Args:
        template_data: Template data to validate

    Returns:
        True if valid

    Raises:
        ValidationError if invalid
    """
    validator = PromptValidator()
    is_valid, errors = validator.validate_template_structure(template_data)

    if not is_valid:
        raise ValidationError("; ".join(errors))

    return True
