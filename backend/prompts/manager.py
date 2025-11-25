"""
Prompt Template Manager

Handles loading, saving, and managing prompt templates
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from .schema import PromptTemplate, PromptMetadata


class PromptManager:
    """Manager for prompt templates"""

    def __init__(self, templates_dir: str = "./prompts/templates"):
        """
        Initialize the prompt manager

        Args:
            templates_dir: Directory where template YAML files are stored
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, PromptTemplate] = {}

    def load_template(self, template_name: str, use_cache: bool = True) -> Optional[PromptTemplate]:
        """
        Load a template by name

        Args:
            template_name: Name of the template (without .yaml extension)
            use_cache: Whether to use cached version if available

        Returns:
            PromptTemplate or None if not found
        """
        # Check cache first
        if use_cache and template_name in self._cache:
            return self._cache[template_name]

        # Load from file
        template_path = self.templates_dir / f"{template_name}.yaml"

        if not template_path.exists():
            return None

        try:
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)

            template = PromptTemplate.from_dict(template_data)

            # Cache it
            self._cache[template_name] = template

            return template

        except Exception as e:
            print(f"Error loading template '{template_name}': {e}")
            return None

    def save_template(self, template: PromptTemplate, overwrite: bool = False) -> bool:
        """
        Save a template to disk

        Args:
            template: The template to save
            overwrite: Whether to overwrite existing template

        Returns:
            True if saved successfully
        """
        template_name = template.metadata.name
        template_path = self.templates_dir / f"{template_name}.yaml"

        # Check if exists
        if template_path.exists() and not overwrite:
            print(f"Template '{template_name}' already exists. Use overwrite=True to replace.")
            return False

        try:
            # Update timestamp
            template.metadata.updated_at = datetime.now()

            # Convert to dict and save
            template_data = template.to_dict()

            with open(template_path, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

            # Update cache
            self._cache[template_name] = template

            return True

        except Exception as e:
            print(f"Error saving template '{template_name}': {e}")
            return False

    def list_templates(self, domain: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all available templates

        Args:
            domain: Filter by domain
            tags: Filter by tags (templates must have ALL specified tags)

        Returns:
            List of template metadata dictionaries
        """
        templates = []

        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)

                metadata = template_data.get('metadata', {})

                # Apply filters
                if domain and metadata.get('domain') != domain:
                    continue

                if tags:
                    template_tags = set(metadata.get('tags', []))
                    if not all(tag in template_tags for tag in tags):
                        continue

                templates.append({
                    'name': metadata.get('name'),
                    'version': metadata.get('version'),
                    'description': metadata.get('description'),
                    'author': metadata.get('author'),
                    'domain': metadata.get('domain'),
                    'subdomain': metadata.get('subdomain'),
                    'tags': metadata.get('tags', []),
                    'created_at': metadata.get('created_at'),
                    'updated_at': metadata.get('updated_at')
                })

            except Exception as e:
                print(f"Error reading template file {template_file}: {e}")
                continue

        return templates

    def delete_template(self, template_name: str) -> bool:
        """
        Delete a template

        Args:
            template_name: Name of the template to delete

        Returns:
            True if deleted successfully
        """
        template_path = self.templates_dir / f"{template_name}.yaml"

        if not template_path.exists():
            return False

        try:
            template_path.unlink()

            # Remove from cache
            if template_name in self._cache:
                del self._cache[template_name]

            return True

        except Exception as e:
            print(f"Error deleting template '{template_name}': {e}")
            return False

    def render_template(self, template_name: str, variables: Dict[str, Any]) -> Optional[str]:
        """
        Load and render a template with given variables

        Args:
            template_name: Name of the template
            variables: Variables to substitute

        Returns:
            Rendered prompt or None if template not found
        """
        template = self.load_template(template_name)

        if not template:
            return None

        try:
            # Validate variables
            template.validate_variables(variables)

            # Render prompt
            return template.render(variables)

        except Exception as e:
            print(f"Error rendering template '{template_name}': {e}")
            return None

    def create_template_from_dict(self, template_data: Dict[str, Any]) -> Optional[PromptTemplate]:
        """
        Create a template from a dictionary

        Args:
            template_data: Template data dictionary

        Returns:
            PromptTemplate or None if invalid
        """
        try:
            return PromptTemplate.from_dict(template_data)
        except Exception as e:
            print(f"Error creating template from dict: {e}")
            return None

    def get_template_variables(self, template_name: str) -> List[Dict[str, Any]]:
        """
        Get the variables defined in a template

        Args:
            template_name: Name of the template

        Returns:
            List of variable definitions
        """
        template = self.load_template(template_name)

        if not template:
            return []

        return [var.model_dump() for var in template.variables]

    def clear_cache(self) -> None:
        """Clear the template cache"""
        self._cache.clear()

    def export_template(self, template_name: str, output_path: str) -> bool:
        """
        Export a template to a specific path

        Args:
            template_name: Name of the template
            output_path: Path to export to

        Returns:
            True if exported successfully
        """
        template = self.load_template(template_name)

        if not template:
            return False

        try:
            template_data = template.to_dict()

            with open(output_path, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

            return True

        except Exception as e:
            print(f"Error exporting template '{template_name}': {e}")
            return False

    def import_template(self, import_path: str, overwrite: bool = False) -> Optional[PromptTemplate]:
        """
        Import a template from a YAML file

        Args:
            import_path: Path to the template file
            overwrite: Whether to overwrite existing template

        Returns:
            Imported PromptTemplate or None if failed
        """
        try:
            with open(import_path, 'r') as f:
                template_data = yaml.safe_load(f)

            template = PromptTemplate.from_dict(template_data)

            # Save it
            if self.save_template(template, overwrite=overwrite):
                return template

            return None

        except Exception as e:
            print(f"Error importing template from '{import_path}': {e}")
            return None

    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """
        Search templates by name, description, or tags

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching template metadata
        """
        query_lower = query.lower()
        results = []

        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)

                metadata = template_data.get('metadata', {})

                # Check if query matches name, description, or tags
                name = metadata.get('name', '').lower()
                description = metadata.get('description', '').lower()
                tags = [tag.lower() for tag in metadata.get('tags', [])]

                if (query_lower in name or
                    query_lower in description or
                    any(query_lower in tag for tag in tags)):

                    results.append({
                        'name': metadata.get('name'),
                        'version': metadata.get('version'),
                        'description': metadata.get('description'),
                        'author': metadata.get('author'),
                        'domain': metadata.get('domain'),
                        'subdomain': metadata.get('subdomain'),
                        'tags': metadata.get('tags', []),
                        'created_at': metadata.get('created_at'),
                        'updated_at': metadata.get('updated_at')
                    })

            except Exception as e:
                print(f"Error reading template file {template_file}: {e}")
                continue

        return results


# Global manager instance
_manager_instance: Optional[PromptManager] = None


def get_manager(templates_dir: Optional[str] = None) -> PromptManager:
    """
    Get the global prompt manager instance

    Args:
        templates_dir: Optional templates directory path

    Returns:
        PromptManager instance
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = PromptManager(templates_dir or "./prompts/templates")

    return _manager_instance
