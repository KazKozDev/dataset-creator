"""
Dataset Versioning System
Track changes, manage versions, and enable rollback functionality
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib


class DatasetVersion:
    """Represents a single version of a dataset"""

    def __init__(
        self,
        version_id: str,
        dataset_id: int,
        version_number: int,
        file_path: str,
        created_at: str,
        created_by: str = "system",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None
    ):
        self.version_id = version_id
        self.dataset_id = dataset_id
        self.version_number = version_number
        self.file_path = file_path
        self.created_at = created_at
        self.created_by = created_by
        self.description = description
        self.metadata = metadata or {}
        self.parent_version = parent_version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'dataset_id': self.dataset_id,
            'version_number': self.version_number,
            'file_path': self.file_path,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'description': self.description,
            'metadata': self.metadata,
            'parent_version': self.parent_version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create from dictionary"""
        return cls(**data)


class VersionManager:
    """Manage dataset versions"""

    def __init__(self, versions_dir: str = "data/versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # Version registry file
        self.registry_file = self.versions_dir / "registry.json"
        self.registry = self.load_registry()

    def load_registry(self) -> Dict[str, Any]:
        """Load version registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry: {e}")
                return {'datasets': {}}
        return {'datasets': {}}

    def save_registry(self):
        """Save version registry"""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")

    def generate_version_id(self, dataset_id: int, version_number: int) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().isoformat()
        data = f"{dataset_id}_{version_number}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_dataset_versions_dir(self, dataset_id: int) -> Path:
        """Get directory for dataset versions"""
        dataset_dir = self.versions_dir / str(dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def create_version(
        self,
        dataset_id: int,
        file_path: str,
        description: str = "",
        created_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        parent_version: Optional[str] = None
    ) -> DatasetVersion:
        """Create a new version of a dataset"""
        # Get dataset registry
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            self.registry['datasets'][dataset_key] = {
                'versions': [],
                'current_version': None,
                'latest_version_number': 0
            }

        dataset_registry = self.registry['datasets'][dataset_key]

        # Increment version number
        version_number = dataset_registry['latest_version_number'] + 1

        # Generate version ID
        version_id = self.generate_version_id(dataset_id, version_number)

        # Create version directory
        version_dir = self.get_dataset_versions_dir(dataset_id) / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset file to version directory
        source_path = Path(file_path)
        dest_path = version_dir / source_path.name

        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            print(f"Error copying file to version directory: {e}")
            raise

        # Create version object
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            file_path=str(dest_path),
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            metadata=metadata,
            parent_version=parent_version
        )

        # Update registry
        dataset_registry['versions'].append(version.to_dict())
        dataset_registry['latest_version_number'] = version_number
        dataset_registry['current_version'] = version_id

        # Save registry
        self.save_registry()

        return version

    def get_version(self, dataset_id: int, version_id: str) -> Optional[DatasetVersion]:
        """Get a specific version"""
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            return None

        dataset_registry = self.registry['datasets'][dataset_key]

        for version_data in dataset_registry['versions']:
            if version_data['version_id'] == version_id:
                return DatasetVersion.from_dict(version_data)

        return None

    def get_version_by_number(self, dataset_id: int, version_number: int) -> Optional[DatasetVersion]:
        """Get a version by its number"""
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            return None

        dataset_registry = self.registry['datasets'][dataset_key]

        for version_data in dataset_registry['versions']:
            if version_data['version_number'] == version_number:
                return DatasetVersion.from_dict(version_data)

        return None

    def list_versions(self, dataset_id: int) -> List[DatasetVersion]:
        """List all versions of a dataset"""
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            return []

        dataset_registry = self.registry['datasets'][dataset_key]
        versions = []

        for version_data in dataset_registry['versions']:
            versions.append(DatasetVersion.from_dict(version_data))

        # Sort by version number
        versions.sort(key=lambda v: v.version_number, reverse=True)

        return versions

    def get_current_version(self, dataset_id: int) -> Optional[DatasetVersion]:
        """Get the current (latest) version"""
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            return None

        dataset_registry = self.registry['datasets'][dataset_key]
        current_version_id = dataset_registry.get('current_version')

        if not current_version_id:
            return None

        return self.get_version(dataset_id, current_version_id)

    def set_current_version(self, dataset_id: int, version_id: str) -> bool:
        """Set a specific version as current (rollback)"""
        # Verify version exists
        version = self.get_version(dataset_id, version_id)
        if not version:
            return False

        # Update registry
        dataset_key = str(dataset_id)
        self.registry['datasets'][dataset_key]['current_version'] = version_id

        # Save registry
        self.save_registry()

        return True

    def delete_version(self, dataset_id: int, version_id: str) -> bool:
        """Delete a specific version"""
        dataset_key = str(dataset_id)
        if dataset_key not in self.registry['datasets']:
            return False

        dataset_registry = self.registry['datasets'][dataset_key]

        # Don't allow deleting current version
        if dataset_registry.get('current_version') == version_id:
            print("Cannot delete current version")
            return False

        # Find and remove version from registry
        version_data = None
        for i, v in enumerate(dataset_registry['versions']):
            if v['version_id'] == version_id:
                version_data = dataset_registry['versions'].pop(i)
                break

        if not version_data:
            return False

        # Delete version directory
        version_dir = self.get_dataset_versions_dir(dataset_id) / version_id
        if version_dir.exists():
            try:
                shutil.rmtree(version_dir)
            except Exception as e:
                print(f"Error deleting version directory: {e}")

        # Save registry
        self.save_registry()

        return True

    def compare_versions(
        self,
        dataset_id: int,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """Compare two versions of a dataset"""
        version_1 = self.get_version(dataset_id, version_id_1)
        version_2 = self.get_version(dataset_id, version_id_2)

        if not version_1 or not version_2:
            return {'error': 'One or both versions not found'}

        # Load examples from both versions
        examples_1 = self.load_version_examples(version_1.file_path)
        examples_2 = self.load_version_examples(version_2.file_path)

        # Calculate differences
        diff = {
            'version_1': {
                'version_id': version_1.version_id,
                'version_number': version_1.version_number,
                'created_at': version_1.created_at,
                'example_count': len(examples_1)
            },
            'version_2': {
                'version_id': version_2.version_id,
                'version_number': version_2.version_number,
                'created_at': version_2.created_at,
                'example_count': len(examples_2)
            },
            'differences': {
                'example_count_diff': len(examples_2) - len(examples_1),
                'added_examples': max(0, len(examples_2) - len(examples_1)),
                'removed_examples': max(0, len(examples_1) - len(examples_2))
            }
        }

        return diff

    def load_version_examples(self, file_path: str) -> List[Dict[str, Any]]:
        """Load examples from a version file"""
        examples = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
        except Exception as e:
            print(f"Error loading version examples: {e}")

        return examples

    def get_version_history(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Get version history for a dataset"""
        versions = self.list_versions(dataset_id)

        history = []
        for version in versions:
            history.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'created_at': version.created_at,
                'created_by': version.created_by,
                'description': version.description,
                'is_current': version.version_id == self.get_current_version(dataset_id).version_id if self.get_current_version(dataset_id) else False
            })

        return history

    def create_snapshot(
        self,
        dataset_id: int,
        file_path: str,
        operation: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DatasetVersion:
        """Create a snapshot before an operation"""
        # Get current version as parent
        current = self.get_current_version(dataset_id)
        parent_version = current.version_id if current else None

        # Add operation info to metadata
        if metadata is None:
            metadata = {}

        metadata['operation'] = operation
        metadata['timestamp'] = datetime.now().isoformat()

        # Create version
        version = self.create_version(
            dataset_id=dataset_id,
            file_path=file_path,
            description=description or f"Snapshot before {operation}",
            created_by="system",
            metadata=metadata,
            parent_version=parent_version
        )

        return version

    def auto_version_on_change(
        self,
        dataset_id: int,
        file_path: str,
        change_description: str
    ) -> Optional[DatasetVersion]:
        """Automatically create a version when dataset changes"""
        # Check if file exists
        if not os.path.exists(file_path):
            return None

        # Create version
        version = self.create_version(
            dataset_id=dataset_id,
            file_path=file_path,
            description=f"Auto-version: {change_description}",
            created_by="auto",
            metadata={'auto_versioned': True}
        )

        return version


# Singleton instance
version_manager = VersionManager()


def create_dataset_version(
    dataset_id: int,
    file_path: str,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> DatasetVersion:
    """Convenience function to create dataset version"""
    return version_manager.create_version(
        dataset_id=dataset_id,
        file_path=file_path,
        description=description,
        metadata=metadata
    )


def get_dataset_version_history(dataset_id: int) -> List[Dict[str, Any]]:
    """Convenience function to get version history"""
    return version_manager.get_version_history(dataset_id)


def rollback_to_version(dataset_id: int, version_id: str) -> bool:
    """Convenience function to rollback to a version"""
    return version_manager.set_current_version(dataset_id, version_id)
