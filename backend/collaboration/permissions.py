"""
Permission Management System

Manages dataset permissions and access control
"""

import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class PermissionLevel(str, Enum):
    """Permission levels"""
    OWNER = "owner"
    WRITE = "write"
    READ = "read"
    NONE = "none"


@dataclass
class DatasetPermission:
    """Dataset permission model"""
    dataset_id: int
    user_id: str
    permission_level: PermissionLevel
    granted_by: str
    granted_at: str


class PermissionManager:
    """Manager for dataset permissions"""

    def __init__(self, permissions_file: str = "./data/collaboration/permissions.json"):
        """
        Initialize permission manager

        Args:
            permissions_file: Path to permissions storage file
        """
        self.permissions_file = Path(permissions_file)
        self.permissions_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_permissions()

    def _load_permissions(self):
        """Load permissions from file"""
        if self.permissions_file.exists():
            with open(self.permissions_file, 'r') as f:
                data = json.load(f)
                self.permissions = [
                    DatasetPermission(**perm_data)
                    for perm_data in data
                ]
        else:
            self.permissions = []
            self._save_permissions()

    def _save_permissions(self):
        """Save permissions to file"""
        data = [asdict(perm) for perm in self.permissions]

        with open(self.permissions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def grant_permission(
        self,
        dataset_id: int,
        user_id: str,
        permission_level: PermissionLevel,
        granted_by: str
    ) -> DatasetPermission:
        """
        Grant permission to a user for a dataset

        Args:
            dataset_id: Dataset ID
            user_id: User ID
            permission_level: Permission level
            granted_by: ID of user granting permission

        Returns:
            Created DatasetPermission object
        """
        # Remove existing permission if any
        self.revoke_permission(dataset_id, user_id)

        permission = DatasetPermission(
            dataset_id=dataset_id,
            user_id=user_id,
            permission_level=permission_level,
            granted_by=granted_by,
            granted_at=datetime.now().isoformat()
        )

        self.permissions.append(permission)
        self._save_permissions()

        return permission

    def revoke_permission(
        self,
        dataset_id: int,
        user_id: str
    ) -> bool:
        """
        Revoke user's permission for a dataset

        Args:
            dataset_id: Dataset ID
            user_id: User ID

        Returns:
            True if revoked successfully
        """
        initial_count = len(self.permissions)

        self.permissions = [
            p for p in self.permissions
            if not (p.dataset_id == dataset_id and p.user_id == user_id)
        ]

        if len(self.permissions) < initial_count:
            self._save_permissions()
            return True

        return False

    def get_permission(
        self,
        dataset_id: int,
        user_id: str
    ) -> Optional[DatasetPermission]:
        """
        Get user's permission for a dataset

        Args:
            dataset_id: Dataset ID
            user_id: User ID

        Returns:
            DatasetPermission or None if not found
        """
        for perm in self.permissions:
            if perm.dataset_id == dataset_id and perm.user_id == user_id:
                return perm

        return None

    def check_permission(
        self,
        dataset_id: int,
        user_id: str,
        required_level: PermissionLevel
    ) -> bool:
        """
        Check if user has required permission level

        Args:
            dataset_id: Dataset ID
            user_id: User ID
            required_level: Required permission level

        Returns:
            True if user has required permission
        """
        permission = self.get_permission(dataset_id, user_id)

        if not permission:
            return False

        # Permission hierarchy: OWNER > WRITE > READ > NONE
        levels = {
            PermissionLevel.OWNER: 3,
            PermissionLevel.WRITE: 2,
            PermissionLevel.READ: 1,
            PermissionLevel.NONE: 0
        }

        user_level = levels.get(permission.permission_level, 0)
        required = levels.get(required_level, 0)

        return user_level >= required

    def list_dataset_permissions(
        self,
        dataset_id: int
    ) -> List[DatasetPermission]:
        """
        List all permissions for a dataset

        Args:
            dataset_id: Dataset ID

        Returns:
            List of DatasetPermission objects
        """
        return [
            p for p in self.permissions
            if p.dataset_id == dataset_id
        ]

    def list_user_permissions(
        self,
        user_id: str
    ) -> List[DatasetPermission]:
        """
        List all permissions for a user

        Args:
            user_id: User ID

        Returns:
            List of DatasetPermission objects
        """
        return [
            p for p in self.permissions
            if p.user_id == user_id
        ]

    def get_accessible_datasets(
        self,
        user_id: str,
        min_level: PermissionLevel = PermissionLevel.READ
    ) -> List[int]:
        """
        Get list of dataset IDs accessible to user

        Args:
            user_id: User ID
            min_level: Minimum permission level required

        Returns:
            List of dataset IDs
        """
        levels = {
            PermissionLevel.OWNER: 3,
            PermissionLevel.WRITE: 2,
            PermissionLevel.READ: 1,
            PermissionLevel.NONE: 0
        }

        min_required = levels.get(min_level, 0)

        dataset_ids = []
        for perm in self.permissions:
            if perm.user_id == user_id:
                user_level = levels.get(perm.permission_level, 0)
                if user_level >= min_required:
                    dataset_ids.append(perm.dataset_id)

        return dataset_ids

    def share_dataset(
        self,
        dataset_id: int,
        from_user_id: str,
        to_user_id: str,
        permission_level: PermissionLevel = PermissionLevel.READ
    ) -> Optional[DatasetPermission]:
        """
        Share a dataset with another user

        Args:
            dataset_id: Dataset ID
            from_user_id: User sharing the dataset
            to_user_id: User receiving access
            permission_level: Permission level to grant

        Returns:
            DatasetPermission or None if sharing user doesn't have permission
        """
        # Check if sharing user has WRITE or OWNER permission
        if not self.check_permission(dataset_id, from_user_id, PermissionLevel.WRITE):
            return None

        # Grant permission
        return self.grant_permission(
            dataset_id=dataset_id,
            user_id=to_user_id,
            permission_level=permission_level,
            granted_by=from_user_id
        )

    def transfer_ownership(
        self,
        dataset_id: int,
        from_user_id: str,
        to_user_id: str
    ) -> bool:
        """
        Transfer dataset ownership to another user

        Args:
            dataset_id: Dataset ID
            from_user_id: Current owner
            to_user_id: New owner

        Returns:
            True if transferred successfully
        """
        # Check if from_user is owner
        perm = self.get_permission(dataset_id, from_user_id)
        if not perm or perm.permission_level != PermissionLevel.OWNER:
            return False

        # Grant owner permission to new user
        self.grant_permission(
            dataset_id=dataset_id,
            user_id=to_user_id,
            permission_level=PermissionLevel.OWNER,
            granted_by=from_user_id
        )

        # Downgrade old owner to WRITE
        self.grant_permission(
            dataset_id=dataset_id,
            user_id=from_user_id,
            permission_level=PermissionLevel.WRITE,
            granted_by=to_user_id
        )

        return True

    def get_statistics(self) -> Dict:
        """
        Get permission statistics

        Returns:
            Statistics dictionary
        """
        # Count by permission level
        level_counts = {}
        for level in PermissionLevel:
            level_counts[level.value] = sum(
                1 for p in self.permissions if p.permission_level == level
            )

        # Count unique datasets
        unique_datasets = len(set(p.dataset_id for p in self.permissions))

        # Count unique users
        unique_users = len(set(p.user_id for p in self.permissions))

        return {
            "total_permissions": len(self.permissions),
            "level_breakdown": level_counts,
            "unique_datasets": unique_datasets,
            "unique_users": unique_users
        }


# Global permission manager instance
_permission_manager_instance: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """Get the global permission manager instance"""
    global _permission_manager_instance

    if _permission_manager_instance is None:
        _permission_manager_instance = PermissionManager()

    return _permission_manager_instance
