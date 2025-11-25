"""
User Management System

Manages users and roles for team collaboration
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: str
    role: UserRole
    created_at: str
    last_active: str
    metadata: Dict


class UserManager:
    """Manager for users"""

    def __init__(self, users_file: str = "./data/collaboration/users.json"):
        """
        Initialize user manager

        Args:
            users_file: Path to users storage file
        """
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_users()

    def _load_users(self):
        """Load users from file"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                data = json.load(f)
                self.users = {
                    uid: User(**user_data)
                    for uid, user_data in data.items()
                }
        else:
            self.users = {}
            self._save_users()

    def _save_users(self):
        """Save users to file"""
        data = {
            uid: asdict(user)
            for uid, user in self.users.items()
        }

        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_user(
        self,
        username: str,
        email: str,
        role: UserRole = UserRole.VIEWER,
        user_id: Optional[str] = None
    ) -> User:
        """
        Create a new user

        Args:
            username: Username
            email: Email address
            role: User role
            user_id: Optional custom user ID

        Returns:
            Created User object
        """
        if user_id is None:
            user_id = f"user_{len(self.users) + 1}"

        # Check if username or email already exists
        for user in self.users.values():
            if user.username == username:
                raise ValueError(f"Username '{username}' already exists")
            if user.email == email:
                raise ValueError(f"Email '{email}' already exists")

        now = datetime.now().isoformat()

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            created_at=now,
            last_active=now,
            metadata={}
        )

        self.users[user_id] = user
        self._save_users()

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID

        Args:
            user_id: User ID

        Returns:
            User or None if not found
        """
        return self.users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username

        Args:
            username: Username

        Returns:
            User or None if not found
        """
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email

        Args:
            email: Email address

        Returns:
            User or None if not found
        """
        for user in self.users.values():
            if user.email == email:
                return user
        return None

    def list_users(
        self,
        role: Optional[UserRole] = None
    ) -> List[User]:
        """
        List all users

        Args:
            role: Optional role filter

        Returns:
            List of User objects
        """
        users = list(self.users.values())

        if role:
            users = [u for u in users if u.role == role]

        return users

    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[UserRole] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update user information

        Args:
            user_id: User ID
            username: New username
            email: New email
            role: New role
            metadata: New metadata

        Returns:
            True if updated successfully
        """
        user = self.users.get(user_id)

        if not user:
            return False

        if username:
            user.username = username
        if email:
            user.email = email
        if role:
            user.role = role
        if metadata:
            user.metadata.update(metadata)

        user.last_active = datetime.now().isoformat()

        self._save_users()
        return True

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user

        Args:
            user_id: User ID

        Returns:
            True if deleted successfully
        """
        if user_id in self.users:
            del self.users[user_id]
            self._save_users()
            return True
        return False

    def update_last_active(self, user_id: str) -> bool:
        """
        Update user's last active timestamp

        Args:
            user_id: User ID

        Returns:
            True if updated successfully
        """
        user = self.users.get(user_id)

        if user:
            user.last_active = datetime.now().isoformat()
            self._save_users()
            return True

        return False

    def get_statistics(self) -> Dict:
        """
        Get user statistics

        Returns:
            Statistics dictionary
        """
        role_counts = {}
        for role in UserRole:
            role_counts[role.value] = sum(
                1 for u in self.users.values() if u.role == role
            )

        return {
            "total_users": len(self.users),
            "role_breakdown": role_counts,
            "active_users": len([
                u for u in self.users.values()
                if self._is_recently_active(u)
            ])
        }

    def _is_recently_active(self, user: User, days: int = 7) -> bool:
        """Check if user was active recently"""
        try:
            last_active = datetime.fromisoformat(user.last_active)
            now = datetime.now()
            diff = (now - last_active).days
            return diff <= days
        except:
            return False


# Global user manager instance
_user_manager_instance: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get the global user manager instance"""
    global _user_manager_instance

    if _user_manager_instance is None:
        _user_manager_instance = UserManager()

    return _user_manager_instance
