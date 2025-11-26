"""
Team Collaboration System

Provides multi-user support, permissions, comments, and review workflow
"""

from .users import UserManager, User, UserRole
from .permissions import PermissionManager, DatasetPermission, PermissionLevel
from .comments import CommentManager, Comment
from .reviews import ReviewManager, ReviewStatus

# Global instances
_user_manager = UserManager()
_permission_manager = PermissionManager()
_comment_manager = CommentManager()
_review_manager = ReviewManager()

def get_user_manager():
    return _user_manager

def get_permission_manager():
    return _permission_manager

def get_comment_manager():
    return _comment_manager

def get_review_manager():
    return _review_manager

__all__ = [
    "UserManager",
    "User",
    "UserRole",
    "PermissionManager",
    "DatasetPermission",
    "PermissionLevel",
    "CommentManager",
    "Comment",
    "ReviewManager",
    "ReviewStatus",
    "get_user_manager",
    "get_permission_manager",
    "get_comment_manager",
    "get_review_manager"
]

