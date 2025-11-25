"""
Comment System

Manages comments on dataset examples for team discussion
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Comment:
    """Comment model"""
    comment_id: str
    dataset_id: int
    example_id: int
    user_id: str
    content: str
    created_at: str
    updated_at: Optional[str]
    parent_comment_id: Optional[str]  # For threaded comments
    resolved: bool


class CommentManager:
    """Manager for comments"""

    def __init__(self, comments_file: str = "./data/collaboration/comments.json"):
        """
        Initialize comment manager

        Args:
            comments_file: Path to comments storage file
        """
        self.comments_file = Path(comments_file)
        self.comments_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_comments()

    def _load_comments(self):
        """Load comments from file"""
        if self.comments_file.exists():
            with open(self.comments_file, 'r') as f:
                data = json.load(f)
                self.comments = {
                    cid: Comment(**comment_data)
                    for cid, comment_data in data.items()
                }
        else:
            self.comments = {}
            self._save_comments()

    def _save_comments(self):
        """Save comments to file"""
        data = {
            cid: asdict(comment)
            for cid, comment in self.comments.items()
        }

        with open(self.comments_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_comment(
        self,
        dataset_id: int,
        example_id: int,
        user_id: str,
        content: str,
        parent_comment_id: Optional[str] = None
    ) -> Comment:
        """
        Create a new comment

        Args:
            dataset_id: Dataset ID
            example_id: Example ID
            user_id: User ID
            content: Comment content
            parent_comment_id: Parent comment ID for replies

        Returns:
            Created Comment object
        """
        comment_id = f"comment_{len(self.comments) + 1}"
        now = datetime.now().isoformat()

        comment = Comment(
            comment_id=comment_id,
            dataset_id=dataset_id,
            example_id=example_id,
            user_id=user_id,
            content=content,
            created_at=now,
            updated_at=None,
            parent_comment_id=parent_comment_id,
            resolved=False
        )

        self.comments[comment_id] = comment
        self._save_comments()

        return comment

    def get_comment(self, comment_id: str) -> Optional[Comment]:
        """
        Get comment by ID

        Args:
            comment_id: Comment ID

        Returns:
            Comment or None if not found
        """
        return self.comments.get(comment_id)

    def list_comments(
        self,
        dataset_id: Optional[int] = None,
        example_id: Optional[int] = None,
        user_id: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> List[Comment]:
        """
        List comments with optional filters

        Args:
            dataset_id: Filter by dataset ID
            example_id: Filter by example ID
            user_id: Filter by user ID
            resolved: Filter by resolved status

        Returns:
            List of Comment objects
        """
        comments = list(self.comments.values())

        if dataset_id is not None:
            comments = [c for c in comments if c.dataset_id == dataset_id]

        if example_id is not None:
            comments = [c for c in comments if c.example_id == example_id]

        if user_id is not None:
            comments = [c for c in comments if c.user_id == user_id]

        if resolved is not None:
            comments = [c for c in comments if c.resolved == resolved]

        # Sort by creation date (newest first)
        comments.sort(key=lambda c: c.created_at, reverse=True)

        return comments

    def update_comment(
        self,
        comment_id: str,
        content: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> bool:
        """
        Update a comment

        Args:
            comment_id: Comment ID
            content: New content
            resolved: New resolved status

        Returns:
            True if updated successfully
        """
        comment = self.comments.get(comment_id)

        if not comment:
            return False

        if content is not None:
            comment.content = content
            comment.updated_at = datetime.now().isoformat()

        if resolved is not None:
            comment.resolved = resolved

        self._save_comments()
        return True

    def delete_comment(self, comment_id: str) -> bool:
        """
        Delete a comment

        Args:
            comment_id: Comment ID

        Returns:
            True if deleted successfully
        """
        if comment_id in self.comments:
            # Also delete replies
            self._delete_replies(comment_id)

            del self.comments[comment_id]
            self._save_comments()
            return True

        return False

    def _delete_replies(self, parent_comment_id: str):
        """Delete all replies to a comment"""
        replies = [
            c.comment_id for c in self.comments.values()
            if c.parent_comment_id == parent_comment_id
        ]

        for reply_id in replies:
            if reply_id in self.comments:
                # Recursively delete nested replies
                self._delete_replies(reply_id)
                del self.comments[reply_id]

    def get_thread(self, comment_id: str) -> List[Comment]:
        """
        Get comment thread (comment + all replies)

        Args:
            comment_id: Root comment ID

        Returns:
            List of Comment objects in thread
        """
        thread = []

        # Get root comment
        root = self.comments.get(comment_id)
        if not root:
            return thread

        thread.append(root)

        # Get all replies recursively
        self._collect_replies(comment_id, thread)

        return thread

    def _collect_replies(self, parent_id: str, thread: List[Comment]):
        """Recursively collect replies to a comment"""
        replies = [
            c for c in self.comments.values()
            if c.parent_comment_id == parent_id
        ]

        for reply in sorted(replies, key=lambda c: c.created_at):
            thread.append(reply)
            self._collect_replies(reply.comment_id, thread)

    def get_example_threads(
        self,
        dataset_id: int,
        example_id: int
    ) -> List[List[Comment]]:
        """
        Get all comment threads for an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID

        Returns:
            List of comment threads
        """
        # Get root comments (no parent)
        root_comments = [
            c for c in self.comments.values()
            if (c.dataset_id == dataset_id and
                c.example_id == example_id and
                c.parent_comment_id is None)
        ]

        threads = []
        for root in sorted(root_comments, key=lambda c: c.created_at, reverse=True):
            thread = self.get_thread(root.comment_id)
            threads.append(thread)

        return threads

    def resolve_thread(self, comment_id: str) -> bool:
        """
        Resolve a comment thread (marks root comment as resolved)

        Args:
            comment_id: Root comment ID

        Returns:
            True if resolved successfully
        """
        return self.update_comment(comment_id, resolved=True)

    def get_statistics(
        self,
        dataset_id: Optional[int] = None
    ) -> Dict:
        """
        Get comment statistics

        Args:
            dataset_id: Optional dataset ID filter

        Returns:
            Statistics dictionary
        """
        comments = list(self.comments.values())

        if dataset_id is not None:
            comments = [c for c in comments if c.dataset_id == dataset_id]

        total = len(comments)
        resolved = sum(1 for c in comments if c.resolved)
        unresolved = total - resolved

        # Count by user
        user_counts = {}
        for comment in comments:
            user_counts[comment.user_id] = user_counts.get(comment.user_id, 0) + 1

        # Count root comments vs replies
        root_count = sum(1 for c in comments if c.parent_comment_id is None)
        reply_count = total - root_count

        return {
            "total_comments": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "root_comments": root_count,
            "replies": reply_count,
            "top_commenters": dict(sorted(
                user_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }


# Global comment manager instance
_comment_manager_instance: Optional[CommentManager] = None


def get_comment_manager() -> CommentManager:
    """Get the global comment manager instance"""
    global _comment_manager_instance

    if _comment_manager_instance is None:
        _comment_manager_instance = CommentManager()

    return _comment_manager_instance
