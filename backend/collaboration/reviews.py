"""
Review Workflow System

Manages review status and workflow for dataset examples
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class ReviewStatus(str, Enum):
    """Review statuses"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


@dataclass
class Review:
    """Review model"""
    review_id: str
    dataset_id: int
    example_id: int
    reviewer_id: str
    status: ReviewStatus
    feedback: Optional[str]
    created_at: str
    updated_at: Optional[str]


class ReviewManager:
    """Manager for review workflow"""

    def __init__(self, reviews_file: str = "./data/collaboration/reviews.json"):
        """
        Initialize review manager

        Args:
            reviews_file: Path to reviews storage file
        """
        self.reviews_file = Path(reviews_file)
        self.reviews_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_reviews()

    def _load_reviews(self):
        """Load reviews from file"""
        if self.reviews_file.exists():
            with open(self.reviews_file, 'r') as f:
                data = json.load(f)
                self.reviews = {
                    rid: Review(**review_data)
                    for rid, review_data in data.items()
                }
        else:
            self.reviews = {}
            self._save_reviews()

    def _save_reviews(self):
        """Save reviews to file"""
        data = {
            rid: asdict(review)
            for rid, review in self.reviews.items()
        }

        with open(self.reviews_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_review(
        self,
        dataset_id: int,
        example_id: int,
        reviewer_id: str,
        status: ReviewStatus,
        feedback: Optional[str] = None
    ) -> Review:
        """
        Create a review for an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID
            reviewer_id: Reviewer user ID
            status: Review status
            feedback: Optional feedback text

        Returns:
            Created Review object
        """
        review_id = f"review_{len(self.reviews) + 1}"
        now = datetime.now().isoformat()

        review = Review(
            review_id=review_id,
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            status=status,
            feedback=feedback,
            created_at=now,
            updated_at=None
        )

        self.reviews[review_id] = review
        self._save_reviews()

        return review

    def get_review(self, review_id: str) -> Optional[Review]:
        """
        Get review by ID

        Args:
            review_id: Review ID

        Returns:
            Review or None if not found
        """
        return self.reviews.get(review_id)

    def get_example_review(
        self,
        dataset_id: int,
        example_id: int
    ) -> Optional[Review]:
        """
        Get the latest review for an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID

        Returns:
            Latest Review or None if not found
        """
        example_reviews = [
            r for r in self.reviews.values()
            if r.dataset_id == dataset_id and r.example_id == example_id
        ]

        if not example_reviews:
            return None

        # Return most recent review
        return max(example_reviews, key=lambda r: r.created_at)

    def list_reviews(
        self,
        dataset_id: Optional[int] = None,
        reviewer_id: Optional[str] = None,
        status: Optional[ReviewStatus] = None
    ) -> List[Review]:
        """
        List reviews with optional filters

        Args:
            dataset_id: Filter by dataset ID
            reviewer_id: Filter by reviewer ID
            status: Filter by review status

        Returns:
            List of Review objects
        """
        reviews = list(self.reviews.values())

        if dataset_id is not None:
            reviews = [r for r in reviews if r.dataset_id == dataset_id]

        if reviewer_id is not None:
            reviews = [r for r in reviews if r.reviewer_id == reviewer_id]

        if status is not None:
            reviews = [r for r in reviews if r.status == status]

        # Sort by creation date (newest first)
        reviews.sort(key=lambda r: r.created_at, reverse=True)

        return reviews

    def update_review(
        self,
        review_id: str,
        status: Optional[ReviewStatus] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Update a review

        Args:
            review_id: Review ID
            status: New status
            feedback: New feedback

        Returns:
            True if updated successfully
        """
        review = self.reviews.get(review_id)

        if not review:
            return False

        if status is not None:
            review.status = status

        if feedback is not None:
            review.feedback = feedback

        review.updated_at = datetime.now().isoformat()

        self._save_reviews()
        return True

    def approve_example(
        self,
        dataset_id: int,
        example_id: int,
        reviewer_id: str,
        feedback: Optional[str] = None
    ) -> Review:
        """
        Approve an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID
            reviewer_id: Reviewer user ID
            feedback: Optional feedback

        Returns:
            Created Review object
        """
        return self.create_review(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            status=ReviewStatus.APPROVED,
            feedback=feedback
        )

    def reject_example(
        self,
        dataset_id: int,
        example_id: int,
        reviewer_id: str,
        feedback: str
    ) -> Review:
        """
        Reject an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID
            reviewer_id: Reviewer user ID
            feedback: Rejection feedback (required)

        Returns:
            Created Review object
        """
        return self.create_review(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            status=ReviewStatus.REJECTED,
            feedback=feedback
        )

    def request_changes(
        self,
        dataset_id: int,
        example_id: int,
        reviewer_id: str,
        feedback: str
    ) -> Review:
        """
        Request changes to an example

        Args:
            dataset_id: Dataset ID
            example_id: Example ID
            reviewer_id: Reviewer user ID
            feedback: Change requests (required)

        Returns:
            Created Review object
        """
        return self.create_review(
            dataset_id=dataset_id,
            example_id=example_id,
            reviewer_id=reviewer_id,
            status=ReviewStatus.NEEDS_CHANGES,
            feedback=feedback
        )

    def get_pending_reviews(
        self,
        dataset_id: int
    ) -> List[int]:
        """
        Get list of example IDs pending review in a dataset

        Args:
            dataset_id: Dataset ID

        Returns:
            List of example IDs pending review
        """
        pending_ids = set()

        for review in self.reviews.values():
            if (review.dataset_id == dataset_id and
                review.status == ReviewStatus.PENDING):
                pending_ids.add(review.example_id)

        return list(pending_ids)

    def get_approved_examples(
        self,
        dataset_id: int
    ) -> List[int]:
        """
        Get list of approved example IDs in a dataset

        Args:
            dataset_id: Dataset ID

        Returns:
            List of approved example IDs
        """
        # Get latest review for each example
        example_reviews = {}

        for review in self.reviews.values():
            if review.dataset_id == dataset_id:
                if (review.example_id not in example_reviews or
                    review.created_at > example_reviews[review.example_id].created_at):
                    example_reviews[review.example_id] = review

        # Filter approved
        approved = [
            ex_id for ex_id, review in example_reviews.items()
            if review.status == ReviewStatus.APPROVED
        ]

        return approved

    def get_review_summary(
        self,
        dataset_id: int
    ) -> Dict:
        """
        Get review summary for a dataset

        Args:
            dataset_id: Dataset ID

        Returns:
            Summary dictionary
        """
        reviews = self.list_reviews(dataset_id=dataset_id)

        # Get latest review per example
        example_reviews = {}
        for review in reviews:
            if (review.example_id not in example_reviews or
                review.created_at > example_reviews[review.example_id].created_at):
                example_reviews[review.example_id] = review

        # Count by status
        status_counts = {}
        for status in ReviewStatus:
            status_counts[status.value] = sum(
                1 for r in example_reviews.values()
                if r.status == status
            )

        # Get reviewer activity
        reviewer_counts = {}
        for review in reviews:
            reviewer_counts[review.reviewer_id] = reviewer_counts.get(review.reviewer_id, 0) + 1

        return {
            "total_reviews": len(reviews),
            "unique_examples_reviewed": len(example_reviews),
            "status_breakdown": status_counts,
            "top_reviewers": dict(sorted(
                reviewer_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }

    def get_statistics(self) -> Dict:
        """
        Get overall review statistics

        Returns:
            Statistics dictionary
        """
        # Count by status
        status_counts = {}
        for status in ReviewStatus:
            status_counts[status.value] = sum(
                1 for r in self.reviews.values()
                if r.status == status
            )

        # Count unique datasets
        unique_datasets = len(set(r.dataset_id for r in self.reviews.values()))

        # Count unique examples
        unique_examples = len(set(
            (r.dataset_id, r.example_id)
            for r in self.reviews.values()
        ))

        # Count unique reviewers
        unique_reviewers = len(set(r.reviewer_id for r in self.reviews.values()))

        return {
            "total_reviews": len(self.reviews),
            "status_breakdown": status_counts,
            "unique_datasets": unique_datasets,
            "unique_examples_reviewed": unique_examples,
            "unique_reviewers": unique_reviewers
        }


# Global review manager instance
_review_manager_instance: Optional[ReviewManager] = None


def get_review_manager() -> ReviewManager:
    """Get the global review manager instance"""
    global _review_manager_instance

    if _review_manager_instance is None:
        _review_manager_instance = ReviewManager()

    return _review_manager_instance
