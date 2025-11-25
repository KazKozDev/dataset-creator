"""
Analytics Metrics Tracker

Tracks generation costs, API usage, and performance metrics
"""

import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


class AnalyticsTracker:
    """Tracker for analytics metrics"""

    def __init__(self, metrics_file: str = "./data/analytics/metrics.jsonl"):
        """
        Initialize the analytics tracker

        Args:
            metrics_file: Path to metrics storage file
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory metrics for current session
        self.session_metrics = {
            "generation_jobs": [],
            "api_calls": [],
            "exports": [],
            "quality_checks": []
        }

    def track_generation(
        self,
        job_id: int,
        provider: str,
        model: str,
        examples_requested: int,
        examples_generated: int,
        tokens_used: int,
        duration_seconds: float,
        cost_estimate: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a generation job

        Args:
            job_id: Generation job ID
            provider: LLM provider used
            model: Model name
            examples_requested: Number of examples requested
            examples_generated: Number of examples successfully generated
            tokens_used: Estimated tokens used
            duration_seconds: Job duration in seconds
            cost_estimate: Estimated cost in USD
            metadata: Additional metadata
        """
        metric = {
            "type": "generation",
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id,
            "provider": provider,
            "model": model,
            "examples_requested": examples_requested,
            "examples_generated": examples_generated,
            "tokens_used": tokens_used,
            "duration_seconds": duration_seconds,
            "cost_estimate": cost_estimate,
            "metadata": metadata or {}
        }

        self.session_metrics["generation_jobs"].append(metric)
        self._append_to_file(metric)

    def track_api_call(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        user_id: Optional[str] = None
    ) -> None:
        """
        Track an API call

        Args:
            endpoint: API endpoint called
            method: HTTP method
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            user_id: Optional user identifier
        """
        metric = {
            "type": "api_call",
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "response_time_ms": response_time_ms,
            "status_code": status_code,
            "user_id": user_id
        }

        self.session_metrics["api_calls"].append(metric)
        self._append_to_file(metric)

    def track_export(
        self,
        dataset_id: int,
        format: str,
        examples_count: int,
        duration_seconds: float,
        success: bool = True
    ) -> None:
        """
        Track a dataset export

        Args:
            dataset_id: Dataset ID
            format: Export format
            examples_count: Number of examples exported
            duration_seconds: Export duration
            success: Whether export succeeded
        """
        metric = {
            "type": "export",
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "format": format,
            "examples_count": examples_count,
            "duration_seconds": duration_seconds,
            "success": success
        }

        self.session_metrics["exports"].append(metric)
        self._append_to_file(metric)

    def track_quality_check(
        self,
        dataset_id: int,
        examples_checked: int,
        average_score: float,
        issues_found: int,
        duration_seconds: float
    ) -> None:
        """
        Track a quality check job

        Args:
            dataset_id: Dataset ID
            examples_checked: Number of examples checked
            average_score: Average quality score
            issues_found: Number of issues found
            duration_seconds: Check duration
        """
        metric = {
            "type": "quality_check",
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "examples_checked": examples_checked,
            "average_score": average_score,
            "issues_found": issues_found,
            "duration_seconds": duration_seconds
        }

        self.session_metrics["quality_checks"].append(metric)
        self._append_to_file(metric)

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get analytics summary for a date range

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: now)

        Returns:
            Summary dictionary with aggregated metrics
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)

        if end_date is None:
            end_date = datetime.now()

        # Load metrics from file
        metrics = self._load_metrics(start_date, end_date)

        # Aggregate metrics
        summary = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generation": self._aggregate_generation_metrics(metrics),
            "api_usage": self._aggregate_api_metrics(metrics),
            "exports": self._aggregate_export_metrics(metrics),
            "quality": self._aggregate_quality_metrics(metrics)
        }

        return summary

    def _aggregate_generation_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate generation metrics"""
        generation_metrics = [m for m in metrics if m.get("type") == "generation"]

        if not generation_metrics:
            return {"total_jobs": 0}

        total_examples = sum(m.get("examples_generated", 0) for m in generation_metrics)
        total_tokens = sum(m.get("tokens_used", 0) for m in generation_metrics)
        total_cost = sum(m.get("cost_estimate", 0) for m in generation_metrics)
        total_duration = sum(m.get("duration_seconds", 0) for m in generation_metrics)

        # Provider breakdown
        providers = defaultdict(int)
        for m in generation_metrics:
            providers[m.get("provider", "unknown")] += 1

        # Model breakdown
        models = defaultdict(int)
        for m in generation_metrics:
            models[m.get("model", "unknown")] += 1

        return {
            "total_jobs": len(generation_metrics),
            "total_examples": total_examples,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 2),
            "total_duration_hours": round(total_duration / 3600, 2),
            "average_examples_per_job": round(total_examples / len(generation_metrics), 1),
            "providers": dict(providers),
            "models": dict(models)
        }

    def _aggregate_api_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate API usage metrics"""
        api_metrics = [m for m in metrics if m.get("type") == "api_call"]

        if not api_metrics:
            return {"total_calls": 0}

        total_calls = len(api_metrics)
        avg_response_time = sum(m.get("response_time_ms", 0) for m in api_metrics) / total_calls

        # Status code breakdown
        status_codes = defaultdict(int)
        for m in api_metrics:
            status_codes[m.get("status_code", 0)] += 1

        # Endpoint breakdown
        endpoints = defaultdict(int)
        for m in api_metrics:
            endpoints[m.get("endpoint", "unknown")] += 1

        return {
            "total_calls": total_calls,
            "average_response_time_ms": round(avg_response_time, 2),
            "status_codes": dict(status_codes),
            "top_endpoints": dict(sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:10])
        }

    def _aggregate_export_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate export metrics"""
        export_metrics = [m for m in metrics if m.get("type") == "export"]

        if not export_metrics:
            return {"total_exports": 0}

        total_exports = len(export_metrics)
        successful_exports = sum(1 for m in export_metrics if m.get("success", False))

        # Format breakdown
        formats = defaultdict(int)
        for m in export_metrics:
            formats[m.get("format", "unknown")] += 1

        return {
            "total_exports": total_exports,
            "successful_exports": successful_exports,
            "success_rate": round(successful_exports / total_exports * 100, 1),
            "formats": dict(formats)
        }

    def _aggregate_quality_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quality check metrics"""
        quality_metrics = [m for m in metrics if m.get("type") == "quality_check"]

        if not quality_metrics:
            return {"total_checks": 0}

        total_examples = sum(m.get("examples_checked", 0) for m in quality_metrics)
        total_issues = sum(m.get("issues_found", 0) for m in quality_metrics)

        scores = [m.get("average_score", 0) for m in quality_metrics]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "total_checks": len(quality_metrics),
            "total_examples_checked": total_examples,
            "total_issues_found": total_issues,
            "average_quality_score": round(avg_score, 2)
        }

    def _append_to_file(self, metric: Dict[str, Any]) -> None:
        """Append metric to file"""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            print(f"Error writing metric to file: {e}")

    def _load_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Load metrics from file within date range"""
        metrics = []

        if not self.metrics_file.exists():
            return metrics

        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        metric = json.loads(line)
                        metric_date = datetime.fromisoformat(metric.get("timestamp", ""))

                        if start_date <= metric_date <= end_date:
                            metrics.append(metric)

                    except (json.JSONDecodeError, ValueError):
                        continue

        except Exception as e:
            print(f"Error loading metrics: {e}")

        return metrics

    def estimate_generation_cost(
        self,
        provider: str,
        model: str,
        tokens: int
    ) -> float:
        """
        Estimate generation cost based on provider and model

        Args:
            provider: LLM provider
            model: Model name
            tokens: Number of tokens

        Returns:
            Estimated cost in USD
        """
        # Cost per 1K tokens (approximate)
        cost_per_1k = {
            "openai": {
                "gpt-4": 0.03,
                "gpt-4-turbo-preview": 0.01,
                "gpt-3.5-turbo": 0.0015
            },
            "anthropic": {
                "claude-3-7-sonnet-20250219": 0.003,
                "claude-3-5-sonnet-20241022": 0.003,
                "claude-3-opus-20240229": 0.015,
                "claude-3-haiku-20240307": 0.00025
            },
            "ollama": {}  # Local models - no cost
        }

        provider_costs = cost_per_1k.get(provider.lower(), {})
        cost_rate = provider_costs.get(model, 0.0)

        return (tokens / 1000) * cost_rate


# Global tracker instance
_tracker_instance: Optional[AnalyticsTracker] = None


def get_tracker() -> AnalyticsTracker:
    """
    Get the global analytics tracker instance

    Returns:
        AnalyticsTracker instance
    """
    global _tracker_instance

    if _tracker_instance is None:
        _tracker_instance = AnalyticsTracker()

    return _tracker_instance
