"""
API Extensions for Dataset Creator
New endpoints for data collection, deduplication, versioning, and config management
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Body, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio

import database as db
from llm_providers import create_provider
import data_collectors
import deduplication
import versioning
import config_loader


# Create API router
router = APIRouter(prefix="/api/v2", tags=["v2"])


# ============================================================================
# MODELS
# ============================================================================

class WebScrapingRequest(BaseModel):
    urls: List[str] = Field(..., description="URLs to scrape")
    selector: Optional[str] = Field(None, description="CSS selector for content")
    max_concurrent: int = Field(5, description="Max concurrent requests")
    convert_to_examples: bool = Field(True, description="Convert to training examples")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    provider: Optional[str] = Field(None, description="LLM provider for conversion")
    model: Optional[str] = Field(None, description="LLM model for conversion")


class WebCrawlingRequest(BaseModel):
    start_url: str = Field(..., description="Starting URL")
    max_pages: int = Field(100, description="Maximum pages to crawl")
    link_pattern: Optional[str] = Field(None, description="Regex pattern for links")
    content_selector: Optional[str] = Field(None, description="CSS selector for content")
    convert_to_examples: bool = Field(True, description="Convert to training examples")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    provider: Optional[str] = Field(None, description="LLM provider for conversion")
    model: Optional[str] = Field(None, description="LLM model for conversion")


class GitHubIssuesRequest(BaseModel):
    repo: str = Field(..., description="Repository (owner/name)")
    state: str = Field("all", description="Issue state (open/closed/all)")
    max_issues: int = Field(100, description="Maximum issues to fetch")
    api_key: Optional[str] = Field(None, description="GitHub API token")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    format_type: str = Field("chat", description="Output format")


class StackOverflowRequest(BaseModel):
    tag: str = Field(..., description="Tag to search")
    max_questions: int = Field(100, description="Maximum questions to fetch")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    format_type: str = Field("chat", description="Output format")


class DeduplicationRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID to deduplicate")
    method: str = Field("hash", description="Method: exact/hash/fuzzy")
    similarity_threshold: float = Field(0.9, description="Similarity threshold for fuzzy")
    keep_first: bool = Field(True, description="Keep first occurrence")
    create_new_dataset: bool = Field(True, description="Create new dataset or update existing")


class CleaningRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID to clean")
    min_length: int = Field(10, description="Minimum content length")
    max_length: int = Field(10000, description="Maximum content length")
    remove_toxic: bool = Field(True, description="Remove toxic content")
    toxic_patterns: List[str] = Field(default_factory=list, description="Toxic patterns")
    create_new_dataset: bool = Field(True, description="Create new dataset or update existing")


class VersionRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID")
    description: str = Field("", description="Version description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Version metadata")


class RollbackRequest(BaseModel):
    dataset_id: int = Field(..., description="Dataset ID")
    version_id: str = Field(..., description="Version ID to rollback to")


class ConfigPipelineRequest(BaseModel):
    config_path: str = Field(..., description="Path to configuration file")
    execute: bool = Field(False, description="Execute pipeline immediately")


# ============================================================================
# WEB SCRAPING ENDPOINTS
# ============================================================================

@router.post("/collect/web-scrape")
async def scrape_websites(
    request: WebScrapingRequest,
    background_tasks: BackgroundTasks
):
    """Scrape websites and optionally convert to training examples"""
    try:
        # Create LLM provider if conversion is requested
        llm_provider = None
        if request.convert_to_examples:
            llm_provider = create_provider(
                request.provider or "ollama",
                model=request.model or "gemma3:27b"
            )

        # Create web scraper
        scraper = data_collectors.WebScraperCollector(llm_provider)

        # Scrape URLs asynchronously
        scraped_data = await scraper.scrape_urls(
            urls=request.urls,
            selector=request.selector,
            max_concurrent=request.max_concurrent
        )

        if not scraped_data:
            raise HTTPException(status_code=400, detail="No data scraped")

        # Convert to training examples if requested
        if request.convert_to_examples and llm_provider:
            examples = scraper.convert_to_training_examples(scraped_data, format_type='chat')
        else:
            # Use raw scraped data
            examples = scraped_data

        # Create dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.dataset_name or f"web_scraped_{timestamp}"
        file_path = f"data/datasets/{dataset_name}.jsonl"

        # Save examples
        import utils
        utils.save_jsonl(examples, file_path)

        # Create dataset record
        dataset_id = db.create_dataset(
            name=dataset_name,
            domain="web_scraping",
            format="chat",
            file_path=file_path,
            example_count=len(examples),
            metadata={
                "source": "web_scraping",
                "urls": request.urls,
                "scraped_count": len(scraped_data)
            }
        )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "scraped_urls": len(scraped_data),
            "examples_created": len(examples)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/web-crawl")
async def crawl_website(
    request: WebCrawlingRequest,
    background_tasks: BackgroundTasks
):
    """Crawl a website and optionally convert to training examples"""
    try:
        # Create LLM provider if conversion is requested
        llm_provider = None
        if request.convert_to_examples:
            llm_provider = create_provider(
                request.provider or "ollama",
                model=request.model or "gemma3:27b"
            )

        # Create web scraper
        scraper = data_collectors.WebScraperCollector(llm_provider)

        # Crawl website asynchronously
        scraped_data = await scraper.crawl_website(
            start_url=request.start_url,
            max_pages=request.max_pages,
            link_pattern=request.link_pattern,
            content_selector=request.content_selector
        )

        if not scraped_data:
            raise HTTPException(status_code=400, detail="No data crawled")

        # Convert to training examples if requested
        if request.convert_to_examples and llm_provider:
            examples = scraper.convert_to_training_examples(scraped_data, format_type='chat')
        else:
            examples = scraped_data

        # Create dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.dataset_name or f"web_crawled_{timestamp}"
        file_path = f"data/datasets/{dataset_name}.jsonl"

        # Save examples
        import utils
        utils.save_jsonl(examples, file_path)

        # Create dataset record
        dataset_id = db.create_dataset(
            name=dataset_name,
            domain="web_crawling",
            format="chat",
            file_path=file_path,
            example_count=len(examples),
            metadata={
                "source": "web_crawling",
                "start_url": request.start_url,
                "pages_crawled": len(scraped_data)
            }
        )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "pages_crawled": len(scraped_data),
            "examples_created": len(examples)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API COLLECTION ENDPOINTS
# ============================================================================

@router.post("/collect/github-issues")
async def collect_github_issues(request: GitHubIssuesRequest):
    """Collect GitHub issues and convert to training examples"""
    try:
        # Create API collector
        collector = data_collectors.APICollector(api_key=request.api_key)

        # Fetch GitHub issues
        issues = await collector.fetch_github_issues(
            repo=request.repo,
            state=request.state,
            max_issues=request.max_issues
        )

        if not issues:
            raise HTTPException(status_code=400, detail="No issues found")

        # Convert to training examples
        examples = collector.convert_to_training_examples(
            issues,
            data_type='github_issues',
            format_type=request.format_type
        )

        # Create dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.dataset_name or f"github_issues_{request.repo.replace('/', '_')}_{timestamp}"
        file_path = f"data/datasets/{dataset_name}.jsonl"

        # Save examples
        import utils
        utils.save_jsonl(examples, file_path)

        # Create dataset record
        dataset_id = db.create_dataset(
            name=dataset_name,
            domain="github_issues",
            format=request.format_type,
            file_path=file_path,
            example_count=len(examples),
            metadata={
                "source": "github_issues",
                "repo": request.repo,
                "issues_collected": len(issues)
            }
        )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "issues_collected": len(issues),
            "examples_created": len(examples)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/stackoverflow")
async def collect_stackoverflow(request: StackOverflowRequest):
    """Collect StackOverflow questions and convert to training examples"""
    try:
        # Create API collector
        collector = data_collectors.APICollector()

        # Fetch StackOverflow questions
        questions = await collector.fetch_stackoverflow_questions(
            tag=request.tag,
            max_questions=request.max_questions
        )

        if not questions:
            raise HTTPException(status_code=400, detail="No questions found")

        # Convert to training examples
        examples = collector.convert_to_training_examples(
            questions,
            data_type='stackoverflow',
            format_type=request.format_type
        )

        # Create dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = request.dataset_name or f"stackoverflow_{request.tag}_{timestamp}"
        file_path = f"data/datasets/{dataset_name}.jsonl"

        # Save examples
        import utils
        utils.save_jsonl(examples, file_path)

        # Create dataset record
        dataset_id = db.create_dataset(
            name=dataset_name,
            domain="stackoverflow",
            format=request.format_type,
            file_path=file_path,
            example_count=len(examples),
            metadata={
                "source": "stackoverflow",
                "tag": request.tag,
                "questions_collected": len(questions)
            }
        )

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "questions_collected": len(questions),
            "examples_created": len(examples)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DEDUPLICATION & CLEANING ENDPOINTS
# ============================================================================

@router.post("/process/deduplicate")
async def deduplicate_dataset(request: DeduplicationRequest):
    """Remove duplicates from a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load examples
        import utils
        examples = utils.load_jsonl(dataset['file_path'])

        # Deduplicate
        deduplicator = deduplication.DataDeduplicator(
            similarity_threshold=request.similarity_threshold
        )

        unique_examples, stats = deduplicator.deduplicate_dataset(
            examples=examples,
            format_type=dataset['format'],
            method=request.method,
            keep_first=request.keep_first
        )

        if request.create_new_dataset:
            # Create new dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{dataset['name']}_deduplicated_{timestamp}"
            file_path = f"data/datasets/{dataset_name}.jsonl"

            # Save deduplicated examples
            utils.save_jsonl(unique_examples, file_path)

            # Create dataset record
            new_dataset_id = db.create_dataset(
                name=dataset_name,
                domain=dataset['domain'],
                format=dataset['format'],
                file_path=file_path,
                example_count=len(unique_examples),
                metadata={
                    "source": "deduplication",
                    "parent_dataset_id": request.dataset_id,
                    "deduplication_stats": stats
                }
            )

            return {
                "dataset_id": new_dataset_id,
                "dataset_name": dataset_name,
                **stats
            }
        else:
            # Update existing dataset
            utils.save_jsonl(unique_examples, dataset['file_path'])
            db.update_dataset(request.dataset_id, example_count=len(unique_examples))

            return {
                "dataset_id": request.dataset_id,
                **stats
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/clean")
async def clean_dataset(request: CleaningRequest):
    """Clean and filter a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load examples
        import utils
        examples = utils.load_jsonl(dataset['file_path'])

        # Clean
        cleaner = deduplication.DataCleaner()

        filtered_examples, stats = cleaner.filter_dataset(
            examples=examples,
            format_type=dataset['format'],
            min_length=request.min_length,
            max_length=request.max_length,
            remove_toxic=request.remove_toxic,
            toxic_patterns=request.toxic_patterns if request.toxic_patterns else None
        )

        if request.create_new_dataset:
            # Create new dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{dataset['name']}_cleaned_{timestamp}"
            file_path = f"data/datasets/{dataset_name}.jsonl"

            # Save cleaned examples
            utils.save_jsonl(filtered_examples, file_path)

            # Create dataset record
            new_dataset_id = db.create_dataset(
                name=dataset_name,
                domain=dataset['domain'],
                format=dataset['format'],
                file_path=file_path,
                example_count=len(filtered_examples),
                metadata={
                    "source": "cleaning",
                    "parent_dataset_id": request.dataset_id,
                    "cleaning_stats": stats
                }
            )

            return {
                "dataset_id": new_dataset_id,
                "dataset_name": dataset_name,
                **stats
            }
        else:
            # Update existing dataset
            utils.save_jsonl(filtered_examples, dataset['file_path'])
            db.update_dataset(request.dataset_id, example_count=len(filtered_examples))

            return {
                "dataset_id": request.dataset_id,
                **stats
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# VERSIONING ENDPOINTS
# ============================================================================

@router.post("/datasets/{dataset_id}/versions")
async def create_version(dataset_id: int, request: VersionRequest):
    """Create a new version of a dataset"""
    try:
        # Get dataset
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Create version
        version = versioning.create_dataset_version(
            dataset_id=dataset_id,
            file_path=dataset['file_path'],
            description=request.description,
            metadata=request.metadata
        )

        return version.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/versions")
async def list_versions(dataset_id: int):
    """List all versions of a dataset"""
    try:
        history = versioning.get_dataset_version_history(dataset_id)
        return {"versions": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/{dataset_id}/rollback")
async def rollback_version(dataset_id: int, request: RollbackRequest):
    """Rollback to a specific version"""
    try:
        success = versioning.rollback_to_version(dataset_id, request.version_id)

        if not success:
            raise HTTPException(status_code=400, detail="Rollback failed")

        return {"status": "success", "version_id": request.version_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@router.get("/configs")
async def list_configs():
    """List available pipeline configurations"""
    try:
        configs = config_loader.config_loader.list_configs()
        return {"configs": configs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/execute")
async def execute_config_pipeline(
    request: ConfigPipelineRequest,
    background_tasks: BackgroundTasks
):
    """Load and optionally execute a configuration pipeline"""
    try:
        # Load configuration
        config = config_loader.load_pipeline_config(request.config_path)

        if not request.execute:
            # Just validate and return
            return {
                "status": "validated",
                "config": config.dict()
            }

        # Execute pipeline in background
        # TODO: Implement full pipeline execution
        # For now, just return the config

        return {
            "status": "executing",
            "pipeline_name": config.name,
            "data_sources": len(config.data_sources)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/examples")
async def create_example_configs():
    """Create example configuration files"""
    try:
        config_loader.create_example_configs()
        return {"status": "success", "message": "Example configs created in configs/examples/"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
