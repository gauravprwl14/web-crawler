"""
Pydantic schemas for API request and response validation.

This module defines all the data models used for API endpoints including
request validation, response formatting, and data serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, validator


class ContentType(str, Enum):
    """
    Supported content types for crawling.

    @description Enumerates all supported content types that the crawler can process
    """

    BLOG = "blog"
    PROMPT = "prompt"
    ARTICLE = "article"
    PRODUCT = "product"


class JobStatus(str, Enum):
    """
    Job execution status values.

    @description Defines the possible states of a crawling job
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobSchedule(str, Enum):
    """
    Job scheduling options.

    @description Defines when a job should be executed
    """

    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"


class SingleCrawlRequest(BaseModel):
    """
    Request model for single URL crawling.

    @description Defines the structure for single URL crawl requests with
    all necessary parameters for content extraction and processing
    """

    url: HttpUrl = Field(..., description="URL to crawl")
    content_type: ContentType = Field(..., description="Type of content to extract")
    selectors: Optional[Dict[str, str]] = Field(
        None,
        description="CSS selectors for content extraction",
        example={"title": "h1", "content": ".post-content", "author": ".author"},
    )
    crawler_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom crawler configuration parameters",
        example={"timeout": 30, "user_agent": "CustomBot/1.0"},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata to store with the content"
    )
    validate_content: bool = Field(
        True,
        description="Whether to validate extracted content against required fields",
    )
    store_raw_data: bool = Field(
        True, description="Whether to store raw HTML data for debugging"
    )
    store_in_db: bool = Field(
        True, description="Whether to store extracted content in database"
    )

    @validator("url")
    def validate_url(cls, v):
        """
        Validate URL format and protocol.

        @description Ensures URL is properly formatted and uses allowed protocols
        @param v: URL value to validate
        @returns: Validated URL
        @throws ValueError: If URL is invalid or uses unsupported protocol
        """
        url_str = str(v)
        if not url_str.startswith(("http://", "https://")):
            raise ValueError("URL must use http or https protocol")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/blog/my-post",
                "content_type": "blog",
                "selectors": {
                    "title": "h1, .post-title",
                    "content": ".post-content, article",
                    "author": ".author, .byline",
                    "published_date": ".date, .published",
                },
                "validate_content": True,
                "store_in_db": True,
            }
        }


class BatchCrawlRequest(BaseModel):
    """
    Request model for batch URL crawling.

    @description Defines the structure for batch crawl requests with
    multiple URLs and concurrency control parameters
    """

    urls: List[HttpUrl] = Field(
        ..., min_items=1, max_items=100, description="List of URLs to crawl (max 100)"
    )
    content_type: ContentType = Field(..., description="Type of content to extract")
    selectors: Optional[Dict[str, str]] = Field(
        None, description="CSS selectors for content extraction"
    )
    max_concurrent: Optional[int] = Field(
        None, ge=1, le=20, description="Maximum concurrent requests (1-20)"
    )
    crawler_config: Optional[Dict[str, Any]] = Field(
        None, description="Custom crawler configuration parameters"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata to store with content"
    )
    validate_content: bool = Field(
        True, description="Whether to validate extracted content"
    )
    store_raw_data: bool = Field(True, description="Whether to store raw HTML data")

    @validator("urls")
    def validate_urls(cls, v):
        """
        Validate URL list for batch processing.

        @description Ensures all URLs are valid and list size is appropriate
        @param v: List of URLs to validate
        @returns: Validated URL list
        @throws ValueError: If any URL is invalid or list is too large
        """
        if len(v) > 100:
            raise ValueError("Maximum 100 URLs allowed per batch request")

        for url in v:
            url_str = str(url)
            if not url_str.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL protocol: {url_str}")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://blog.com/post-1",
                    "https://blog.com/post-2",
                    "https://blog.com/post-3",
                ],
                "content_type": "blog",
                "max_concurrent": 3,
                "selectors": {"title": "h1", "content": ".content"},
            }
        }


class JobCreateRequest(BaseModel):
    """
    Request model for creating background crawling jobs.

    @description Defines the structure for job creation requests with
    scheduling and processing configuration options
    """

    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    urls: List[HttpUrl] = Field(
        ..., min_items=1, max_items=1000, description="List of URLs to crawl in the job"
    )
    content_type: ContentType = Field(..., description="Type of content to extract")
    selectors: Optional[Dict[str, str]] = Field(
        None, description="CSS selectors for content extraction"
    )
    schedule: JobSchedule = Field(
        JobSchedule.IMMEDIATE, description="When to execute the job"
    )
    crawler_config: Optional[Dict[str, Any]] = Field(
        None, description="Custom crawler configuration"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional job metadata"
    )
    validate_content: bool = Field(
        True, description="Whether to validate extracted content"
    )
    store_raw_data: bool = Field(
        False, description="Whether to store raw HTML (not recommended for large jobs)"
    )

    @validator("name")
    def validate_name(cls, v):
        """
        Validate job name.

        @description Ensures job name is properly formatted
        @param v: Job name to validate
        @returns: Validated job name
        """
        if not v.strip():
            raise ValueError("Job name cannot be empty")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Daily Blog Crawl",
                "urls": ["https://techblog.com/latest", "https://newsblog.com/recent"],
                "content_type": "blog",
                "schedule": "immediate",
                "selectors": {"title": "h1", "content": ".article-body"},
            }
        }


class CrawlResponse(BaseModel):
    """
    Response model for crawl operations.

    @description Defines the structure for crawl operation responses including
    extracted content, processing metadata, and error information
    """

    success: bool = Field(..., description="Whether the crawl was successful")
    url: str = Field(..., description="URL that was crawled")
    content_type: str = Field(..., description="Type of content extracted")
    extracted_data: Optional[Dict[str, Any]] = Field(
        None, description="Extracted content data"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during crawling"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    content_id: Optional[str] = Field(
        None, description="Database ID of stored content (if stored)"
    )
    validation_errors: Optional[List[str]] = Field(
        None, description="Content validation errors"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional processing metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "url": "https://example.com/blog/post",
                "content_type": "blog",
                "extracted_data": {
                    "title": "Example Blog Post",
                    "content": "This is the blog content...",
                    "author": "John Doe",
                    "published_date": "2024-01-15T10:30:00Z",
                },
                "errors": [],
                "processing_time": 2.5,
                "content_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }


class BatchCrawlResponse(BaseModel):
    """
    Response model for batch crawl operations.

    @description Defines the structure for batch crawl responses with
    aggregate statistics and individual results
    """

    success: bool = Field(..., description="Whether the batch operation completed")
    total_urls: int = Field(..., description="Total number of URLs processed")
    successful_count: int = Field(..., description="Number of successful crawls")
    failed_count: int = Field(..., description="Number of failed crawls")
    results: List[CrawlResponse] = Field(..., description="Individual crawl results")
    processing_time: float = Field(..., description="Total processing time in seconds")

    @validator("results")
    def validate_results_count(cls, v, values):
        """
        Validate that results count matches total URLs.

        @description Ensures consistency between result count and URL count
        @param v: Results list to validate
        @param values: Other field values for cross-validation
        @returns: Validated results list
        """
        if "total_urls" in values and len(v) != values["total_urls"]:
            raise ValueError("Results count must match total URLs")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_urls": 3,
                "successful_count": 2,
                "failed_count": 1,
                "processing_time": 7.5,
                "results": [
                    {
                        "success": True,
                        "url": "https://blog.com/post1",
                        "content_type": "blog",
                        "processing_time": 2.5,
                    }
                ],
            }
        }


class JobResponse(BaseModel):
    """
    Response model for job operations.

    @description Defines the structure for job-related responses including
    status information and progress tracking
    """

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    total_urls: int = Field(..., description="Total number of URLs in the job")
    completed_urls: int = Field(..., description="Number of completed URLs")
    message: Optional[str] = Field(None, description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "total_urls": 100,
                "completed_urls": 45,
                "message": "Job is processing...",
            }
        }


class JobStatusResponse(BaseModel):
    """
    Detailed response model for job status queries.

    @description Provides comprehensive status information about a crawling job
    including progress, results, and error details
    """

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    total_urls: int = Field(..., description="Total number of URLs")
    completed_urls: int = Field(..., description="Number of completed URLs")
    successful_urls: int = Field(..., description="Number of successful extractions")
    failed_urls: int = Field(..., description="Number of failed extractions")
    progress_percentage: float = Field(..., description="Completion percentage")
    estimated_remaining_time: Optional[float] = Field(
        None, description="Estimated remaining time in seconds"
    )
    errors: List[str] = Field(default_factory=list, description="Job-level errors")
    config: Optional[Dict[str, Any]] = Field(None, description="Job configuration")

    @validator("progress_percentage")
    def validate_progress(cls, v):
        """
        Validate progress percentage.

        @description Ensures progress is within valid range
        @param v: Progress percentage to validate
        @returns: Validated progress percentage
        """
        return max(0.0, min(100.0, v))

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:30:05Z",
                "total_urls": 100,
                "completed_urls": 75,
                "successful_urls": 70,
                "failed_urls": 5,
                "progress_percentage": 75.0,
                "estimated_remaining_time": 30.0,
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    @description Defines the structure for error responses across all endpoints
    """

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Invalid URL format provided",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_123456",
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    @description Defines the structure for system health status responses
    """

    status: str = Field(..., description="Overall system status")
    database: str = Field(..., description="Database connectivity status")
    crawler: str = Field(..., description="Crawler service status")
    timestamp: datetime = Field(..., description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "database": "connected",
                "crawler": "available",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class StatsResponse(BaseModel):
    """
    Response model for system statistics.

    @description Defines the structure for comprehensive system statistics
    """

    database: Dict[str, Any] = Field(..., description="Database statistics")
    jobs: Dict[str, Any] = Field(..., description="Job processing statistics")
    system: Dict[str, Any] = Field(..., description="System configuration and status")

    class Config:
        json_schema_extra = {
            "example": {
                "database": {
                    "total_content_items": 1234,
                    "content_by_type": {"blog": 800, "article": 434},
                },
                "jobs": {"total_jobs": 56, "active_jobs": 3, "completed_jobs": 53},
                "system": {
                    "api_version": "1.0.0",
                    "supported_content_types": ["blog", "article", "prompt", "product"],
                },
            }
        }


class ContentFilter(BaseModel):
    """
    Model for content filtering parameters.

    @description Defines structure for filtering content queries
    """

    field: str = Field(..., description="Field name to filter by")
    operator: str = Field(..., description="Filter operator (eq, ne, gt, lt, in)")
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(
        ..., description="Filter value"
    )

    @validator("operator")
    def validate_operator(cls, v):
        """
        Validate filter operator.

        @description Ensures operator is supported
        @param v: Operator to validate
        @returns: Validated operator
        """
        allowed_operators = [
            "eq",
            "ne",
            "gt",
            "lt",
            "gte",
            "lte",
            "in",
            "not_in",
            "contains",
        ]
        if v not in allowed_operators:
            raise ValueError(f"Operator must be one of: {allowed_operators}")
        return v


class ContentQuery(BaseModel):
    """
    Model for complex content queries.

    @description Defines structure for advanced content search queries
    """

    content_type: ContentType = Field(..., description="Type of content to query")
    filters: Optional[List[ContentFilter]] = Field(
        None, description="List of filters to apply"
    )
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("desc", description="Sort order (asc, desc)")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Number of results to skip")

    @validator("sort_order")
    def validate_sort_order(cls, v):
        """
        Validate sort order.

        @description Ensures sort order is valid
        @param v: Sort order to validate
        @returns: Validated sort order
        """
        if v not in ["asc", "desc"]:
            raise ValueError("Sort order must be asc or desc")
        return v
