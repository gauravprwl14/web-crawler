"""
Utility classes and functions for API operations.

This module provides utility classes for job management, validation functions,
and other common API operations with proper error handling and logging.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from api.schemas import JobCreateRequest, JobResponse, JobStatusResponse
from config.settings import get_settings
from crawler.core import CrawlRequest, FlexibleWebCrawler
from database.models import CONTENT_TYPE_MODELS, db_manager

settings = get_settings()


class JobManager:
    """
    Manager for handling background crawling jobs.

    @description Manages the lifecycle of crawling jobs including creation,
    execution, monitoring, and cleanup with proper state management
    """

    def __init__(self):
        """Initialize job manager with tracking structures."""
        self._active_jobs: Dict[str, Dict] = {}
        self._job_history: Dict[str, Dict] = {}
        self._max_concurrent_jobs = 5
        self._cleanup_interval = 3600  # 1 hour
        self._last_cleanup = time.time()

    async def execute_job(self, job_id: str, request: JobCreateRequest) -> None:
        """
        Execute a crawling job in the background.

        @description Processes a crawling job by iterating through all URLs
        and storing results with proper progress tracking and error handling
        @param job_id: Unique identifier for the job
        @param request: Job creation request with URLs and configuration
        @throws Exception: If job execution fails catastrophically

        @example
        # Execute job in background
        await job_manager.execute_job("job-123", job_request)
        """
        start_time = time.time()
        job_info = {
            "job_id": job_id,
            "status": "running",
            "started_at": datetime.utcnow(),
            "total_urls": len(request.urls),
            "completed_urls": 0,
            "successful_urls": 0,
            "failed_urls": 0,
            "errors": [],
            "config": request.dict(),
        }

        self._active_jobs[job_id] = job_info

        try:
            logger.info(f"Starting job execution: {job_id}")

            # Update job status in database
            await self._update_job_status(
                job_id, "running", {"started_at": datetime.utcnow(), "progress": 0.0}
            )

            # Create crawler instance
            async with FlexibleWebCrawler(request.crawler_config) as crawler:
                # Process URLs in batches for better performance
                batch_size = min(10, settings.crawler.concurrent_requests)
                url_batches = [
                    request.urls[i : i + batch_size]
                    for i in range(0, len(request.urls), batch_size)
                ]

                for batch in url_batches:
                    # Create crawl requests for batch
                    crawl_requests = []
                    for url in batch:
                        crawl_request = CrawlRequest(
                            url=str(url),
                            content_type=request.content_type,
                            selectors=request.selectors,
                            custom_config=request.crawler_config,
                            metadata=request.metadata,
                            job_id=job_id,
                            validate_content=request.validate_content,
                            store_raw_data=request.store_raw_data,
                        )
                        crawl_requests.append(crawl_request)

                    # Process batch
                    batch_results = await crawler.crawl_multiple(
                        crawl_requests, max_concurrent=batch_size
                    )

                    # Update progress
                    for result in batch_results:
                        job_info["completed_urls"] += 1

                        if result.success:
                            job_info["successful_urls"] += 1
                        else:
                            job_info["failed_urls"] += 1
                            if result.errors:
                                job_info["errors"].extend(result.errors)

                        # Store individual result
                        await self._store_job_result(job_id, result)

                    # Update progress in database
                    progress = (
                        job_info["completed_urls"] / job_info["total_urls"]
                    ) * 100
                    await self._update_job_status(
                        job_id,
                        "running",
                        {
                            "progress": progress,
                            "completed_urls": job_info["completed_urls"],
                            "successful_urls": job_info["successful_urls"],
                            "failed_urls": job_info["failed_urls"],
                        },
                    )

                    # Check if job was cancelled
                    if job_id not in self._active_jobs:
                        logger.info(f"Job {job_id} was cancelled")
                        return

                    # Add delay between batches if configured
                    if settings.crawler.delay > 0:
                        await asyncio.sleep(settings.crawler.delay)

            # Mark job as completed
            job_info["status"] = "completed"
            job_info["completed_at"] = datetime.utcnow()
            job_info["processing_time"] = time.time() - start_time

            await self._update_job_status(
                job_id,
                "completed",
                {
                    "completed_at": datetime.utcnow(),
                    "processing_time": job_info["processing_time"],
                    "final_results": {
                        "total_urls": job_info["total_urls"],
                        "successful_urls": job_info["successful_urls"],
                        "failed_urls": job_info["failed_urls"],
                    },
                },
            )

            logger.info(
                f"Completed job {job_id}: {job_info['successful_urls']}/{job_info['total_urls']} successful"
            )

        except Exception as e:
            error_msg = f"Job execution failed: {str(e)}"
            logger.error(f"Job {job_id} failed: {error_msg}")

            job_info["status"] = "failed"
            job_info["completed_at"] = datetime.utcnow()
            job_info["errors"].append(error_msg)

            await self._update_job_status(
                job_id,
                "failed",
                {"completed_at": datetime.utcnow(), "error_message": error_msg},
            )

        finally:
            # Move job to history and cleanup
            self._job_history[job_id] = job_info
            if job_id in self._active_jobs:
                del self._active_jobs[job_id]

            # Periodic cleanup
            await self._cleanup_old_jobs()

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific job.

        @description Retrieves comprehensive status information for a job
        including progress, statistics, and configuration
        @param job_id: Unique identifier of the job
        @returns: Job status dictionary or None if job not found

        @example
        # Get job status
        status = await job_manager.get_job_status("job-123")
        if status:
            print(f"Progress: {status['progress_percentage']}%")
        """
        # Check active jobs first
        if job_id in self._active_jobs:
            job_info = self._active_jobs[job_id]
            return self._format_job_status(job_info)

        # Check job history
        if job_id in self._job_history:
            job_info = self._job_history[job_id]
            return self._format_job_status(job_info)

        # Check database for persistent job info
        try:
            # Query database for job information
            # This is a simplified approach - in production you'd have dedicated job tables
            job_data = await db_manager.search_content(
                "job", filters={"job_id": job_id}, limit=1
            )

            if job_data:
                job_record = job_data[0]
                return {
                    "job_id": job_id,
                    "status": job_record.get("status", "unknown"),
                    "created_at": job_record.get("created_at"),
                    "started_at": job_record.get("metadata", {}).get("started_at"),
                    "completed_at": job_record.get("metadata", {}).get("completed_at"),
                    "total_urls": len(job_record.get("target_urls", [])),
                    "completed_urls": job_record.get("results_count", 0),
                    "successful_urls": job_record.get("results_count", 0)
                    - job_record.get("error_count", 0),
                    "failed_urls": job_record.get("error_count", 0),
                    "progress_percentage": (
                        100.0 if job_record.get("status") == "completed" else 0.0
                    ),
                    "errors": [],
                    "config": job_record.get("config", {}),
                }
        except Exception as e:
            logger.error(f"Error retrieving job {job_id} from database: {str(e)}")

        return None

    async def list_jobs(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[JobResponse]:
        """
        List jobs with optional filtering and pagination.

        @description Retrieves a list of jobs with support for status filtering
        and pagination for efficient job management
        @param status: Optional status filter
        @param limit: Maximum number of jobs to return
        @param offset: Number of jobs to skip
        @returns: List of job response objects

        @example
        # List running jobs
        jobs = await job_manager.list_jobs(status="running", limit=10)
        """
        jobs = []

        # Add active jobs
        for job_info in self._active_jobs.values():
            if not status or job_info["status"] == status:
                jobs.append(self._format_job_response(job_info))

        # Add historical jobs
        for job_info in self._job_history.values():
            if not status or job_info["status"] == status:
                jobs.append(self._format_job_response(job_info))

        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        return jobs[offset : offset + limit]

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        @description Attempts to cancel a job by removing it from active jobs
        @param job_id: Unique identifier of the job to cancel
        @returns: True if job was cancelled, False if not found or already finished

        @example
        # Cancel a job
        success = await job_manager.cancel_job("job-123")
        if success:
            print("Job cancelled successfully")
        """
        if job_id in self._active_jobs:
            job_info = self._active_jobs[job_id]
            job_info["status"] = "cancelled"
            job_info["completed_at"] = datetime.utcnow()

            # Update in database
            await self._update_job_status(
                job_id, "cancelled", {"completed_at": datetime.utcnow()}
            )

            # Move to history
            self._job_history[job_id] = job_info
            del self._active_jobs[job_id]

            logger.info(f"Cancelled job {job_id}")
            return True

        return False

    async def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive job statistics.

        @description Returns statistics about job processing for monitoring
        @returns: Dictionary containing job statistics

        @example
        # Get job stats
        stats = await job_manager.get_job_statistics()
        print(f"Active jobs: {stats['active_jobs']}")
        """
        total_jobs = len(self._active_jobs) + len(self._job_history)
        active_jobs = len(self._active_jobs)
        completed_jobs = len(
            [j for j in self._job_history.values() if j["status"] == "completed"]
        )
        failed_jobs = len(
            [j for j in self._job_history.values() if j["status"] == "failed"]
        )
        cancelled_jobs = len(
            [j for j in self._job_history.values() if j["status"] == "cancelled"]
        )

        return {
            "total_jobs": total_jobs,
            "active_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "cancelled_jobs": cancelled_jobs,
            "success_rate": completed_jobs / max(total_jobs - active_jobs, 1) * 100,
            "avg_urls_per_job": sum(j["total_urls"] for j in self._job_history.values())
            / max(len(self._job_history), 1),
        }

    def _format_job_status(self, job_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format job information for status response.

        @description Converts internal job info to API response format
        @param job_info: Internal job information dictionary
        @returns: Formatted job status dictionary
        """
        progress = 0.0
        if job_info["total_urls"] > 0:
            progress = (job_info["completed_urls"] / job_info["total_urls"]) * 100

        estimated_remaining = None
        if job_info["status"] == "running" and job_info["completed_urls"] > 0:
            elapsed = (
                time.time() - job_info.get("started_at", datetime.utcnow()).timestamp()
            )
            rate = job_info["completed_urls"] / elapsed
            remaining_urls = job_info["total_urls"] - job_info["completed_urls"]
            estimated_remaining = remaining_urls / rate if rate > 0 else None

        return {
            "job_id": job_info["job_id"],
            "status": job_info["status"],
            "created_at": job_info.get("config", {}).get(
                "created_at", datetime.utcnow()
            ),
            "started_at": job_info.get("started_at"),
            "completed_at": job_info.get("completed_at"),
            "total_urls": job_info["total_urls"],
            "completed_urls": job_info["completed_urls"],
            "successful_urls": job_info["successful_urls"],
            "failed_urls": job_info["failed_urls"],
            "progress_percentage": progress,
            "estimated_remaining_time": estimated_remaining,
            "errors": job_info.get("errors", []),
            "config": job_info.get("config", {}),
        }

    def _format_job_response(self, job_info: Dict[str, Any]) -> JobResponse:
        """
        Format job information for job response.

        @description Converts internal job info to JobResponse model
        @param job_info: Internal job information dictionary
        @returns: JobResponse object
        """
        return JobResponse(
            job_id=job_info["job_id"],
            status=job_info["status"],
            created_at=job_info.get("config", {}).get("created_at", datetime.utcnow()),
            total_urls=job_info["total_urls"],
            completed_urls=job_info["completed_urls"],
            message=f"Job {job_info['status']}: {job_info['completed_urls']}/{job_info['total_urls']} URLs processed",
        )

    async def _update_job_status(
        self, job_id: str, status: str, metadata: Dict[str, Any]
    ) -> None:
        """
        Update job status in database.

        @description Updates job status and metadata in persistent storage
        @param job_id: Job identifier
        @param status: New job status
        @param metadata: Additional metadata to store
        """
        try:
            # This is a simplified approach - in production you'd have dedicated job tables
            # For now, we update the job record created during job creation
            pass  # Implementation would depend on specific database schema
        except Exception as e:
            logger.error(f"Failed to update job status in database: {str(e)}")

    async def _store_job_result(self, job_id: str, result) -> None:
        """
        Store individual crawl result for a job.

        @description Stores the result of a single URL crawl within a job
        @param job_id: Job identifier
        @param result: Crawl result to store
        """
        try:
            # Store job result in database
            # This would typically go into a job_results table
            pass  # Implementation would depend on specific database schema
        except Exception as e:
            logger.error(f"Failed to store job result: {str(e)}")

    async def _cleanup_old_jobs(self) -> None:
        """
        Clean up old job history to prevent memory bloat.

        @description Removes old completed jobs from memory cache
        """
        current_time = time.time()

        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        # Remove jobs older than 24 hours from memory
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        jobs_to_remove = []

        for job_id, job_info in self._job_history.items():
            completed_at = job_info.get("completed_at")
            if completed_at and completed_at < cutoff_time:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self._job_history[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old job records from memory")

        self._last_cleanup = current_time


def validate_content_type(content_type: str) -> bool:
    """
    Validate if content type is supported.

    @description Checks if the provided content type is supported by the system
    @param content_type: Content type to validate
    @returns: True if content type is supported, False otherwise

    @example
    # Validate content type
    if validate_content_type("blog"):
        print("Blog content type is supported")
    """
    return content_type in CONTENT_TYPE_MODELS


def validate_selectors(selectors: Dict[str, str]) -> List[str]:
    """
    Validate CSS selectors for common issues.

    @description Checks CSS selectors for syntax issues and potential problems
    @param selectors: Dictionary of field names to CSS selectors
    @returns: List of validation error messages (empty if valid)

    @example
    # Validate selectors
    errors = validate_selectors({"title": "h1", "content": ".invalid["})
    if errors:
        print(f"Selector errors: {errors}")
    """
    errors = []

    if not selectors or not isinstance(selectors, dict):
        return ["Selectors must be a non-empty dictionary"]

    for field_name, selector in selectors.items():
        if not isinstance(selector, str):
            errors.append(f"Selector for '{field_name}' must be a string")
            continue

        if not selector.strip():
            errors.append(f"Selector for '{field_name}' cannot be empty")
            continue

        # Basic CSS selector validation
        if selector.count("[") != selector.count("]"):
            errors.append(
                f"Unmatched brackets in selector for '{field_name}': {selector}"
            )

        if selector.count("(") != selector.count(")"):
            errors.append(
                f"Unmatched parentheses in selector for '{field_name}': {selector}"
            )

        # Check for suspicious patterns
        suspicious_patterns = ["<script", "javascript:", "data:"]
        if any(pattern in selector.lower() for pattern in suspicious_patterns):
            errors.append(
                f"Suspicious content in selector for '{field_name}': {selector}"
            )

    return errors


def validate_url_list(urls: List[str], max_count: int = 1000) -> List[str]:
    """
    Validate a list of URLs for batch processing.

    @description Validates URLs for format, protocol, and count limits
    @param urls: List of URLs to validate
    @param max_count: Maximum allowed URL count
    @returns: List of validation error messages (empty if valid)

    @example
    # Validate URL list
    urls = ["https://example.com", "invalid-url"]
    errors = validate_url_list(urls, max_count=10)
    """
    errors = []

    if not urls:
        return ["URL list cannot be empty"]

    if len(urls) > max_count:
        errors.append(f"Too many URLs: {len(urls)}. Maximum allowed: {max_count}")

    seen_urls = set()
    for i, url in enumerate(urls):
        if not isinstance(url, str):
            errors.append(f"URL at index {i} must be a string")
            continue

        url = url.strip()
        if not url:
            errors.append(f"URL at index {i} cannot be empty")
            continue

        if not url.startswith(("http://", "https://")):
            errors.append(f"URL at index {i} must use http or https protocol: {url}")
            continue

        if url in seen_urls:
            errors.append(f"Duplicate URL found: {url}")
        else:
            seen_urls.add(url)

        # Basic URL format validation
        if " " in url:
            errors.append(f"URL at index {i} contains spaces: {url}")

        if len(url) > 2048:
            errors.append(f"URL at index {i} is too long (max 2048 characters)")

    return errors


def format_error_response(
    error: Exception, request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format exception into standardized error response.

    @description Converts exceptions into consistent error response format
    @param error: Exception to format
    @param request_id: Optional request identifier for tracking
    @returns: Formatted error response dictionary

    @example
    # Format error response
    try:
        # Some operation
        pass
    except Exception as e:
        error_response = format_error_response(e, "req_123")
    """
    return {
        "error": type(error).__name__,
        "detail": str(error),
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id or str(uuid.uuid4()),
    }


def calculate_rate_limit_delay(
    domain: str, last_request_times: Dict[str, float]
) -> float:
    """
    Calculate delay needed for rate limiting.

    @description Determines how long to wait before making next request to domain
    @param domain: Domain to check rate limit for
    @param last_request_times: Dictionary tracking last request times per domain
    @returns: Delay in seconds (0 if no delay needed)

    @example
    # Calculate rate limit delay
    delay = calculate_rate_limit_delay("example.com", request_times)
    if delay > 0:
        await asyncio.sleep(delay)
    """
    current_time = time.time()
    min_delay = settings.crawler.delay

    if domain in last_request_times:
        time_since_last = current_time - last_request_times[domain]
        if time_since_last < min_delay:
            return min_delay - time_since_last

    return 0.0


def extract_domain_from_url(url: str) -> Optional[str]:
    """
    Extract domain from URL safely.

    @description Safely extracts domain name from URL with error handling
    @param url: URL to extract domain from
    @returns: Domain name or None if extraction fails

    @example
    # Extract domain
    domain = extract_domain_from_url("https://example.com/path")
    # Returns: "example.com"
    """
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


def generate_unique_id() -> str:
    """
    Generate a unique identifier.

    @description Creates a unique identifier for jobs, requests, etc.
    @returns: Unique identifier string

    @example
    # Generate unique ID
    job_id = generate_unique_id()
    # Returns: "123e4567-e89b-12d3-a456-426614174000"
    """
    return str(uuid.uuid4())
