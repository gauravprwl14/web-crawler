"""
FastAPI endpoints for the flexible web crawler system.

This module provides REST API endpoints for web crawling operations,
job management, and content retrieval with comprehensive error handling.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, validator

from api.schemas import *
from api.utils import JobManager, validate_content_type
from config.settings import get_settings
from crawler.core import CrawlRequest, CrawlResult, FlexibleWebCrawler
from database.models import ScrapingJob, ScrapingResult, db_manager, get_db_session

settings = get_settings()
router = APIRouter()
job_manager = JobManager()


@router.post("/crawl/single", response_model=CrawlResponse)
async def crawl_single_url(
    request: SingleCrawlRequest, background_tasks: BackgroundTasks
) -> CrawlResponse:
    """
    Crawl a single URL and return the extracted content.

    @description Performs immediate crawling of a single URL with the specified
    content type and selectors, returning processed content and metadata
    @param request: Single crawl request with URL, content type, and options
    @param background_tasks: FastAPI background tasks for async processing
    @returns: Crawl response with extracted content and processing status
    @throws HTTPException: If crawl request is invalid or crawling fails

    @example
    POST /crawl/single
    {
        "url": "https://example.com/blog/post",
        "content_type": "blog",
        "selectors": {"title": "h1", "content": ".post-content"},
        "store_in_db": true
    }
    """
    try:
        logger.info(f"Starting single crawl for URL: {request.url}")

        # Validate content type
        if not validate_content_type(request.content_type):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {request.content_type}",
            )

        # Create crawler instance
        async with FlexibleWebCrawler(request.crawler_config) as crawler:
            # Create crawl request
            crawl_request = CrawlRequest(
                url=request.url,
                content_type=request.content_type,
                selectors=request.selectors,
                custom_config=request.crawler_config,
                metadata=request.metadata,
                validate_content=request.validate_content,
                store_raw_data=request.store_raw_data,
            )

            # Perform crawl
            result = await crawler.crawl_single(crawl_request)

            # Store in database if requested and successful
            if request.store_in_db and result.success and result.content_id:
                logger.info(f"Content stored with ID: {result.content_id}")

            # Prepare response
            response = CrawlResponse(
                success=result.success,
                url=result.url,
                content_type=result.content_type,
                extracted_data=result.extracted_data,
                errors=result.errors or [],
                processing_time=result.processing_time,
                content_id=result.content_id,
                validation_errors=result.validation_errors,
                metadata=result.metadata,
            )

            logger.info(
                f"Completed single crawl for {request.url}: {'success' if result.success else 'failed'}"
            )
            return response

    except Exception as e:
        logger.error(f"Error in single crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@router.post("/crawl/batch", response_model=BatchCrawlResponse)
async def crawl_batch_urls(
    request: BatchCrawlRequest, background_tasks: BackgroundTasks
) -> BatchCrawlResponse:
    """
    Crawl multiple URLs concurrently and return results.

    @description Processes multiple URLs concurrently with configurable
    concurrency limits and comprehensive error handling per URL
    @param request: Batch crawl request with multiple URLs and settings
    @param background_tasks: FastAPI background tasks for async processing
    @returns: Batch crawl response with results for each URL
    @throws HTTPException: If batch request is invalid or processing fails

    @example
    POST /crawl/batch
    {
        "urls": ["https://blog.com/post1", "https://blog.com/post2"],
        "content_type": "blog",
        "max_concurrent": 3,
        "selectors": {"title": "h1", "content": ".content"}
    }
    """
    try:
        logger.info(f"Starting batch crawl for {len(request.urls)} URLs")

        # Validate content type
        if not validate_content_type(request.content_type):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {request.content_type}",
            )

        # Validate URL count
        if len(request.urls) > 100:  # Configurable limit
            raise HTTPException(
                status_code=400, detail="Too many URLs in batch request (max: 100)"
            )

        # Create crawler instance
        async with FlexibleWebCrawler(request.crawler_config) as crawler:
            # Create crawl requests
            crawl_requests = []
            for url in request.urls:
                crawl_request = CrawlRequest(
                    url=url,
                    content_type=request.content_type,
                    selectors=request.selectors,
                    custom_config=request.crawler_config,
                    metadata=request.metadata,
                    validate_content=request.validate_content,
                    store_raw_data=request.store_raw_data,
                )
                crawl_requests.append(crawl_request)

            # Perform batch crawl
            results = await crawler.crawl_multiple(
                crawl_requests,
                max_concurrent=request.max_concurrent
                or settings.crawler.concurrent_requests,
            )

            # Prepare response
            crawl_results = []
            successful_count = 0

            for result in results:
                crawl_response = CrawlResponse(
                    success=result.success,
                    url=result.url,
                    content_type=result.content_type,
                    extracted_data=result.extracted_data,
                    errors=result.errors or [],
                    processing_time=result.processing_time,
                    content_id=result.content_id,
                    validation_errors=result.validation_errors,
                    metadata=result.metadata,
                )
                crawl_results.append(crawl_response)

                if result.success:
                    successful_count += 1

            response = BatchCrawlResponse(
                success=True,
                total_urls=len(request.urls),
                successful_count=successful_count,
                failed_count=len(request.urls) - successful_count,
                results=crawl_results,
                processing_time=sum(r.processing_time for r in results),
            )

            logger.info(
                f"Completed batch crawl: {successful_count}/{len(request.urls)} successful"
            )
            return response

    except Exception as e:
        logger.error(f"Error in batch crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch crawl failed: {str(e)}")


@router.post("/jobs", response_model=JobResponse)
async def create_crawl_job(
    request: JobCreateRequest, background_tasks: BackgroundTasks
) -> JobResponse:
    """
    Create a background crawling job for processing multiple URLs.

    @description Creates an asynchronous crawling job that processes URLs
    in the background and stores results in the database
    @param request: Job creation request with URLs and processing options
    @param background_tasks: FastAPI background tasks for async execution
    @returns: Job response with job ID and initial status
    @throws HTTPException: If job creation fails or request is invalid

    @example
    POST /jobs
    {
        "name": "Blog Crawl Job",
        "urls": ["https://blog.com/post1", "https://blog.com/post2"],
        "content_type": "blog",
        "schedule": "immediate"
    }
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Creating crawl job {job_id} with {len(request.urls)} URLs")

        # Validate content type
        if not validate_content_type(request.content_type):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {request.content_type}",
            )

        # Create job in database
        job_data = {
            "job_id": job_id,
            "url": f"job://{job_id}",  # Special URL for job tracking
            "content_type": "job",
            "status": "pending",
            "target_urls": request.urls,
            "selectors": request.selectors,
            "config": {
                "name": request.name,
                "content_type": request.content_type,
                "crawler_config": request.crawler_config,
                "validate_content": request.validate_content,
                "store_raw_data": request.store_raw_data,
                "metadata": request.metadata,
            },
        }

        # Insert job into database
        await db_manager.insert_content("job", job_data)

        # Schedule job execution
        if request.schedule == "immediate":
            background_tasks.add_task(job_manager.execute_job, job_id, request)

        response = JobResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow(),
            total_urls=len(request.urls),
            completed_urls=0,
            message=f"Job created with {len(request.urls)} URLs",
        )

        logger.info(f"Created crawl job {job_id}")
        return response

    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job creation failed: {str(e)}")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str = Path(..., description="Job ID to check status for")
) -> JobStatusResponse:
    """
    Get the status and progress of a crawling job.

    @description Retrieves detailed status information about a specific
    crawling job including progress, results, and any errors
    @param job_id: Unique identifier of the job to check
    @returns: Detailed job status with progress and results information
    @throws HTTPException: If job is not found or status check fails

    @example
    GET /jobs/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        # Get job status from job manager
        status = await job_manager.get_job_status(job_id)

        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobStatusResponse(**status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(
        50, ge=1, le=100, description="Maximum number of jobs to return"
    ),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
) -> List[JobResponse]:
    """
    List crawling jobs with optional filtering and pagination.

    @description Retrieves a list of crawling jobs with support for
    status filtering and pagination for efficient job management
    @param status: Optional status filter (pending, running, completed, failed)
    @param limit: Maximum number of jobs to return (1-100)
    @param offset: Number of jobs to skip for pagination
    @returns: List of job responses with basic information
    @throws HTTPException: If listing fails or invalid parameters

    @example
    GET /jobs?status=completed&limit=10&offset=0
    """
    try:
        jobs = await job_manager.list_jobs(status=status, limit=limit, offset=offset)
        return jobs

    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str = Path(..., description="Job ID to cancel")
) -> Dict[str, str]:
    """
    Cancel a running or pending crawling job.

    @description Attempts to cancel a job that is currently running or pending
    @param job_id: Unique identifier of the job to cancel
    @returns: Cancellation confirmation message
    @throws HTTPException: If job is not found or cannot be cancelled

    @example
    DELETE /jobs/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        success = await job_manager.cancel_job(job_id)

        if not success:
            raise HTTPException(
                status_code=404, detail="Job not found or cannot be cancelled"
            )

        return {"message": f"Job {job_id} cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Job cancellation failed: {str(e)}"
        )


@router.get("/content/{content_type}", response_model=List[Dict[str, Any]])
async def get_content(
    content_type: str = Path(..., description="Type of content to retrieve"),
    limit: int = Query(
        50, ge=1, le=1000, description="Maximum number of items to return"
    ),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    filters: Optional[str] = Query(None, description="JSON string of field filters"),
) -> List[Dict[str, Any]]:
    """
    Retrieve stored content by type with filtering and pagination.

    @description Fetches stored content from the database with support for
    filtering, pagination, and content type selection
    @param content_type: Type of content to retrieve (blog, prompt, article, product)
    @param limit: Maximum number of items to return (1-1000)
    @param offset: Number of items to skip for pagination
    @param filters: Optional JSON string of field filters
    @returns: List of content items matching the criteria
    @throws HTTPException: If content type is invalid or retrieval fails

    @example
    GET /content/blog?limit=10&offset=0&filters={"author":"John Doe"}
    """
    try:
        # Validate content type
        if not validate_content_type(content_type):
            raise HTTPException(
                status_code=400, detail=f"Unsupported content type: {content_type}"
            )

        # Parse filters if provided
        filter_dict = None
        if filters:
            import json

            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid JSON in filters parameter"
                )

        # Retrieve content from database
        content = await db_manager.search_content(
            content_type=content_type, filters=filter_dict, limit=limit, offset=offset
        )

        return content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Content retrieval failed: {str(e)}"
        )


@router.get("/content/{content_type}/{content_id}", response_model=Dict[str, Any])
async def get_content_by_id(
    content_type: str = Path(..., description="Type of content"),
    content_id: str = Path(..., description="Unique identifier of the content"),
) -> Dict[str, Any]:
    """
    Retrieve specific content item by ID and type.

    @description Fetches a specific content item using its unique identifier
    and content type for detailed viewing and processing
    @param content_type: Type of content (blog, prompt, article, product)
    @param content_id: Unique identifier of the content item
    @returns: Detailed content item with all fields and metadata
    @throws HTTPException: If content is not found or retrieval fails

    @example
    GET /content/blog/123e4567-e89b-12d3-a456-426614174000
    """
    try:
        # Validate content type
        if not validate_content_type(content_type):
            raise HTTPException(
                status_code=400, detail=f"Unsupported content type: {content_type}"
            )

        # Retrieve content from database
        content = await db_manager.get_content(content_type, content_id)

        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        return content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content by ID: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Content retrieval failed: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_crawler_stats() -> Dict[str, Any]:
    """
    Get comprehensive crawler system statistics.

    @description Retrieves system-wide statistics including crawl performance,
    content type distribution, and database statistics
    @returns: Dictionary containing various system statistics
    @throws HTTPException: If statistics retrieval fails

    @example
    GET /stats
    {
        "total_content_items": 1234,
        "content_by_type": {"blog": 800, "article": 434},
        "crawler_performance": {"avg_processing_time": 2.5}
    }
    """
    try:
        # Get database statistics
        db_stats = {}
        content_types = ["blog", "prompt", "article", "product"]

        total_items = 0
        for content_type in content_types:
            try:
                items = await db_manager.search_content(content_type, limit=1, offset=0)
                # This is a simple count - in production, you'd want a proper count query
                count_result = await db_manager.search_content(
                    content_type, limit=10000
                )
                count = len(count_result)
                db_stats[content_type] = count
                total_items += count
            except:
                db_stats[content_type] = 0

        # Get job statistics
        job_stats = await job_manager.get_job_statistics()

        stats = {
            "database": {
                "total_content_items": total_items,
                "content_by_type": db_stats,
            },
            "jobs": job_stats,
            "system": {
                "supported_content_types": content_types,
                "api_version": "1.0.0",
                "settings": {
                    "max_concurrent_requests": settings.crawler.concurrent_requests,
                    "default_timeout": settings.crawler.timeout,
                    "respect_robots_txt": settings.crawler.respect_robots_txt,
                },
            },
        }

        return stats

    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Statistics retrieval failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Perform system health check.

    @description Checks the health of various system components including
    database connectivity and crawler functionality
    @returns: Health status information

    @example
    GET /health
    {"status": "healthy", "database": "connected", "timestamp": "2024-01-15T10:30:00Z"}
    """
    try:
        # Test database connectivity
        db_status = "connected"
        try:
            # Simple database test
            await db_manager.search_content("blog", limit=1)
        except Exception:
            db_status = "disconnected"

        # Test crawler functionality
        crawler_status = "available"
        try:
            # Simple crawler test
            from ..crawler.utils import URLValidator

            validator = URLValidator()
            validator.is_valid_url("https://example.com")
        except Exception:
            crawler_status = "unavailable"

        overall_status = (
            "healthy"
            if db_status == "connected" and crawler_status == "available"
            else "degraded"
        )

        return {
            "status": overall_status,
            "database": db_status,
            "crawler": crawler_status,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
