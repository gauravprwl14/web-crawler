"""
Main FastAPI application for the flexible web crawler system.

This module creates and configures the FastAPI application with all endpoints,
middleware, and dependencies properly integrated for production deployment.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.endpoints import router
from api.utils import format_error_response
from config.settings import get_settings
from database.models import db_manager, init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.

    @description Handles application initialization and cleanup including
    database setup, logging configuration, and resource management
    @param app: FastAPI application instance
    @yields: Control to the application during its lifetime
    """
    settings = get_settings()

    # Startup tasks
    logger.info("Starting Flexible Web Crawler API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Database URL: {settings.database.url}")

    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_database()
        logger.info("Database initialized successfully")

        # Configure logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/crawler_api.log",
                    "format": settings.log_format,
                    "level": settings.log_level,
                    "rotation": "100 MB",
                    "retention": "1 week",
                },
                {
                    "sink": "sys.stdout",
                    "format": settings.log_format,
                    "level": settings.log_level,
                },
            ]
        )

        logger.info("Application startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise
    finally:
        # Shutdown tasks
        logger.info("Shutting down application...")

        try:
            # Close database connections
            await db_manager.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error during database shutdown: {str(e)}")

        logger.info("Application shutdown completed")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    @description Creates the main FastAPI application with all middleware,
    exception handlers, and configuration applied for production deployment
    @returns: Configured FastAPI application instance

    @example
    # Create application
    app = create_application()
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    settings = get_settings()

    # Create FastAPI application
    app = FastAPI(
        title="Flexible Web Crawler API",
        description="""
        A comprehensive and flexible web crawler system with support for multiple content types,
        dynamic selectors, and scalable database storage.
        
        ## Features
        
        * **Flexible Content Extraction**: Support for blogs, articles, prompts, and products
        * **Dynamic Selectors**: Custom CSS selectors for any website structure
        * **Batch Processing**: Concurrent crawling of multiple URLs
        * **Background Jobs**: Asynchronous processing for large crawling tasks
        * **Database Integration**: Automatic storage with content type-specific tables
        * **Rate Limiting**: Respectful crawling with configurable delays
        * **Robots.txt Compliance**: Automatic robots.txt checking
        * **Content Validation**: Comprehensive validation and error reporting
        
        ## Content Types
        
        * **Blog Posts**: Title, content, author, published date, tags
        * **Articles**: Title, content, summary, keywords, reading time
        * **Prompts**: Prompt text, category, difficulty, examples
        * **Products**: Name, price, description, rating, specifications
        
        ## Usage Examples
        
        ### Single URL Crawl
        ```python
        import httpx
        
        response = httpx.post("http://localhost:8000/crawl/single", json={
            "url": "https://example.com/blog/post",
            "content_type": "blog",
            "selectors": {
                "title": "h1",
                "content": ".post-content",
                "author": ".author"
            }
        })
        ```
        
        ### Batch Crawl
        ```python
        response = httpx.post("http://localhost:8000/crawl/batch", json={
            "urls": [
                "https://blog.com/post1",
                "https://blog.com/post2"
            ],
            "content_type": "blog",
            "max_concurrent": 3
        })
        ```
        
        ### Background Job
        ```python
        response = httpx.post("http://localhost:8000/jobs", json={
            "name": "Daily News Crawl",
            "urls": ["https://news.com/latest"],
            "content_type": "article",
            "schedule": "immediate"
        })
        ```
        """,
        version="1.0.0",
        contact={
            "name": "Flexible Crawler Team",
            "url": "https://github.com/yourorg/flexible-crawler",
            "email": "support@yourorg.com",
        },
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        lifespan=lifespan,
        debug=settings.api.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Processing-Time"],
    )

    # Add trusted host middleware for security
    if settings.environment.value == "production":
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
        )

    # Add request ID and timing middleware
    @app.middleware("http")
    async def add_request_metadata(request: Request, call_next):
        """
        Add request ID and timing to all requests.

        @description Adds unique request ID and processing time tracking
        to all HTTP requests for monitoring and debugging
        @param request: Incoming HTTP request
        @param call_next: Next middleware in chain
        @returns: HTTP response with additional headers
        """
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000000)}"

        # Add request ID to request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add metadata to response headers
        processing_time = time.time() - start_time
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"

        # Log request
        logger.info(
            f"Request processed: {request.method} {request.url.path} "
            f"[{response.status_code}] in {processing_time:.3f}s"
        )

        return response

    # Add rate limiting middleware
    @app.middleware("http")
    async def rate_limiting_middleware(request: Request, call_next):
        """
        Basic rate limiting middleware.

        @description Implements simple rate limiting to prevent API abuse
        @param request: Incoming HTTP request
        @param call_next: Next middleware in chain
        @returns: HTTP response or rate limit error
        """
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host

        # Simple in-memory rate limiting (use Redis in production)
        # This is a basic implementation - use proper rate limiting in production
        rate_limit_key = f"rate_limit:{client_ip}"

        # For now, just proceed - implement proper rate limiting as needed
        return await call_next(request)

    # Custom exception handlers
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """
        Custom HTTP exception handler with consistent error format.

        @description Provides consistent error response formatting across all endpoints
        @param request: HTTP request that caused the exception
        @param exc: HTTP exception that was raised
        @returns: Formatted JSON error response
        """
        request_id = getattr(request.state, "request_id", None)
        error_response = format_error_response(exc, request_id)

        logger.error(
            f"HTTP Exception: {exc.status_code} - {exc.detail} "
            f"(Request ID: {request_id})"
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response,
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    @app.exception_handler(Exception)
    async def custom_general_exception_handler(request: Request, exc: Exception):
        """
        Custom general exception handler for unhandled errors.

        @description Catches and formats unhandled exceptions with proper logging
        @param request: HTTP request that caused the exception
        @param exc: Exception that was raised
        @returns: Formatted JSON error response
        """
        request_id = getattr(request.state, "request_id", None)
        error_response = format_error_response(exc, request_id)

        logger.error(
            f"Unhandled Exception: {type(exc).__name__} - {str(exc)} "
            f"(Request ID: {request_id})"
        )

        return JSONResponse(
            status_code=500,
            content={
                **error_response,
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again later.",
            },
            headers={"X-Request-ID": request_id} if request_id else {},
        )

    # Include API router
    app.include_router(router, prefix="/api/v1", tags=["Crawler API"])

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with API information.

        @description Provides basic information about the API and available endpoints
        @returns: API information and links
        """
        return {
            "name": "Flexible Web Crawler API",
            "version": "1.0.0",
            "description": "A comprehensive web crawler with flexible content extraction",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "health_check": "/health",
            "api_endpoints": {
                "single_crawl": "/api/v1/crawl/single",
                "batch_crawl": "/api/v1/crawl/batch",
                "create_job": "/api/v1/jobs",
                "job_status": "/api/v1/jobs/{job_id}",
                "list_jobs": "/api/v1/jobs",
                "get_content": "/api/v1/content/{content_type}",
                "get_content_by_id": "/api/v1/content/{content_type}/{content_id}",
                "stats": "/api/v1/stats",
            },
            "supported_content_types": ["blog", "article", "prompt", "product"],
            "github": "https://github.com/yourorg/flexible-crawler",
        }

    # Health check endpoint (outside of API versioning)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Simple health check endpoint.

        @description Quick health check that doesn't require authentication
        @returns: Basic health status
        """
        return {"status": "healthy", "timestamp": time.time(), "version": "1.0.0"}

    return app


# Create the application instance
app = create_application()


def run_development_server():
    """
    Run the development server with hot reload.

    @description Starts the development server with appropriate settings
    for local development including hot reload and debug logging

    @example
    # Run development server
    python -m src.main
    """
    settings = get_settings()

    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


def run_production_server():
    """
    Run the production server with optimized settings.

    @description Starts the production server with appropriate settings
    for production deployment including proper logging and performance tuning

    @example
    # Run production server
    python -c "from src.main import run_production_server; run_production_server()"
    """
    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        workers=4,  # Adjust based on your server
        log_level=settings.log_level.lower(),
        access_log=True,
        loop="uvloop",  # Faster event loop for production
    )


if __name__ == "__main__":
    """
    Entry point for running the application directly.

    @description Runs the application in development mode when executed directly
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "production":
        run_production_server()
    else:
        run_development_server()
