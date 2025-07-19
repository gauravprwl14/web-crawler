"""
Core web crawler module with flexible selector support and database integration.

This module provides the main crawler functionality built on top of crawl4ai,
with enhanced selector capabilities, content processing, and automatic database storage.
"""

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from config.settings import get_settings
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import CosineStrategy, LLMExtractionStrategy
from crawler.processors import ContentProcessorFactory
from crawler.utils import RobotsTxtChecker, URLValidator
from database.models import ScrapingJob, ScrapingResult, db_manager

settings = get_settings()


@dataclass
class CrawlRequest:
    """
    Data class representing a crawl request with all necessary parameters.

    @description Encapsulates all parameters needed for a web crawling operation
    including URL, selectors, content type, and processing options
    """

    url: str
    content_type: str
    selectors: Optional[Dict[str, str]] = None
    custom_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    validate_content: bool = True
    store_raw_data: bool = True


@dataclass
class CrawlResult:
    """
    Data class representing the result of a crawl operation.

    @description Contains all information about a completed crawl operation
    including extracted content, metadata, and processing status
    """

    url: str
    content_type: str
    success: bool
    extracted_data: Optional[Dict[str, Any]] = None
    raw_html: Optional[str] = None
    errors: Optional[List[str]] = None
    processing_time: float = 0.0
    content_id: Optional[str] = None
    validation_errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FlexibleWebCrawler:
    """
    Enhanced web crawler with flexible selector support and database integration.

    @description Main crawler class that extends crawl4ai functionality with
    advanced content extraction, processing, and automatic database storage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the flexible web crawler.

        @description Sets up the crawler with configuration, processors, and utilities
        @param config: Optional configuration overrides for crawler behavior
        """
        self.settings = settings
        self.config = config or {}
        self.content_processor_factory = ContentProcessorFactory()
        self.url_validator = URLValidator()
        self.robots_checker = RobotsTxtChecker()

        # Initialize crawl4ai crawler with settings
        self.crawler_config = {
            "verbose": self.settings.api.debug,
            "headless": True,
            "browser_type": "chromium",
            "delay_before_return_html": 2.0,
            "magic": True,  # Enable smart waiting
        }
        self.crawler_config.update(self.config)

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "content_types_processed": {},
        }

    async def crawl_single(self, request: CrawlRequest) -> CrawlResult:
        """
        Crawl a single URL with the specified parameters.

        @description Performs a complete crawl operation for a single URL including
        content extraction, processing, validation, and database storage
        @param request: CrawlRequest object containing all crawl parameters
        @returns: CrawlResult object with extraction results and metadata
        @throws Exception: If crawling fails catastrophically

        @example
        # Crawl a blog post
        request = CrawlRequest(
            url="https://example.com/blog/post",
            content_type="blog",
            selectors={"title": "h1", "content": ".post-content"}
        )
        result = await crawler.crawl_single(request)
        if result.success:
            print(f"Extracted: {result.extracted_data['title']}")
        """
        start_time = time.time()
        self.stats["total_requests"] += 1

        logger.info(f"Starting crawl for URL: {request.url}")

        # Validate URL
        if not self.url_validator.is_valid_url(request.url):
            error_msg = f"Invalid URL: {request.url}"
            logger.error(error_msg)
            return CrawlResult(
                url=request.url,
                content_type=request.content_type,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time,
            )

        # Check robots.txt if enabled
        if self.settings.crawler.respect_robots_txt:
            if not await self.robots_checker.can_fetch(
                request.url, self.settings.crawler.user_agent
            ):
                error_msg = f"Robots.txt disallows crawling: {request.url}"
                logger.warning(error_msg)
                return CrawlResult(
                    url=request.url,
                    content_type=request.content_type,
                    success=False,
                    errors=[error_msg],
                    processing_time=time.time() - start_time,
                )

        try:
            # Perform the actual crawl
            crawl_result = await self._perform_crawl(request)

            # Process the content if crawl was successful
            if crawl_result.success and crawl_result.raw_html:
                processed_result = await self._process_content(request, crawl_result)

                # Store in database if processing was successful
                if processed_result.success and processed_result.extracted_data:
                    await self._store_content(request, processed_result)

                return processed_result
            else:
                self.stats["failed_requests"] += 1
                return crawl_result

        except Exception as e:
            error_msg = f"Crawl failed for {request.url}: {str(e)}"
            logger.error(error_msg)
            self.stats["failed_requests"] += 1

            return CrawlResult(
                url=request.url,
                content_type=request.content_type,
                success=False,
                errors=[error_msg],
                processing_time=time.time() - start_time,
            )
        finally:
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            logger.info(f"Completed crawl for {request.url} in {processing_time:.2f}s")

    async def _perform_crawl(self, request: CrawlRequest) -> CrawlResult:
        """
        Perform the actual web crawling using crawl4ai.

        @description Executes the web crawling operation using crawl4ai with
        configured parameters and handles various crawling scenarios
        @param request: CrawlRequest object with crawl parameters
        @returns: CrawlResult with raw HTML and basic metadata
        @throws Exception: If the underlying crawl operation fails
        """
        async with AsyncWebCrawler(**self.crawler_config) as crawler:
            try:
                # Prepare crawl parameters
                crawl_params = {
                    "url": request.url,
                    "word_count_threshold": 10,
                    "bypass_cache": True,
                    "process_iframes": True,
                    "remove_overlay_elements": True,
                }

                # Add custom configuration if provided
                if request.custom_config:
                    crawl_params.update(request.custom_config)

                # Execute the crawl
                logger.debug(f"Executing crawl with params: {crawl_params}")
                result = await crawler.arun(**crawl_params)

                if result.success:
                    logger.info(f"Successfully crawled {request.url}")
                    return CrawlResult(
                        url=request.url,
                        content_type=request.content_type,
                        success=True,
                        raw_html=result.html,
                        metadata={
                            "status_code": getattr(result, "status_code", None),
                            "response_headers": getattr(result, "response_headers", {}),
                            "page_title": getattr(result, "title", ""),
                            "links_found": len(getattr(result, "links", [])),
                            "images_found": len(getattr(result, "images", [])),
                            "crawl_timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                else:
                    error_msg = f"Crawl4ai failed for {request.url}: {getattr(result, 'error_message', 'Unknown error')}"
                    logger.error(error_msg)
                    return CrawlResult(
                        url=request.url,
                        content_type=request.content_type,
                        success=False,
                        errors=[error_msg],
                    )

            except Exception as e:
                error_msg = f"Exception during crawl: {str(e)}"
                logger.error(error_msg)
                return CrawlResult(
                    url=request.url,
                    content_type=request.content_type,
                    success=False,
                    errors=[error_msg],
                )

    async def _process_content(
        self, request: CrawlRequest, crawl_result: CrawlResult
    ) -> CrawlResult:
        """
        Process crawled content using selectors and content processors.

        @description Extracts structured data from raw HTML using CSS selectors
        and content-type specific processors for cleaning and validation
        @param request: Original crawl request with selectors and content type
        @param crawl_result: Result from the crawl operation containing raw HTML
        @returns: Updated CrawlResult with extracted and processed content
        @throws Exception: If content processing fails
        """
        try:
            logger.debug(f"Processing content for {request.url}")

            # Get content type configuration
            content_config = self.settings.get_content_type_config(request.content_type)

            # Use provided selectors or fall back to default ones
            selectors = request.selectors or content_config.get("selectors", {})

            # Extract data using selectors
            extracted_data = await self._extract_with_selectors(
                crawl_result.raw_html, selectors, request.url
            )

            # Get content processor for this content type
            processor = self.content_processor_factory.get_processor(
                request.content_type
            )

            # Process and validate the extracted data
            processed_data = await processor.process(extracted_data, request.url)

            # Validate required fields
            validation_errors = self._validate_required_fields(
                processed_data, content_config.get("required_fields", [])
            )

            # Update statistics
            content_type = request.content_type
            if content_type not in self.stats["content_types_processed"]:
                self.stats["content_types_processed"][content_type] = 0
            self.stats["content_types_processed"][content_type] += 1

            if validation_errors and request.validate_content:
                logger.warning(
                    f"Content validation failed for {request.url}: {validation_errors}"
                )
                return CrawlResult(
                    url=request.url,
                    content_type=request.content_type,
                    success=False,
                    errors=[
                        f"Content validation failed: {', '.join(validation_errors)}"
                    ],
                    validation_errors=validation_errors,
                    extracted_data=processed_data,
                    raw_html=crawl_result.raw_html if request.store_raw_data else None,
                    metadata=crawl_result.metadata,
                )
            else:
                self.stats["successful_requests"] += 1
                logger.info(f"Successfully processed content for {request.url}")
                return CrawlResult(
                    url=request.url,
                    content_type=request.content_type,
                    success=True,
                    extracted_data=processed_data,
                    raw_html=crawl_result.raw_html if request.store_raw_data else None,
                    validation_errors=validation_errors,
                    metadata=crawl_result.metadata,
                )

        except Exception as e:
            error_msg = f"Content processing failed: {str(e)}"
            logger.error(error_msg)
            return CrawlResult(
                url=request.url,
                content_type=request.content_type,
                success=False,
                errors=[error_msg],
                raw_html=crawl_result.raw_html if request.store_raw_data else None,
                metadata=crawl_result.metadata,
            )

    async def _extract_with_selectors(
        self, html: str, selectors: Dict[str, str], base_url: str
    ) -> Dict[str, Any]:
        """
        Extract data from HTML using CSS selectors.

        @description Uses BeautifulSoup to extract content based on provided CSS selectors
        with intelligent fallbacks and content cleaning
        @param html: Raw HTML content to process
        @param selectors: Dictionary mapping field names to CSS selectors
        @param base_url: Base URL for resolving relative links
        @returns: Dictionary of extracted data with field names as keys
        @throws Exception: If HTML parsing fails

        @example
        # Extract blog content
        selectors = {
            "title": "h1, .title",
            "content": ".post-content, article",
            "author": ".author, .byline"
        }
        data = await self._extract_with_selectors(html, selectors, "https://example.com")
        # Returns: {"title": "Blog Title", "content": "...", "author": "John Doe"}
        """
        soup = BeautifulSoup(html, "lxml")
        extracted_data = {}

        for field_name, selector in selectors.items():
            try:
                # Handle multiple selectors separated by commas
                selector_list = [s.strip() for s in selector.split(",")]

                element = None
                for sel in selector_list:
                    element = soup.select_one(sel)
                    if element:
                        break

                if element:
                    # Extract content based on element type
                    if field_name in ["url", "link", "href"]:
                        # Extract URL from href attribute
                        value = element.get("href", "")
                        if value and not value.startswith("http"):
                            value = urljoin(base_url, value)
                    elif field_name in ["image", "img", "src"]:
                        # Extract image URL from src attribute
                        value = element.get("src", "")
                        if value and not value.startswith("http"):
                            value = urljoin(base_url, value)
                    elif field_name in ["date", "published_date", "datetime"]:
                        # Extract datetime from datetime attribute or text
                        value = element.get("datetime") or element.get_text(strip=True)
                        value = self._parse_date(value)
                    elif field_name in ["tags", "categories"]:
                        # Extract multiple tags/categories
                        if element.name in ["ul", "ol"]:
                            value = [
                                li.get_text(strip=True) for li in element.find_all("li")
                            ]
                        else:
                            value = [
                                tag.strip() for tag in element.get_text().split(",")
                            ]
                    else:
                        # Extract text content
                        value = element.get_text(strip=True)

                        # Clean up whitespace
                        value = re.sub(r"\s+", " ", value)

                    extracted_data[field_name] = value
                    logger.debug(f"Extracted {field_name}: {str(value)[:100]}...")
                else:
                    logger.debug(
                        f"No element found for selector '{selector}' (field: {field_name})"
                    )
                    extracted_data[field_name] = None

            except Exception as e:
                logger.warning(
                    f"Failed to extract {field_name} with selector '{selector}': {str(e)}"
                )
                extracted_data[field_name] = None

        return extracted_data

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """
        Parse date string into datetime object.

        @description Attempts to parse various date formats into a standardized datetime object
        @param date_string: String representation of a date
        @returns: Parsed datetime object or None if parsing fails

        @example
        # Parse various date formats
        date1 = self._parse_date("2024-01-15T10:30:00Z")
        date2 = self._parse_date("January 15, 2024")
        date3 = self._parse_date("15/01/2024")
        """
        if not date_string:
            return None

        # Common date formats to try
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%d %B %Y",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_string}")
        return None

    def _validate_required_fields(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> List[str]:
        """
        Validate that required fields are present and not empty.

        @description Checks that all required fields are present in the extracted data
        and contain meaningful content (not empty or None)
        @param data: Extracted data dictionary to validate
        @param required_fields: List of field names that are required
        @returns: List of validation error messages (empty if validation passes)

        @example
        # Validate blog data
        required = ["title", "content", "url"]
        errors = self._validate_required_fields(blog_data, required)
        if errors:
            print(f"Validation failed: {errors}")
        """
        errors = []

        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Required field '{field}' is missing or empty")
            elif isinstance(data[field], str) and len(data[field].strip()) == 0:
                errors.append(f"Required field '{field}' is empty")

        return errors

    async def _store_content(self, request: CrawlRequest, result: CrawlResult) -> None:
        """
        Store extracted content in the database.

        @description Saves the processed content to the appropriate database table
        based on content type and updates the crawl result with the content ID
        @param request: Original crawl request
        @param result: Processed crawl result with extracted data
        @throws Exception: If database storage fails
        """
        try:
            # Prepare data for database insertion
            data_to_store = result.extracted_data.copy()
            data_to_store.update(
                {
                    "url": request.url,
                    "content_type": request.content_type,
                    "raw_data": {
                        "html": result.raw_html if request.store_raw_data else None,
                        "extraction_metadata": result.metadata,
                    },
                    "metadata": {
                        "crawl_timestamp": datetime.utcnow().isoformat(),
                        "job_id": request.job_id,
                        "processing_time": result.processing_time,
                        "selectors_used": request.selectors,
                        "validation_errors": result.validation_errors,
                    },
                }
            )

            # Insert into appropriate table
            content_id = await db_manager.insert_content(
                request.content_type, data_to_store
            )
            result.content_id = content_id

            logger.info(f"Stored content with ID: {content_id}")

        except Exception as e:
            error_msg = f"Failed to store content in database: {str(e)}"
            logger.error(error_msg)
            # Don't fail the entire operation if storage fails
            if not result.errors:
                result.errors = []
            result.errors.append(error_msg)

    async def crawl_multiple(
        self, requests: List[CrawlRequest], max_concurrent: Optional[int] = None
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently.

        @description Processes multiple crawl requests concurrently with configurable
        concurrency limits and proper error handling for each request
        @param requests: List of CrawlRequest objects to process
        @param max_concurrent: Maximum number of concurrent requests (defaults to settings)
        @returns: List of CrawlResult objects in the same order as requests
        @throws Exception: If the overall crawling process fails

        @example
        # Crawl multiple blog posts
        requests = [
            CrawlRequest(url="https://blog.com/post1", content_type="blog"),
            CrawlRequest(url="https://blog.com/post2", content_type="blog"),
        ]
        results = await crawler.crawl_multiple(requests, max_concurrent=3)
        """
        if not max_concurrent:
            max_concurrent = self.settings.crawler.concurrent_requests

        logger.info(
            f"Starting batch crawl of {len(requests)} URLs with {max_concurrent} max concurrent"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(request: CrawlRequest) -> CrawlResult:
            async with semaphore:
                # Add delay between requests if configured
                if self.settings.crawler.delay > 0:
                    await asyncio.sleep(self.settings.crawler.delay)
                return await self.crawl_single(request)

        # Execute all crawls concurrently
        tasks = [crawl_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Exception in batch crawl for {requests[i].url}: {str(result)}"
                )
                processed_results.append(
                    CrawlResult(
                        url=requests[i].url,
                        content_type=requests[i].content_type,
                        success=False,
                        errors=[f"Batch crawl exception: {str(result)}"],
                    )
                )
            else:
                processed_results.append(result)

        logger.info(
            f"Completed batch crawl: {sum(1 for r in processed_results if r.success)}/{len(requests)} successful"
        )
        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get crawler statistics.

        @description Returns comprehensive statistics about crawler performance
        and usage patterns for monitoring and optimization
        @returns: Dictionary containing various crawler statistics

        @example
        # Get crawler performance stats
        stats = crawler.get_stats()
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
        """
        success_rate = (
            self.stats["successful_requests"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0
        )

        avg_processing_time = (
            self.stats["total_processing_time"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "settings_summary": {
                "timeout": self.settings.crawler.timeout,
                "max_retries": self.settings.crawler.max_retries,
                "delay": self.settings.crawler.delay,
                "concurrent_requests": self.settings.crawler.concurrent_requests,
                "respect_robots_txt": self.settings.crawler.respect_robots_txt,
            },
        }

    async def close(self) -> None:
        """
        Clean up crawler resources.

        @description Properly closes all crawler resources and connections
        to ensure clean shutdown of the crawler system
        """
        logger.info("Closing crawler resources")
        # Additional cleanup if needed

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
