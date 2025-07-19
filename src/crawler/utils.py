"""
Utility classes for the web crawler system.

This module provides utility classes for URL validation, robots.txt checking,
and other common crawler operations with proper error handling and caching.
"""

import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Set
from urllib.parse import quote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from loguru import logger


class URLValidator:
    """
    URL validation utility with comprehensive validation rules.

    @description Provides URL validation, normalization, and filtering
    capabilities for web crawling with security and performance considerations
    """

    def __init__(self):
        """Initialize URL validator with default settings."""
        # Blocked domains for security
        self.blocked_domains = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "example.com",
            "example.org",
            "example.net",
        }

        # Allowed schemes
        self.allowed_schemes = {"http", "https"}

        # File extensions to skip
        self.skip_extensions = {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".rar",
            ".tar",
            ".gz",
            ".exe",
            ".dmg",
            ".pkg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".svg",
            ".ico",
            ".css",
            ".js",
            ".xml",
            ".json",
            ".txt",
            ".log",
        }

    def is_valid_url(self, url: str) -> bool:
        """
        Validate if URL is acceptable for crawling.

        @description Performs comprehensive URL validation including scheme,
        domain, path, and security checks to ensure safe crawling
        @param url: URL to validate
        @returns: True if URL is valid and safe to crawl, False otherwise

        @example
        # Validate URLs
        validator = URLValidator()
        print(validator.is_valid_url("https://example.com/page"))  # True
        print(validator.is_valid_url("file:///etc/passwd"))        # False
        print(validator.is_valid_url("https://localhost/admin"))   # False
        """
        if not url or not isinstance(url, str):
            return False

        try:
            parsed = urlparse(url.strip())

            # Check scheme
            if parsed.scheme.lower() not in self.allowed_schemes:
                logger.debug(f"Invalid scheme in URL: {url}")
                return False

            # Check if domain is blocked
            hostname = parsed.hostname
            if not hostname:
                logger.debug(f"No hostname in URL: {url}")
                return False

            if hostname.lower() in self.blocked_domains:
                logger.debug(f"Blocked domain in URL: {url}")
                return False

            # Check for private IP ranges
            if self._is_private_ip(hostname):
                logger.debug(f"Private IP in URL: {url}")
                return False

            # Check file extension
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in self.skip_extensions):
                logger.debug(f"Skipped file extension in URL: {url}")
                return False

            # Check for suspicious patterns
            if self._has_suspicious_patterns(url):
                logger.debug(f"Suspicious pattern in URL: {url}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating URL {url}: {str(e)}")
            return False

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for consistent processing.

        @description Normalizes URL by removing fragments, sorting query parameters,
        and ensuring consistent formatting for deduplication
        @param url: URL to normalize
        @returns: Normalized URL string

        @example
        # Normalize URLs
        normalized = validator.normalize_url("https://example.com/page?b=2&a=1#section")
        # Returns: "https://example.com/page?a=1&b=2"
        """
        try:
            parsed = urlparse(url.strip())

            # Remove fragment
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Sort query parameters
            if parsed.query:
                params = sorted(parsed.query.split("&"))
                normalized += f"?{'&'.join(params)}"

            # Remove trailing slash for root paths
            if normalized.endswith("/") and len(parsed.path) <= 1:
                normalized = normalized.rstrip("/")

            return normalized

        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {str(e)}")
            return url

    def extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.

        @description Safely extracts the domain name from a URL
        @param url: URL to extract domain from
        @returns: Domain name or None if extraction fails
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs belong to the same domain.

        @description Compares domains of two URLs for same-domain validation
        @param url1: First URL to compare
        @param url2: Second URL to compare
        @returns: True if URLs are from the same domain
        """
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        return domain1 is not None and domain1 == domain2

    def _is_private_ip(self, hostname: str) -> bool:
        """
        Check if hostname is a private IP address.

        @description Identifies private IP ranges to prevent crawling internal networks
        @param hostname: Hostname to check
        @returns: True if hostname is a private IP address
        """
        import ipaddress

        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            # Not an IP address, check for localhost patterns
            localhost_patterns = ["localhost", "local", "127.0.0.1", "::1"]
            return any(pattern in hostname.lower() for pattern in localhost_patterns)

    def _has_suspicious_patterns(self, url: str) -> bool:
        """
        Check for suspicious URL patterns.

        @description Identifies potentially dangerous URL patterns
        @param url: URL to check
        @returns: True if URL contains suspicious patterns
        """
        suspicious_patterns = [
            r"\.\./",
            r"%2e%2e%2f",
            r"%252e%252e%252f",  # Directory traversal
            r"file://",
            r"ftp://",
            r"sftp://",  # Non-HTTP schemes
            r"admin",
            r"login",
            r"password",
            r"secret",  # Sensitive paths
            r"wp-admin",
            r"wp-login",
            r".env",
            r"config",  # Common sensitive files
        ]

        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in suspicious_patterns)

    def add_blocked_domain(self, domain: str) -> None:
        """
        Add domain to blocked list.

        @description Adds a domain to the blocked domains list
        @param domain: Domain to block
        """
        self.blocked_domains.add(domain.lower())

    def remove_blocked_domain(self, domain: str) -> None:
        """
        Remove domain from blocked list.

        @description Removes a domain from the blocked domains list
        @param domain: Domain to unblock
        """
        self.blocked_domains.discard(domain.lower())


class RobotsTxtChecker:
    """
    Robots.txt checker with caching for efficient crawling compliance.

    @description Checks robots.txt files to ensure crawling compliance
    with proper caching and error handling for performance optimization
    """

    def __init__(self, cache_duration: int = 3600):
        """
        Initialize robots.txt checker with caching.

        @description Sets up robots.txt checker with configurable cache duration
        @param cache_duration: Cache duration in seconds (default: 1 hour)
        """
        self.cache_duration = cache_duration
        self._cache: Dict[str, Dict] = {}
        self._last_cleanup = time.time()
        self.cleanup_interval = 300  # Clean cache every 5 minutes

    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """
        Check if URL can be fetched according to robots.txt.

        @description Verifies if a URL can be crawled by checking robots.txt rules
        with caching for performance and proper error handling
        @param url: URL to check
        @param user_agent: User agent string to check against
        @returns: True if URL can be fetched, False if disallowed

        @example
        # Check if URL is allowed
        checker = RobotsTxtChecker()
        allowed = await checker.can_fetch("https://example.com/page", "MyBot")
        if allowed:
            print("Crawling is allowed")
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Check cache first
            robots_data = await self._get_robots_txt(base_url)

            if robots_data is None:
                # If robots.txt not accessible, assume allowed
                logger.debug(
                    f"Robots.txt not accessible for {base_url}, assuming allowed"
                )
                return True

            # Parse robots.txt and check
            rp = RobotFileParser()
            rp.set_url(robots_data["url"])
            rp.feed(robots_data["content"])

            can_fetch = rp.can_fetch(user_agent, url)
            logger.debug(
                f"Robots.txt check for {url}: {'allowed' if can_fetch else 'disallowed'}"
            )

            return can_fetch

        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
            # On error, assume allowed to not block legitimate crawling
            return True

    async def get_crawl_delay(self, url: str, user_agent: str = "*") -> Optional[float]:
        """
        Get crawl delay from robots.txt.

        @description Retrieves the crawl delay specified in robots.txt for rate limiting
        @param url: URL to check
        @param user_agent: User agent string to check against
        @returns: Crawl delay in seconds or None if not specified
        """
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            robots_data = await self._get_robots_txt(base_url)
            if robots_data is None:
                return None

            rp = RobotFileParser()
            rp.set_url(robots_data["url"])
            rp.feed(robots_data["content"])

            return rp.crawl_delay(user_agent)

        except Exception as e:
            logger.warning(f"Error getting crawl delay for {url}: {str(e)}")
            return None

    async def _get_robots_txt(self, base_url: str) -> Optional[Dict]:
        """
        Get robots.txt content with caching.

        @description Fetches and caches robots.txt content for efficient access
        @param base_url: Base URL to fetch robots.txt from
        @returns: Dictionary with robots.txt URL and content, or None if not available
        """
        # Clean cache periodically
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._cleanup_cache()
            self._last_cleanup = current_time

        # Check cache
        if base_url in self._cache:
            cache_entry = self._cache[base_url]
            if current_time - cache_entry["timestamp"] < self.cache_duration:
                return cache_entry["data"]

        # Fetch robots.txt
        robots_url = urljoin(base_url, "/robots.txt")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(robots_url)

                if response.status_code == 200:
                    robots_data = {"url": robots_url, "content": response.text}

                    # Cache the result
                    self._cache[base_url] = {
                        "data": robots_data,
                        "timestamp": current_time,
                    }

                    logger.debug(f"Fetched robots.txt from {robots_url}")
                    return robots_data
                else:
                    logger.debug(
                        f"Robots.txt not found at {robots_url} (status: {response.status_code})"
                    )

                    # Cache negative result to avoid repeated requests
                    self._cache[base_url] = {"data": None, "timestamp": current_time}

                    return None

        except Exception as e:
            logger.debug(f"Error fetching robots.txt from {robots_url}: {str(e)}")

            # Cache negative result
            self._cache[base_url] = {"data": None, "timestamp": current_time}

            return None

    def _cleanup_cache(self) -> None:
        """
        Clean expired entries from cache.

        @description Removes expired cache entries to prevent memory bloat
        """
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if current_time - entry["timestamp"] > self.cache_duration:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(
                f"Cleaned {len(expired_keys)} expired robots.txt cache entries"
            )

    def clear_cache(self) -> None:
        """
        Clear all cached robots.txt data.

        @description Clears the entire robots.txt cache
        """
        self._cache.clear()
        logger.debug("Cleared robots.txt cache")


class RateLimiter:
    """
    Rate limiter for controlling request frequency per domain.

    @description Implements per-domain rate limiting to respect server resources
    and avoid being blocked by anti-bot measures
    """

    def __init__(self, default_delay: float = 1.0):
        """
        Initialize rate limiter with default delay.

        @description Sets up rate limiter with configurable default delay
        @param default_delay: Default delay between requests in seconds
        """
        self.default_delay = default_delay
        self._last_request: Dict[str, float] = {}
        self._domain_delays: Dict[str, float] = {}

    async def wait_if_needed(self, url: str) -> None:
        """
        Wait if needed to respect rate limits.

        @description Implements rate limiting by waiting before allowing requests
        based on per-domain delays and last request timestamps
        @param url: URL being requested (used to extract domain)

        @example
        # Use rate limiter before making request
        rate_limiter = RateLimiter(delay=2.0)
        await rate_limiter.wait_if_needed("https://example.com/page")
        # Now safe to make request
        """
        domain = self._extract_domain(url)
        if not domain:
            return

        current_time = time.time()

        # Get delay for this domain
        delay = self._domain_delays.get(domain, self.default_delay)

        # Check if we need to wait
        if domain in self._last_request:
            time_since_last = current_time - self._last_request[domain]
            if time_since_last < delay:
                wait_time = delay - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {domain}")
                await asyncio.sleep(wait_time)

        # Update last request time
        self._last_request[domain] = time.time()

    def set_domain_delay(self, domain: str, delay: float) -> None:
        """
        Set custom delay for specific domain.

        @description Configures domain-specific rate limiting delays
        @param domain: Domain to configure
        @param delay: Delay in seconds between requests to this domain
        """
        self._domain_delays[domain.lower()] = delay
        logger.debug(f"Set rate limit for {domain}: {delay}s")

    def update_delay_from_robots(self, url: str, crawl_delay: float) -> None:
        """
        Update domain delay based on robots.txt crawl-delay.

        @description Updates rate limiting based on robots.txt specifications
        @param url: URL from the domain
        @param crawl_delay: Crawl delay from robots.txt
        """
        domain = self._extract_domain(url)
        if domain and crawl_delay > 0:
            self.set_domain_delay(domain, crawl_delay)

    def _extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.

        @description Helper method to extract domain for rate limiting
        @param url: URL to extract domain from
        @returns: Domain name or None if extraction fails
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Dict]:
        """
        Get rate limiter statistics.

        @description Returns statistics about rate limiting activity
        @returns: Dictionary with rate limiting statistics per domain
        """
        current_time = time.time()
        stats = {}

        for domain, last_request_time in self._last_request.items():
            delay = self._domain_delays.get(domain, self.default_delay)
            time_since_last = current_time - last_request_time

            stats[domain] = {
                "delay": delay,
                "last_request": datetime.fromtimestamp(last_request_time).isoformat(),
                "time_since_last": time_since_last,
                "ready_to_request": time_since_last >= delay,
            }

        return stats


class URLFilter:
    """
    URL filtering utility for advanced crawling control.

    @description Provides pattern-based URL filtering with support for
    whitelisting, blacklisting, and complex matching rules
    """

    def __init__(self):
        """Initialize URL filter with empty rules."""
        self.whitelist_patterns: Set[str] = set()
        self.blacklist_patterns: Set[str] = set()
        self.regex_whitelist: Set[re.Pattern] = set()
        self.regex_blacklist: Set[re.Pattern] = set()

    def add_whitelist_pattern(self, pattern: str, is_regex: bool = False) -> None:
        """
        Add pattern to whitelist.

        @description Adds URL pattern to whitelist for allowed URLs
        @param pattern: Pattern to match URLs against
        @param is_regex: Whether pattern is a regular expression
        """
        if is_regex:
            self.regex_whitelist.add(re.compile(pattern))
        else:
            self.whitelist_patterns.add(pattern)

    def add_blacklist_pattern(self, pattern: str, is_regex: bool = False) -> None:
        """
        Add pattern to blacklist.

        @description Adds URL pattern to blacklist for blocked URLs
        @param pattern: Pattern to match URLs against
        @param is_regex: Whether pattern is a regular expression
        """
        if is_regex:
            self.regex_blacklist.add(re.compile(pattern))
        else:
            self.blacklist_patterns.add(pattern)

    def is_allowed(self, url: str) -> bool:
        """
        Check if URL is allowed by filter rules.

        @description Applies whitelist and blacklist rules to determine if URL is allowed
        @param url: URL to check
        @returns: True if URL passes filter rules

        @example
        # Set up URL filter
        url_filter = URLFilter()
        url_filter.add_whitelist_pattern("https://example.com/*")
        url_filter.add_blacklist_pattern("*/admin/*")

        # Check URLs
        print(url_filter.is_allowed("https://example.com/page"))     # True
        print(url_filter.is_allowed("https://example.com/admin/"))   # False
        """
        # If blacklisted, reject
        if self._matches_blacklist(url):
            return False

        # If whitelist is empty, allow (unless blacklisted)
        if not self.whitelist_patterns and not self.regex_whitelist:
            return True

        # If whitelist exists, must match whitelist
        return self._matches_whitelist(url)

    def _matches_whitelist(self, url: str) -> bool:
        """Check if URL matches whitelist patterns."""
        # Check simple patterns
        for pattern in self.whitelist_patterns:
            if self._simple_match(url, pattern):
                return True

        # Check regex patterns
        for regex in self.regex_whitelist:
            if regex.search(url):
                return True

        return False

    def _matches_blacklist(self, url: str) -> bool:
        """Check if URL matches blacklist patterns."""
        # Check simple patterns
        for pattern in self.blacklist_patterns:
            if self._simple_match(url, pattern):
                return True

        # Check regex patterns
        for regex in self.regex_blacklist:
            if regex.search(url):
                return True

        return False

    def _simple_match(self, url: str, pattern: str) -> bool:
        """
        Simple pattern matching with wildcards.

        @description Implements simple wildcard pattern matching
        @param url: URL to match
        @param pattern: Pattern with optional wildcards (*)
        @returns: True if URL matches pattern
        """
        # Convert simple wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        regex_pattern = f"^{regex_pattern}$"

        try:
            return bool(re.match(regex_pattern, url))
        except re.error:
            # If regex is invalid, fall back to simple string matching
            return pattern in url

    def clear_rules(self) -> None:
        """Clear all filter rules."""
        self.whitelist_patterns.clear()
        self.blacklist_patterns.clear()
        self.regex_whitelist.clear()
        self.regex_blacklist.clear()

    def get_rules_summary(self) -> Dict[str, int]:
        """
        Get summary of filter rules.

        @description Returns count of different types of filter rules
        @returns: Dictionary with rule counts
        """
        return {
            "whitelist_patterns": len(self.whitelist_patterns),
            "blacklist_patterns": len(self.blacklist_patterns),
            "regex_whitelist": len(self.regex_whitelist),
            "regex_blacklist": len(self.regex_blacklist),
        }
