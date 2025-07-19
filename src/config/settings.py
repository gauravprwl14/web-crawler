"""
Configuration management for the flexible web crawler system.

This module provides centralized configuration management with environment variable support,
validation, and different configuration profiles for development, testing, and production.
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class EnvironmentType(str, Enum):
    """
    Environment types for different deployment scenarios.

    @description Defines the available environment types for configuration profiles
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseConfig(BaseSettings):
    """
    Database configuration settings.

    @description Manages database connection parameters with validation and defaults
    @param url: Database connection URL
    @param host: Database host address
    @param port: Database port number
    @param username: Database username
    @param password: Database password
    @param database: Database name
    @param echo: Enable SQLAlchemy query logging
    @param pool_size: Connection pool size
    @param max_overflow: Maximum pool overflow connections
    """

    # Individual connection parameters
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=3306, env="DATABASE_PORT")
    username: str = Field(default="root", env="DATABASE_USERNAME")
    password: str = Field(default="", env="DATABASE_PASSWORD")
    database: str = Field(default="propmts_directory", env="DATABASE_NAME")

    # Complete URL (if provided, overrides individual parameters)
    url: Optional[str] = Field(default=None, env="DATABASE_URL")

    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    def get_database_url(self) -> str:
        """
        Get the complete database URL.

        @description Constructs the database URL from individual parameters
        or returns the provided URL if available
        @returns: Complete async MySQL database connection URL using aiomysql driver

        @example
        # Returns: mysql+aiomysql://root:password@localhost:3306/prompts-directory
        """
        if self.url:
            return self.url

        # Construct MySQL URL from individual parameters
        password_part = f":{self.password}" if self.password else ""
        return (
            f"mysql+aiomysql://{self.username}@{self.host}:{self.port}/{self.database}"
        )

    @validator("database")
    def validate_database_name(cls, v):
        """
        Validate database name format.

        @description Ensures database name follows MySQL naming conventions
        @param v: The database name to validate
        @returns: Validated database name
        @throws ValueError: If database name contains invalid characters
        """
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Database name can only contain letters, numbers, underscores, and hyphens"
            )
        return v

    class Config:
        env_prefix = "DB_"


class CrawlerConfig(BaseSettings):
    """
    Web crawler configuration settings.

    @description Manages crawler behavior, timeouts, retries, and rate limiting
    @param timeout: Request timeout in seconds
    @param max_retries: Maximum retry attempts for failed requests
    @param delay: Delay between requests in seconds
    @param concurrent_requests: Maximum concurrent requests
    @param user_agent: User agent string for requests
    @param respect_robots_txt: Whether to respect robots.txt
    """

    timeout: int = Field(default=30, env="CRAWLER_TIMEOUT")
    max_retries: int = Field(default=3, env="CRAWLER_MAX_RETRIES")
    delay: float = Field(default=1.0, env="CRAWLER_DELAY")
    concurrent_requests: int = Field(default=5, env="CRAWLER_CONCURRENT_REQUESTS")
    user_agent: str = Field(
        default="FlexibleCrawler/1.0 (+https://example.com/bot)",
        env="CRAWLER_USER_AGENT",
    )
    respect_robots_txt: bool = Field(default=True, env="CRAWLER_RESPECT_ROBOTS")

    @validator("concurrent_requests")
    def validate_concurrent_requests(cls, v):
        """
        Validate concurrent requests limit.

        @description Ensures concurrent requests don't exceed reasonable limits
        @param v: The value to validate
        @returns: Validated value
        @throws ValueError: If value is out of acceptable range
        """
        if v < 1 or v > 50:
            raise ValueError("Concurrent requests must be between 1 and 50")
        return v

    class Config:
        env_prefix = "CRAWLER_"


class APIConfig(BaseSettings):
    """
    API server configuration settings.

    @description Manages API server parameters, security, and performance settings
    @param host: API server host address
    @param port: API server port
    @param debug: Enable debug mode
    @param secret_key: Secret key for JWT and session management
    @param cors_origins: Allowed CORS origins
    @param rate_limit: Rate limit per minute for API requests
    """

    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    secret_key: str = Field(
        default="your-secret-key-change-in-production", env="API_SECRET_KEY"
    )
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    rate_limit: int = Field(default=100, env="API_RATE_LIMIT")

    class Config:
        env_prefix = "API_"


class RedisConfig(BaseSettings):
    """
    Redis configuration for task queue and caching.

    @description Manages Redis connection and behavior settings
    @param url: Redis connection URL
    @param max_connections: Maximum connection pool size
    @param decode_responses: Automatically decode Redis responses
    """

    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")

    class Config:
        env_prefix = "REDIS_"


class Settings(BaseSettings):
    """
    Main application settings container.

    @description Central configuration class that aggregates all configuration sections
    and provides environment-specific settings management
    @param environment: Current environment type
    @param log_level: Logging level
    @param log_format: Log message format
    """

    environment: EnvironmentType = Field(
        default=EnvironmentType.DEVELOPMENT, env="ENVIRONMENT"
    )
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        env="LOG_FORMAT",
    )

    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    def __init__(self, **kwargs):
        """
        Initialize settings with environment-specific defaults.

        @description Creates settings instance and applies environment-specific configurations
        @param kwargs: Optional configuration overrides
        """
        super().__init__(**kwargs)
        self._apply_environment_settings()

    def _apply_environment_settings(self) -> None:
        """
        Apply environment-specific configuration adjustments.

        @description Modifies settings based on the current environment type
        for optimal performance and security in different deployment scenarios
        """
        if self.environment == EnvironmentType.PRODUCTION:
            self.api.debug = False
            self.database.echo = False
            self.log_level = "WARNING"
            # Production should use environment variables for database credentials
        elif self.environment == EnvironmentType.TESTING:
            # Use in-memory SQLite for testing to avoid external dependencies
            self.database.url = "sqlite+aiosqlite:///:memory:"
            self.crawler.delay = 0.1
            self.log_level = "DEBUG"
        elif self.environment == EnvironmentType.DEVELOPMENT:
            self.api.debug = True
            self.database.echo = True
            self.log_level = "DEBUG"
            # Development uses default MySQL settings

    def get_content_type_config(self, content_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific content types.

        @description Provides content-type specific configuration settings
        for different scraping scenarios and data processing requirements
        @param content_type: The type of content being processed
        @returns: Dictionary containing content-type specific settings
        @throws ValueError: If content type is not supported

        @example
        # Get blog configuration
        blog_config = settings.get_content_type_config("blog")
        # Returns: {"table_name": "blogs", "required_fields": ["title", "content"], ...}
        """
        content_configs = {
            "blog": {
                "table_name": "blogs",
                "required_fields": ["title", "content", "url", "published_date"],
                "optional_fields": ["author", "tags", "excerpt", "featured_image"],
                "selectors": {
                    "title": "h1, .title, .post-title",
                    "content": ".content, .post-content, article",
                    "author": ".author, .by-author, .post-author",
                    "published_date": ".date, .published, .post-date",
                },
            },
            "prompt": {
                "table_name": "prompts",
                "required_fields": ["prompt_text", "url", "category"],
                "optional_fields": ["difficulty", "tags", "examples"],
                "selectors": {
                    "prompt_text": ".prompt, .question, .challenge",
                    "category": ".category, .tag, .type",
                    "difficulty": ".difficulty, .level",
                },
            },
            "article": {
                "table_name": "articles",
                "required_fields": ["title", "content", "url"],
                "optional_fields": ["summary", "keywords", "reading_time"],
                "selectors": {
                    "title": "h1, .article-title, .headline",
                    "content": "article, .article-body, .content",
                    "summary": ".summary, .excerpt, .abstract",
                },
            },
            "product": {
                "table_name": "products",
                "required_fields": ["name", "price", "url"],
                "optional_fields": ["description", "rating", "availability", "images"],
                "selectors": {
                    "name": ".product-name, h1, .title",
                    "price": ".price, .cost, .amount",
                    "description": ".description, .product-desc",
                    "rating": ".rating, .stars, .score",
                },
            },
        }

        if content_type not in content_configs:
            raise ValueError(f"Unsupported content type: {content_type}")

        return content_configs[content_type]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.

    @description Provides access to the application's configuration settings
    @returns: The global settings instance

    @example
    # Access settings in your application
    config = get_settings()
    db_url = config.database.url
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment and configuration files.

    @description Forces a reload of all configuration settings, useful for
    testing or when configuration changes need to be applied at runtime
    @returns: The reloaded settings instance
    """
    global settings
    settings = Settings()
    return settings
