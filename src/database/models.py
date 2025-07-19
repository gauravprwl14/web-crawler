"""
Database models for the flexible web crawler system.

This module provides SQLAlchemy models with dynamic table creation capabilities
for different content types, ensuring flexible data storage and retrieval.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    desc,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.dialects.sqlite import TEXT
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import text

from config.settings import get_settings

Base = declarative_base()
settings = get_settings()


class BaseModel(Base):
    """
    Abstract base model with common fields for all content types.

    @description Provides common functionality and fields that all content models inherit
    @param id: Unique identifier for the record
    @param created_at: Timestamp when the record was created
    @param updated_at: Timestamp when the record was last updated
    @param url: Source URL where the content was scraped from
    @param content_type: Type of content (blog, prompt, article, etc.)
    @param raw_data: Original scraped data in JSON format
    @param meta_data: Additional meta_data about the scraping process
    """

    __abstract__ = True

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    url = Column(String(767), nullable=False, index=True)
    content_type = Column(String(50), nullable=False, index=True)
    raw_data = Column(JSON, nullable=True)
    meta_data = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.

        @description Serializes the model instance to a dictionary format
        suitable for JSON responses and data processing
        @returns: Dictionary representation of the model instance

        @example
        # Convert a blog instance to dictionary
        blog_dict = blog_instance.to_dict()
        # Returns: {"id": "...", "title": "...", "content": "...", ...}
        """
        return {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update model instance from dictionary.

        @description Updates the model instance with values from a dictionary
        @param data: Dictionary containing field names and values to update
        @throws AttributeError: If a field in the dictionary doesn't exist on the model

        @example
        # Update a blog instance
        blog_instance.update_from_dict({"title": "New Title", "content": "Updated content"})
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()


class Blog(BaseModel):
    """
    Model for blog content storage.

    @description Stores blog posts with comprehensive metadata and content fields
    optimized for blog-specific data structures and search capabilities
    """

    __tablename__ = "blogs"

    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    excerpt = Column(Text, nullable=True)
    author = Column(String(200), nullable=True, index=True)
    published_date = Column(DateTime, nullable=True, index=True)
    tags = Column(JSON, nullable=True)  # List of tag strings
    featured_image = Column(String(767), nullable=True)
    word_count = Column(Integer, nullable=True)
    reading_time = Column(Integer, nullable=True)  # in minutes
    category = Column(String(100), nullable=True, index=True)
    slug = Column(String(500), nullable=True, unique=True)

    __table_args__ = (
        Index("idx_blog_published_date_desc", desc("published_date")),
        Index("idx_blog_author_date", author, published_date),
        UniqueConstraint("url", name="uq_blog_url"),
    )


class Prompt(BaseModel):
    """
    Model for prompt content storage.

    @description Stores AI prompts, coding challenges, and questions with
    categorization and difficulty tracking for educational and training purposes
    """

    __tablename__ = "prompts"

    prompt_text = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, index=True)
    difficulty = Column(String(20), nullable=True, index=True)  # easy, medium, hard
    tags = Column(JSON, nullable=True)  # List of tag strings
    examples = Column(JSON, nullable=True)  # List of example inputs/outputs
    solution = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    programming_language = Column(String(50), nullable=True, index=True)
    time_limit = Column(Integer, nullable=True)  # in seconds
    memory_limit = Column(Integer, nullable=True)  # in MB

    __table_args__ = (
        Index("idx_prompt_category_difficulty", category, difficulty),
        Index("idx_prompt_language_category", programming_language, category),
        UniqueConstraint("url", name="uq_prompt_url"),
    )


class Article(BaseModel):
    """
    Model for general article content storage.

    @description Stores general articles, news, and informational content
    with flexible metadata and content organization capabilities
    """

    __tablename__ = "articles"

    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)  # List of keyword strings
    reading_time = Column(Integer, nullable=True)  # in minutes
    word_count = Column(Integer, nullable=True)
    language = Column(String(10), nullable=True, default="en")
    topic = Column(String(100), nullable=True, index=True)
    source_domain = Column(String(200), nullable=True, index=True)

    __table_args__ = (
        Index("idx_article_topic_date", "topic", desc("created_at")),
        Index("idx_article_domain_date", "source_domain", desc("created_at")),
        UniqueConstraint("url", name="uq_article_url"),
    )


class Product(BaseModel):
    """
    Model for product information storage.

    @description Stores e-commerce product data with pricing, availability,
    and comprehensive product details for market analysis and comparison
    """

    __tablename__ = "products"

    name = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=True, index=True)
    currency = Column(String(3), nullable=True, default="USD")
    availability = Column(String(50), nullable=True, index=True)
    rating = Column(Float, nullable=True, index=True)
    review_count = Column(Integer, nullable=True)
    brand = Column(String(200), nullable=True, index=True)
    category = Column(String(100), nullable=True, index=True)
    sku = Column(String(100), nullable=True, unique=True)
    images = Column(JSON, nullable=True)  # List of image URLs
    specifications = Column(JSON, nullable=True)  # Product specifications

    __table_args__ = (
        Index("idx_product_brand_category", brand, category),
        Index("idx_product_price_rating", price, rating),
        UniqueConstraint("url", name="uq_product_url"),
    )


class ScrapingJob(BaseModel):
    """
    Model for tracking scraping jobs and their status.

    @description Manages scraping job lifecycle, status tracking, and results
    for monitoring and debugging crawling operations
    """

    __tablename__ = "scraping_jobs"

    job_id = Column(String(100), nullable=False, unique=True, index=True)
    status = Column(
        String(20), nullable=False, default="pending", index=True
    )  # pending, running, completed, failed
    target_urls = Column(JSON, nullable=False)  # List of URLs to scrape
    content_type = Column(String(50), nullable=False, index=True)
    selectors = Column(JSON, nullable=True)  # Custom selectors for this job
    results_count = Column(Integer, nullable=True, default=0)
    error_count = Column(Integer, nullable=True, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    config = Column(JSON, nullable=True)  # Job-specific configuration

    __table_args__ = (
        Index("idx_job_status_date", "status", desc("created_at")),
        Index("idx_job_type_status", content_type, status),
    )


class ScrapingResult(BaseModel):
    """
    Model for storing individual scraping results.

    @description Links scraped content to scraping jobs and tracks
    processing status and validation results for quality control
    """

    __tablename__ = "scraping_results"

    job_id = Column(
        String(100), ForeignKey("scraping_jobs.job_id"), nullable=False, index=True
    )
    target_url = Column(String(767), nullable=False, index=True)
    scraped_data = Column(JSON, nullable=True)
    processing_status = Column(
        String(20), nullable=False, default="pending", index=True
    )
    validation_errors = Column(JSON, nullable=True)  # List of validation error messages
    content_id = Column(
        String(36), nullable=True
    )  # Reference to content in specific table
    processing_time = Column(Float, nullable=True)  # Processing time in seconds

    # Relationship to scraping job
    job = relationship("ScrapingJob", backref="results")

    __table_args__ = (
        Index("idx_result_job_status", job_id, processing_status),
        Index("idx_result_url_job", target_url, job_id),
    )


# Content type to model mapping
CONTENT_TYPE_MODELS = {
    "blog": Blog,
    "prompt": Prompt,
    "article": Article,
    "product": Product,
}


class DatabaseManager:
    """
    Database manager for handling connections and dynamic operations.

    @description Manages database connections, table creation, and provides
    high-level operations for dynamic content type handling
    """

    def __init__(self):
        """
        Initialize database manager with async engine and session factory.

        @description Sets up the database connection and session management
        for asynchronous operations with proper connection pooling
        """
        self.engine = create_async_engine(
            settings.database.get_database_url(),
            echo=settings.database.echo,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        self.async_session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """
        Create all database tables.

        @description Creates all tables defined in the models, including
        indexes and constraints for optimal performance
        @throws Exception: If table creation fails

        @example
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.create_tables()
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """
        Drop all database tables.

        @description Removes all tables from the database - use with caution
        @throws Exception: If table deletion fails
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def get_model_class(self, content_type: str):
        """
        Get the appropriate model class for a content type.

        @description Returns the SQLAlchemy model class for the specified content type
        @param content_type: The type of content (blog, prompt, article, product)
        @returns: The corresponding SQLAlchemy model class
        @throws ValueError: If content type is not supported

        @example
        # Get blog model class
        BlogModel = db_manager.get_model_class("blog")
        # Create new blog instance
        blog = BlogModel(title="Test", content="Content", url="http://example.com")
        """
        if content_type not in CONTENT_TYPE_MODELS:
            raise ValueError(f"Unsupported content type: {content_type}")
        return CONTENT_TYPE_MODELS[content_type]

    async def insert_content(self, content_type: str, data: Dict[str, Any]) -> str:
        """
        Insert content into the appropriate table based on content type.

        @description Dynamically inserts data into the correct table based on
        content type, handling validation and relationship management
        @param content_type: Type of content to insert
        @param data: Dictionary containing the content data
        @returns: The ID of the inserted record
        @throws ValueError: If content type is unsupported or data is invalid
        @throws Exception: If database insertion fails

        @example
        # Insert blog content
        blog_data = {
            "title": "My Blog Post",
            "content": "Blog content here...",
            "url": "https://example.com/blog/post",
            "author": "John Doe",
            "published_date": datetime.utcnow()
        }
        content_id = await db_manager.insert_content("blog", blog_data)
        """
        model_class = self.get_model_class(content_type)

        # Ensure content_type is set
        data["content_type"] = content_type

        async with self.async_session_factory() as session:
            try:
                instance = model_class(**data)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                return instance.id
            except Exception as e:
                await session.rollback()
                raise e

    async def get_content(
        self, content_type: str, content_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve content by ID and type.

        @description Fetches a specific content record by its ID and content type
        @param content_type: Type of content to retrieve
        @param content_id: Unique identifier of the content
        @returns: Dictionary representation of the content or None if not found
        @throws ValueError: If content type is unsupported

        @example
        # Get specific blog post
        blog = await db_manager.get_content("blog", "blog-id-123")
        if blog:
            print(f"Title: {blog['title']}")
        """
        model_class = self.get_model_class(content_type)

        async with self.async_session_factory() as session:
            result = await session.get(model_class, content_id)
            return result.to_dict() if result else None

    async def search_content(
        self,
        content_type: str,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search content with filters and pagination.

        @description Performs filtered search across content with pagination support
        @param content_type: Type of content to search
        @param filters: Dictionary of field names and values to filter by
        @param limit: Maximum number of results to return
        @param offset: Number of results to skip for pagination
        @returns: List of matching content dictionaries
        @throws ValueError: If content type is unsupported

        @example
        # Search for blogs by author
        blogs = await db_manager.search_content(
            "blog",
            filters={"author": "John Doe"},
            limit=10
        )
        """
        model_class = self.get_model_class(content_type)

        async with self.async_session_factory() as session:
            query = session.query(model_class)

            if filters:
                for field, value in filters.items():
                    if hasattr(model_class, field):
                        query = query.filter(getattr(model_class, field) == value)

            query = query.offset(offset).limit(limit)
            result = await session.execute(query)
            return [item.to_dict() for item in result.scalars().all()]

    async def close(self) -> None:
        """
        Close database connections.

        @description Properly closes all database connections and cleans up resources
        """
        await self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncSession:
    """
    Get async database session for dependency injection.

    @description Provides an async database session for use in API endpoints
    and other components requiring database access
    @returns: Async SQLAlchemy session
    @yields: Database session that automatically closes after use

    @example
    # Use in FastAPI dependency injection
    async def my_endpoint(db: AsyncSession = Depends(get_db_session)):
        # Use db session here
        result = await db.execute(select(Blog))
    """
    async with db_manager.async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_database() -> None:
    """
    Initialize database tables and perform any necessary migrations.

    @description Sets up the database schema and performs initial configuration
    for the crawler system, ensuring all tables and indexes are properly created
    @throws Exception: If database initialization fails
    """
    await db_manager.create_tables()
