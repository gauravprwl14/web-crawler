"""
Content processors for different content types.

This module provides specialized processors for cleaning, validating,
and normalizing content extracted from different types of web pages.
"""

import html
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger


class BaseContentProcessor(ABC):
    """
    Abstract base class for content processors.

    @description Defines the interface for content processors that handle
    different types of web content with specialized cleaning and validation
    """

    @abstractmethod
    async def process(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """
        Process and clean extracted content.

        @description Abstract method for processing extracted content data
        @param data: Raw extracted data dictionary
        @param source_url: Source URL where content was extracted from
        @returns: Processed and cleaned data dictionary
        """
        pass

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        @description Removes unwanted characters, normalizes whitespace,
        and handles HTML entities in text content
        @param text: Raw text to clean
        @returns: Cleaned and normalized text

        @example
        # Clean messy text
        clean = self._clean_text("  Hello\n\nWorld  &amp; stuff  ")
        # Returns: "Hello World & stuff"
        """
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove common unwanted characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]", "", text)

        return text

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.

        @description Parses URL to extract the domain name
        @param url: URL to extract domain from
        @returns: Domain name or empty string if parsing fails
        """
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""

    def _calculate_reading_time(self, text: str, words_per_minute: int = 200) -> int:
        """
        Calculate estimated reading time for text.

        @description Estimates reading time based on word count and average reading speed
        @param text: Text content to analyze
        @param words_per_minute: Average reading speed (default: 200 WPM)
        @returns: Estimated reading time in minutes
        """
        if not text:
            return 0

        word_count = len(text.split())
        reading_time = max(1, round(word_count / words_per_minute))
        return reading_time

    def _calculate_word_count(self, text: str) -> int:
        """
        Calculate word count for text.

        @description Counts words in text content
        @param text: Text to count words in
        @returns: Number of words
        """
        if not text:
            return 0
        return len(text.split())


class BlogContentProcessor(BaseContentProcessor):
    """
    Processor for blog content with blog-specific cleaning and validation.

    @description Handles blog posts with title extraction, content cleaning,
    author processing, and metadata generation optimized for blog content
    """

    async def process(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """
        Process blog content with specialized cleaning and validation.

        @description Cleans and validates blog-specific fields including title,
        content, author information, and generates additional metadata
        @param data: Raw extracted blog data
        @param source_url: Source URL of the blog post
        @returns: Processed blog data with cleaned fields and metadata
        """
        processed = {}

        # Process title
        title = self._clean_text(data.get("title", ""))
        processed["title"] = title

        # Process content
        content = self._clean_text(data.get("content", ""))
        processed["content"] = content

        # Process excerpt
        excerpt = data.get("excerpt")
        if not excerpt and content:
            # Generate excerpt from content if not provided
            excerpt = self._generate_excerpt(content)
        processed["excerpt"] = self._clean_text(excerpt) if excerpt else None

        # Process author
        author = self._clean_text(data.get("author", ""))
        processed["author"] = author if author else None

        # Process published date
        published_date = data.get("published_date")
        if isinstance(published_date, str):
            processed["published_date"] = self._parse_date(published_date)
        elif isinstance(published_date, datetime):
            processed["published_date"] = published_date
        else:
            processed["published_date"] = None

        # Process tags
        tags = data.get("tags")
        if tags:
            if isinstance(tags, str):
                # Split string tags
                processed["tags"] = [
                    tag.strip() for tag in tags.split(",") if tag.strip()
                ]
            elif isinstance(tags, list):
                processed["tags"] = [self._clean_text(tag) for tag in tags if tag]
            else:
                processed["tags"] = None
        else:
            processed["tags"] = None

        # Process featured image
        featured_image = data.get("featured_image")
        processed["featured_image"] = featured_image if featured_image else None

        # Calculate metadata
        if content:
            processed["word_count"] = self._calculate_word_count(content)
            processed["reading_time"] = self._calculate_reading_time(content)
        else:
            processed["word_count"] = 0
            processed["reading_time"] = 0

        # Generate category from tags or domain
        category = data.get("category")
        if not category and processed.get("tags"):
            category = processed["tags"][0]  # Use first tag as category
        elif not category:
            category = self._extract_domain(source_url)
        processed["category"] = category

        # Generate slug from title
        if title:
            processed["slug"] = self._generate_slug(title)
        else:
            processed["slug"] = None

        # Add source domain
        processed["source_domain"] = self._extract_domain(source_url)

        logger.debug(f"Processed blog content: {title[:50]}...")
        return processed

    def _generate_excerpt(self, content: str, max_length: int = 200) -> str:
        """
        Generate excerpt from content.

        @description Creates a brief excerpt from the main content
        @param content: Full content text
        @param max_length: Maximum length of excerpt
        @returns: Generated excerpt
        """
        if not content:
            return ""

        # Take first paragraph or first sentences up to max_length
        sentences = content.split(". ")
        excerpt = ""

        for sentence in sentences:
            if len(excerpt + sentence) <= max_length:
                excerpt += sentence + ". "
            else:
                break

        return excerpt.strip()

    def _generate_slug(self, title: str) -> str:
        """
        Generate URL slug from title.

        @description Creates a URL-friendly slug from the blog title
        @param title: Blog post title
        @returns: URL slug
        """
        if not title:
            return ""

        # Convert to lowercase and replace spaces with hyphens
        slug = title.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)  # Remove special characters
        slug = re.sub(r"[\s_-]+", "-", slug)  # Replace whitespace with single hyphen
        slug = slug.strip("-")  # Remove leading/trailing hyphens

        return slug

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """
        Parse date string for blog posts.

        @description Attempts to parse various date formats commonly used in blogs
        @param date_string: Date string to parse
        @returns: Parsed datetime or None if parsing fails
        """
        if not date_string:
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%B %d, %Y",
            "%d %B %Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue

        return None


class PromptContentProcessor(BaseContentProcessor):
    """
    Processor for prompt/challenge content with difficulty analysis.

    @description Handles AI prompts, coding challenges, and educational content
    with categorization, difficulty assessment, and example processing
    """

    async def process(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """
        Process prompt content with specialized validation and categorization.

        @description Cleans prompt text, analyzes difficulty, processes examples,
        and generates metadata for educational and training content
        @param data: Raw extracted prompt data
        @param source_url: Source URL of the prompt
        @returns: Processed prompt data with analysis and metadata
        """
        processed = {}

        # Process prompt text
        prompt_text = self._clean_text(data.get("prompt_text", ""))
        processed["prompt_text"] = prompt_text

        # Process category
        category = self._clean_text(data.get("category", ""))
        if not category:
            category = self._infer_category(prompt_text, source_url)
        processed["category"] = category

        # Process difficulty
        difficulty = data.get("difficulty")
        if not difficulty:
            difficulty = self._analyze_difficulty(prompt_text)
        processed["difficulty"] = difficulty.lower() if difficulty else None

        # Process tags
        tags = data.get("tags")
        if tags:
            if isinstance(tags, str):
                processed["tags"] = [
                    tag.strip() for tag in tags.split(",") if tag.strip()
                ]
            elif isinstance(tags, list):
                processed["tags"] = [self._clean_text(tag) for tag in tags if tag]
            else:
                processed["tags"] = []
        else:
            processed["tags"] = self._extract_tags_from_text(prompt_text)

        # Process examples
        examples = data.get("examples")
        if examples:
            processed["examples"] = self._process_examples(examples)
        else:
            processed["examples"] = None

        # Process solution and explanation
        processed["solution"] = self._clean_text(data.get("solution", "")) or None
        processed["explanation"] = self._clean_text(data.get("explanation", "")) or None

        # Detect programming language
        programming_language = data.get("programming_language")
        if not programming_language:
            programming_language = self._detect_programming_language(prompt_text)
        processed["programming_language"] = programming_language

        # Process time and memory limits
        processed["time_limit"] = self._parse_int(data.get("time_limit"))
        processed["memory_limit"] = self._parse_int(data.get("memory_limit"))

        logger.debug(f"Processed prompt content: {category} - {difficulty}")
        return processed

    def _infer_category(self, prompt_text: str, source_url: str) -> str:
        """
        Infer category from prompt text and source URL.

        @description Analyzes content to determine the most appropriate category
        @param prompt_text: The prompt text to analyze
        @param source_url: Source URL for additional context
        @returns: Inferred category
        """
        text_lower = prompt_text.lower()
        domain = self._extract_domain(source_url)

        # Programming-related keywords
        if any(
            keyword in text_lower
            for keyword in [
                "algorithm",
                "function",
                "code",
                "programming",
                "python",
                "javascript",
            ]
        ):
            return "programming"

        # Math-related keywords
        if any(
            keyword in text_lower
            for keyword in ["calculate", "equation", "mathematics", "solve", "formula"]
        ):
            return "mathematics"

        # Logic-related keywords
        if any(
            keyword in text_lower
            for keyword in ["puzzle", "logic", "reasoning", "think", "problem"]
        ):
            return "logic"

        # Domain-based inference
        if "leetcode" in domain:
            return "programming"
        elif "math" in domain:
            return "mathematics"

        return "general"

    def _analyze_difficulty(self, prompt_text: str) -> str:
        """
        Analyze difficulty level from prompt text.

        @description Uses text analysis to estimate difficulty level
        @param prompt_text: Prompt text to analyze
        @returns: Estimated difficulty level (easy, medium, hard)
        """
        text_lower = prompt_text.lower()

        # Easy indicators
        easy_keywords = ["simple", "basic", "easy", "beginner", "introduction"]
        if any(keyword in text_lower for keyword in easy_keywords):
            return "easy"

        # Hard indicators
        hard_keywords = [
            "complex",
            "advanced",
            "difficult",
            "challenging",
            "optimize",
            "efficient",
        ]
        if any(keyword in text_lower for keyword in hard_keywords):
            return "hard"

        # Text length and complexity heuristics
        word_count = len(prompt_text.split())
        if word_count < 50:
            return "easy"
        elif word_count > 200:
            return "hard"

        return "medium"

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """
        Extract relevant tags from prompt text.

        @description Analyzes text to identify relevant tags and topics
        @param text: Text to analyze for tags
        @returns: List of extracted tags
        """
        text_lower = text.lower()
        potential_tags = []

        # Programming languages
        languages = [
            "python",
            "javascript",
            "java",
            "c++",
            "c#",
            "ruby",
            "go",
            "rust",
            "php",
        ]
        for lang in languages:
            if lang in text_lower:
                potential_tags.append(lang)

        # Data structures
        data_structures = [
            "array",
            "list",
            "tree",
            "graph",
            "queue",
            "stack",
            "hash",
            "linked list",
        ]
        for ds in data_structures:
            if ds in text_lower:
                potential_tags.append(ds.replace(" ", "-"))

        # Algorithms
        algorithms = [
            "sorting",
            "searching",
            "dynamic programming",
            "recursion",
            "greedy",
        ]
        for algo in algorithms:
            if algo in text_lower:
                potential_tags.append(algo.replace(" ", "-"))

        return potential_tags[:5]  # Limit to 5 tags

    def _process_examples(self, examples: Any) -> List[Dict[str, Any]]:
        """
        Process examples data into structured format.

        @description Normalizes examples into a consistent structure
        @param examples: Raw examples data
        @returns: Processed examples list
        """
        if not examples:
            return []

        if isinstance(examples, str):
            # Try to parse simple input/output format
            return [{"input": examples, "output": "", "explanation": ""}]
        elif isinstance(examples, list):
            processed_examples = []
            for example in examples:
                if isinstance(example, dict):
                    processed_examples.append(
                        {
                            "input": self._clean_text(str(example.get("input", ""))),
                            "output": self._clean_text(str(example.get("output", ""))),
                            "explanation": self._clean_text(
                                str(example.get("explanation", ""))
                            ),
                        }
                    )
                else:
                    processed_examples.append(
                        {
                            "input": self._clean_text(str(example)),
                            "output": "",
                            "explanation": "",
                        }
                    )
            return processed_examples

        return []

    def _detect_programming_language(self, text: str) -> Optional[str]:
        """
        Detect programming language from text content.

        @description Analyzes text for language-specific keywords and syntax
        @param text: Text to analyze
        @returns: Detected programming language or None
        """
        text_lower = text.lower()

        # Language-specific keywords
        language_patterns = {
            "python": ["def ", "import ", "python", "print(", "if __name__"],
            "javascript": ["function", "var ", "let ", "const ", "console.log"],
            "java": ["public class", "public static", "System.out"],
            "c++": ["#include", "int main", "std::", "cout"],
            "c#": ["using System", "Console.WriteLine", "public class"],
            "go": ["func ", "package ", "fmt."],
            "rust": ["fn ", "let mut", "println!"],
        }

        for language, patterns in language_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                return language

        return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """
        Parse integer value safely.

        @description Safely converts various types to integer
        @param value: Value to convert
        @returns: Integer value or None if conversion fails
        """
        if value is None:
            return None

        try:
            return int(value)
        except (ValueError, TypeError):
            return None


class ArticleContentProcessor(BaseContentProcessor):
    """
    Processor for general article content with SEO and readability analysis.

    @description Handles news articles, informational content, and general web articles
    with keyword extraction, readability analysis, and content optimization
    """

    async def process(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """
        Process article content with SEO and readability analysis.

        @description Cleans article content and generates metadata for SEO optimization
        @param data: Raw extracted article data
        @param source_url: Source URL of the article
        @returns: Processed article data with SEO metadata
        """
        processed = {}

        # Process title
        title = self._clean_text(data.get("title", ""))
        processed["title"] = title

        # Process content
        content = self._clean_text(data.get("content", ""))
        processed["content"] = content

        # Process summary
        summary = data.get("summary")
        if not summary and content:
            summary = self._generate_summary(content)
        processed["summary"] = self._clean_text(summary) if summary else None

        # Extract keywords
        keywords = data.get("keywords")
        if not keywords and content:
            keywords = self._extract_keywords(content, title)
        processed["keywords"] = keywords

        # Calculate reading time and word count
        if content:
            processed["word_count"] = self._calculate_word_count(content)
            processed["reading_time"] = self._calculate_reading_time(content)
        else:
            processed["word_count"] = 0
            processed["reading_time"] = 0

        # Detect language
        language = data.get("language")
        if not language:
            language = self._detect_language(content)
        processed["language"] = language or "en"

        # Infer topic
        topic = data.get("topic")
        if not topic:
            topic = self._infer_topic(content, title)
        processed["topic"] = topic

        # Add source domain
        processed["source_domain"] = self._extract_domain(source_url)

        logger.debug(f"Processed article content: {title[:50]}...")
        return processed

    def _generate_summary(self, content: str, max_length: int = 300) -> str:
        """
        Generate summary from article content.

        @description Creates a summary by extracting key sentences
        @param content: Full article content
        @param max_length: Maximum summary length
        @returns: Generated summary
        """
        if not content:
            return ""

        # Split into sentences
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # Take first few sentences up to max_length
        summary = ""
        for sentence in sentences[:3]:  # Max 3 sentences
            if len(summary + sentence) <= max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip()

    def _extract_keywords(self, content: str, title: str = "") -> List[str]:
        """
        Extract keywords from content and title.

        @description Identifies important keywords and phrases from the text
        @param content: Article content
        @param title: Article title
        @returns: List of extracted keywords
        """
        text = (title + " " + content).lower()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        word_freq = {}

        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]

    def _detect_language(self, content: str) -> Optional[str]:
        """
        Detect content language (basic implementation).

        @description Simple language detection based on common words
        @param content: Text content to analyze
        @returns: Detected language code or None
        """
        if not content:
            return None

        text_lower = content.lower()

        # Simple language detection based on common words
        english_words = [
            "the",
            "and",
            "is",
            "in",
            "to",
            "of",
            "a",
            "that",
            "it",
            "with",
        ]
        spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"]
        french_words = ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"]

        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        french_count = sum(1 for word in french_words if word in text_lower)

        if english_count > spanish_count and english_count > french_count:
            return "en"
        elif spanish_count > french_count:
            return "es"
        elif french_count > 0:
            return "fr"

        return "en"  # Default to English

    def _infer_topic(self, content: str, title: str = "") -> Optional[str]:
        """
        Infer article topic from content and title.

        @description Analyzes content to determine the main topic
        @param content: Article content
        @param title: Article title
        @returns: Inferred topic
        """
        text = (title + " " + content).lower()

        topic_keywords = {
            "technology": [
                "software",
                "computer",
                "programming",
                "tech",
                "ai",
                "machine learning",
            ],
            "business": [
                "company",
                "market",
                "business",
                "finance",
                "economy",
                "investment",
            ],
            "health": [
                "health",
                "medical",
                "doctor",
                "patient",
                "treatment",
                "medicine",
            ],
            "sports": ["sport", "game", "team", "player", "match", "championship"],
            "science": [
                "research",
                "study",
                "scientist",
                "experiment",
                "discovery",
                "science",
            ],
            "politics": [
                "government",
                "election",
                "political",
                "policy",
                "president",
                "congress",
            ],
            "entertainment": [
                "movie",
                "music",
                "celebrity",
                "entertainment",
                "film",
                "actor",
            ],
        }

        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            return max(topic_scores, key=topic_scores.get)

        return "general"


class ProductContentProcessor(BaseContentProcessor):
    """
    Processor for e-commerce product content with pricing and specification handling.

    @description Handles product listings with price parsing, specification processing,
    and availability tracking for e-commerce applications
    """

    async def process(self, data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
        """
        Process product content with e-commerce specific validation.

        @description Cleans product data and normalizes pricing and availability information
        @param data: Raw extracted product data
        @param source_url: Source URL of the product
        @returns: Processed product data with normalized e-commerce fields
        """
        processed = {}

        # Process name
        name = self._clean_text(data.get("name", ""))
        processed["name"] = name

        # Process description
        description = self._clean_text(data.get("description", ""))
        processed["description"] = description if description else None

        # Process price
        price = self._parse_price(data.get("price"))
        processed["price"] = price

        # Process currency
        currency = data.get("currency")
        if not currency and data.get("price"):
            currency = self._extract_currency(str(data.get("price")))
        processed["currency"] = currency or "USD"

        # Process availability
        availability = data.get("availability")
        if availability:
            processed["availability"] = self._normalize_availability(availability)
        else:
            processed["availability"] = None

        # Process rating
        rating = self._parse_float(data.get("rating"))
        if rating is not None:
            processed["rating"] = max(0, min(5, rating))  # Normalize to 0-5 scale
        else:
            processed["rating"] = None

        # Process review count
        processed["review_count"] = self._parse_int(data.get("review_count"))

        # Process brand
        brand = self._clean_text(data.get("brand", ""))
        processed["brand"] = brand if brand else None

        # Process category
        category = self._clean_text(data.get("category", ""))
        if not category:
            category = self._infer_category_from_url(source_url)
        processed["category"] = category if category else None

        # Process SKU
        sku = data.get("sku")
        processed["sku"] = sku if sku else None

        # Process images
        images = data.get("images")
        if images:
            processed["images"] = self._process_images(images, source_url)
        else:
            processed["images"] = None

        # Process specifications
        specifications = data.get("specifications")
        if specifications:
            processed["specifications"] = self._process_specifications(specifications)
        else:
            processed["specifications"] = None

        logger.debug(f"Processed product content: {name[:50]}...")
        return processed

    def _parse_price(self, price_str: Any) -> Optional[float]:
        """
        Parse price from various string formats.

        @description Extracts numeric price value from price strings
        @param price_str: Price string to parse
        @returns: Parsed price as float or None if parsing fails
        """
        if price_str is None:
            return None

        if isinstance(price_str, (int, float)):
            return float(price_str)

        price_str = str(price_str)

        # Remove currency symbols and common formatting
        price_clean = re.sub(r"[^\d.,]", "", price_str)

        # Handle different decimal separators
        if "," in price_clean and "." in price_clean:
            # Assume last separator is decimal
            if price_clean.rfind(",") > price_clean.rfind("."):
                price_clean = price_clean.replace(".", "").replace(",", ".")
            else:
                price_clean = price_clean.replace(",", "")
        elif "," in price_clean:
            # Could be thousands separator or decimal
            if price_clean.count(",") == 1 and len(price_clean.split(",")[1]) <= 2:
                price_clean = price_clean.replace(",", ".")
            else:
                price_clean = price_clean.replace(",", "")

        try:
            return float(price_clean)
        except ValueError:
            return None

    def _extract_currency(self, price_str: str) -> str:
        """
        Extract currency from price string.

        @description Identifies currency symbols or codes in price strings
        @param price_str: Price string containing currency information
        @returns: Currency code (e.g., 'USD', 'EUR', 'GBP')
        """
        price_str = price_str.upper()

        # Currency symbols
        if "$" in price_str:
            return "USD"
        elif "€" in price_str:
            return "EUR"
        elif "£" in price_str:
            return "GBP"
        elif "¥" in price_str:
            return "JPY"

        # Currency codes
        currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
        for currency in currencies:
            if currency in price_str:
                return currency

        return "USD"  # Default

    def _normalize_availability(self, availability: str) -> str:
        """
        Normalize availability status.

        @description Standardizes availability strings to common values
        @param availability: Raw availability string
        @returns: Normalized availability status
        """
        availability_lower = availability.lower()

        if any(
            word in availability_lower for word in ["in stock", "available", "ready"]
        ):
            return "in_stock"
        elif any(
            word in availability_lower
            for word in ["out of stock", "unavailable", "sold out"]
        ):
            return "out_of_stock"
        elif any(
            word in availability_lower
            for word in ["pre-order", "preorder", "coming soon"]
        ):
            return "pre_order"
        elif any(
            word in availability_lower for word in ["limited", "few left", "last"]
        ):
            return "limited"

        return "unknown"

    def _parse_float(self, value: Any) -> Optional[float]:
        """
        Parse float value safely.

        @description Safely converts various types to float
        @param value: Value to convert
        @returns: Float value or None if conversion fails
        """
        if value is None:
            return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, value: Any) -> Optional[int]:
        """
        Parse integer value safely.

        @description Safely converts various types to integer
        @param value: Value to convert
        @returns: Integer value or None if conversion fails
        """
        if value is None:
            return None

        try:
            return int(float(value))  # Convert via float to handle decimals
        except (ValueError, TypeError):
            return None

    def _infer_category_from_url(self, url: str) -> Optional[str]:
        """
        Infer product category from URL path.

        @description Analyzes URL structure to determine product category
        @param url: Product URL
        @returns: Inferred category or None
        """
        path = urlparse(url).path.lower()

        categories = {
            "electronics": ["electronics", "computers", "phones", "tablets"],
            "clothing": ["clothing", "fashion", "apparel", "shoes"],
            "books": ["books", "literature", "reading"],
            "home": ["home", "furniture", "kitchen", "garden"],
            "sports": ["sports", "fitness", "outdoor"],
            "toys": ["toys", "games", "children"],
        }

        for category, keywords in categories.items():
            if any(keyword in path for keyword in keywords):
                return category

        return None

    def _process_images(self, images: Any, base_url: str) -> List[str]:
        """
        Process product images.

        @description Normalizes image URLs and handles relative paths
        @param images: Raw image data
        @param base_url: Base URL for resolving relative paths
        @returns: List of processed image URLs
        """
        if not images:
            return []

        if isinstance(images, str):
            return [images]
        elif isinstance(images, list):
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    # Handle relative URLs
                    if img.startswith("/"):
                        from urllib.parse import urljoin

                        img = urljoin(base_url, img)
                    processed_images.append(img)
            return processed_images

        return []

    def _process_specifications(self, specifications: Any) -> Dict[str, Any]:
        """
        Process product specifications.

        @description Normalizes product specifications into structured format
        @param specifications: Raw specifications data
        @returns: Processed specifications dictionary
        """
        if not specifications:
            return {}

        if isinstance(specifications, dict):
            return {k: self._clean_text(str(v)) for k, v in specifications.items()}
        elif isinstance(specifications, str):
            # Try to parse simple key-value format
            specs = {}
            lines = specifications.split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    specs[key.strip()] = value.strip()
            return specs

        return {}


class ContentProcessorFactory:
    """
    Factory for creating content processors based on content type.

    @description Provides a centralized way to create and manage content processors
    for different types of web content with proper processor selection
    """

    def __init__(self):
        """Initialize the processor factory with available processors."""
        self._processors = {
            "blog": BlogContentProcessor(),
            "prompt": PromptContentProcessor(),
            "article": ArticleContentProcessor(),
            "product": ProductContentProcessor(),
        }

    def get_processor(self, content_type: str) -> BaseContentProcessor:
        """
        Get processor for specified content type.

        @description Returns the appropriate processor for the given content type
        @param content_type: Type of content to process
        @returns: Content processor instance
        @throws ValueError: If content type is not supported

        @example
        # Get blog processor
        factory = ContentProcessorFactory()
        processor = factory.get_processor("blog")
        processed_data = await processor.process(raw_data, url)
        """
        if content_type not in self._processors:
            raise ValueError(f"Unsupported content type: {content_type}")

        return self._processors[content_type]

    def register_processor(
        self, content_type: str, processor: BaseContentProcessor
    ) -> None:
        """
        Register a new content processor.

        @description Allows registration of custom processors for new content types
        @param content_type: Content type identifier
        @param processor: Processor instance for the content type

        @example
        # Register custom processor
        factory.register_processor("news", NewsContentProcessor())
        """
        self._processors[content_type] = processor

    def get_supported_types(self) -> List[str]:
        """
        Get list of supported content types.

        @description Returns all currently supported content types
        @returns: List of supported content type identifiers
        """
        return list(self._processors.keys())
