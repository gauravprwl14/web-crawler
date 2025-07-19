# Flexible Web Crawler API

A comprehensive and flexible web crawler system built in Python with support for multiple content types, dynamic selectors, and scalable database storage. Perfect for scraping blogs, articles, product listings, and educational content with intelligent content extraction and validation.

## 🚀 Features

- **🎯 Flexible Content Extraction**: Support for blogs, articles, prompts, and products
- **🔧 Dynamic Selectors**: Custom CSS selectors for any website structure  
- **⚡ Batch Processing**: Concurrent crawling of multiple URLs
- **🔄 Background Jobs**: Asynchronous processing for large crawling tasks
- **💾 Database Integration**: Automatic storage with content type-specific tables
- **⏱️ Rate Limiting**: Respectful crawling with configurable delays
- **🤖 Robots.txt Compliance**: Automatic robots.txt checking
- **✅ Content Validation**: Comprehensive validation and error reporting
- **📊 Analytics & Monitoring**: Built-in statistics and health monitoring
- **🔌 REST API**: Complete REST API with interactive documentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Content Types](#content-types)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or poetry
- SQLite (default) or PostgreSQL

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourorg/flexible-crawler.git
cd flexible-crawler

# Install dependencies
pip install -r requirements.txt

# Or using poetry
poetry install
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### Database Setup

```bash
# Initialize database (automatic on first run)
python -m src.main
```

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Development mode
python -m src.main

# Production mode  
python -m src.main production
```

The API will be available at `http://localhost:8000`

### 2. Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### 3. Your First Crawl

```python
import httpx

# Crawl a single blog post
response = httpx.post("http://localhost:8000/api/v1/crawl/single", json={
    "url": "https://example-blog.com/my-post",
    "content_type": "blog",
    "selectors": {
        "title": "h1",
        "content": ".post-content", 
        "author": ".author",
        "published_date": ".date"
    }
})

print(response.json())
```

## ⚙️ Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:///./crawler_data.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Crawler Settings
CRAWLER_TIMEOUT=30
CRAWLER_MAX_RETRIES=3
CRAWLER_DELAY=1.0
CRAWLER_CONCURRENT_REQUESTS=5
CRAWLER_RESPECT_ROBOTS=true

# Logging
LOG_LEVEL=INFO
```

### Configuration Files

The system uses Pydantic settings with support for:
- Environment variables
- `.env` files
- Configuration overrides

## 📝 Content Types

### Blog Posts
Extract blog content with metadata:
```json
{
  "title": "string",
  "content": "string", 
  "author": "string",
  "published_date": "datetime",
  "tags": ["string"],
  "excerpt": "string",
  "reading_time": "integer"
}
```

### Articles  
General article content with SEO data:
```json
{
  "title": "string",
  "content": "string",
  "summary": "string", 
  "keywords": ["string"],
  "reading_time": "integer",
  "topic": "string"
}
```

### Prompts
AI prompts and coding challenges:
```json
{
  "prompt_text": "string",
  "category": "string",
  "difficulty": "string",
  "examples": [{"input": "string", "output": "string"}],
  "programming_language": "string"
}
```

### Products
E-commerce product information:
```json
{
  "name": "string",
  "price": "float",
  "description": "string",
  "rating": "float", 
  "availability": "string",
  "specifications": {"key": "value"}
}
```

## 🌐 API Endpoints

### Crawling Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/crawl/single` | Crawl single URL |
| POST | `/api/v1/crawl/batch` | Crawl multiple URLs |

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/jobs` | Create background job |
| GET | `/api/v1/jobs` | List all jobs |
| GET | `/api/v1/jobs/{job_id}` | Get job status |
| DELETE | `/api/v1/jobs/{job_id}` | Cancel job |

### Content Retrieval

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/content/{type}` | List content by type |
| GET | `/api/v1/content/{type}/{id}` | Get specific content |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/stats` | System statistics |
| GET | `/health` | Health check |

## 💡 Usage Examples

### Single URL Crawl

```python
import httpx

# Basic crawl
response = httpx.post("http://localhost:8000/api/v1/crawl/single", json={
    "url": "https://blog.example.com/post",
    "content_type": "blog"
})

# With custom selectors
response = httpx.post("http://localhost:8000/api/v1/crawl/single", json={
    "url": "https://news.example.com/article", 
    "content_type": "article",
    "selectors": {
        "title": "h1.headline",
        "content": ".article-body",
        "author": ".byline .author",
        "published_date": "time[datetime]"
    }
})
```

### Batch Crawling

```python
# Crawl multiple URLs
response = httpx.post("http://localhost:8000/api/v1/crawl/batch", json={
    "urls": [
        "https://blog.com/post1",
        "https://blog.com/post2", 
        "https://blog.com/post3"
    ],
    "content_type": "blog",
    "max_concurrent": 3
})
```

### Background Jobs

```python
# Create job
job_response = httpx.post("http://localhost:8000/api/v1/jobs", json={
    "name": "Daily News Crawl",
    "urls": ["https://news.com/tech", "https://news.com/business"],
    "content_type": "article",
    "schedule": "immediate"
})

job_id = job_response.json()["job_id"]

# Check status
status = httpx.get(f"http://localhost:8000/api/v1/jobs/{job_id}")
print(f"Progress: {status.json()['progress_percentage']}%")
```

### Retrieve Stored Content

```python
# Get all blogs
blogs = httpx.get("http://localhost:8000/api/v1/content/blog?limit=10")

# Get specific blog
blog_id = "some-blog-id"
blog = httpx.get(f"http://localhost:8000/api/v1/content/blog/{blog_id}")

# Filter content
filtered = httpx.get(
    "http://localhost:8000/api/v1/content/blog",
    params={"filters": '{"author": "John Doe"}'}
)
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Core Crawler  │    │   Database      │
│   Endpoints     │◄──►│   Engine        │◄──►│   Layer         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Job Manager   │    │   Content       │    │   Models &      │
│   & Scheduler   │    │   Processors    │    │   Storage       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Core Crawler**: Built on crawl4ai with enhanced selector support
2. **Content Processors**: Type-specific content cleaning and validation  
3. **Database Layer**: SQLAlchemy with dynamic table management
4. **Job Manager**: Background task processing with progress tracking
5. **API Layer**: FastAPI with comprehensive error handling
6. **Configuration**: Pydantic-based settings management

### Database Schema

- **Content Tables**: Separate tables per content type (blogs, articles, etc.)
- **Job Tracking**: Job status and progress monitoring
- **Result Storage**: Individual crawl results with metadata
- **Error Logging**: Comprehensive error tracking and debugging

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_crawler.py::test_single_crawl
```

## 📊 Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system stats
curl http://localhost:8000/api/v1/stats
```

### Logging

Logs are written to:
- Console (development)
- `logs/crawler_api.log` (production)

Log levels: DEBUG, INFO, WARNING, ERROR

## 🔧 Development

### Project Structure

```
crawl4ai/
├── src/
│   ├── config/          # Configuration management
│   ├── crawler/         # Core crawler components
│   ├── database/        # Database models and managers
│   ├── api/            # FastAPI endpoints and schemas
│   └── main.py         # Application entry point
├── tests/              # Test suite
├── docs/              # Documentation
├── requirements.txt   # Dependencies
└── README.md         # This file
```

### Adding New Content Types

1. **Add Model**: Create new model in `src/database/models.py`
2. **Add Processor**: Create processor in `src/crawler/processors.py`
3. **Update Config**: Add type to `src/config/settings.py`
4. **Add Schema**: Create Pydantic schemas in `src/api/schemas.py`

### Custom Selectors

Selectors support:
- Standard CSS selectors
- Comma-separated fallbacks
- Attribute extraction
- Multiple element selection

Example:
```json
{
  "title": "h1, .title, .headline",
  "content": "article, .content, .post-body",
  "author": ".author, .byline, [data-author]",
  "date": "time[datetime], .date, .published"
}
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 8000

CMD ["python", "-m", "src.main", "production"]
```

### Production Considerations

1. **Database**: Use PostgreSQL for production
2. **Caching**: Add Redis for job queuing and rate limiting
3. **Monitoring**: Integrate with Prometheus/Grafana
4. **Scaling**: Use multiple workers with load balancer
5. **Security**: Enable authentication and HTTPS

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests before committing
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [crawl4ai](https://github.com/unclecode/crawl4ai) for core crawling
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [SQLAlchemy](https://www.sqlalchemy.org/) for database operations
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📞 Support

- 📖 Documentation: `http://localhost:8000/docs`
- 🐛 Issues: [GitHub Issues](https://github.com/yourorg/flexible-crawler/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourorg/flexible-crawler/discussions)
- 📧 Email: support@yourorg.com

---

**Happy Crawling! 🕷️✨** 