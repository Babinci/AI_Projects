# System Patterns

## System Architecture

- Python with unified virtual environment (myenv)
- Docker containers for isolation and deployment
- Cloned/forked repositories for specialized functionality

## Database Strategy

- Moving from Django/SQLite to Supabase with FastAPI
- PostgreSQL with pg_vector for embedding storage and retrieval
- Potential for Redis integration for caching

## Component Organization

- Core Backend: FastAPI application with Supabase integration
- MCP Servers: Model Context Protocol servers for Claude Desktop
- Specialized Projects: Audio, Visual, and LLM directories
- Memory Bank: Project documentation and context

## Integration Patterns

- REST APIs for service communication
- MCP for model context integration
- Docker for deployment standardization