# Use uv python base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Performance hints for uv in containers
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (uv creates .venv automatically)
RUN uv sync --frozen 

# Now copy the app source
COPY . .

# Check if Streamlit is installed
RUN uv run streamlit --version

# Expose the port that Streamlit uses
EXPOSE 8501

# Set the entry point to run your Streamlit app
CMD ["uv", "run", "streamlit", "run", "St_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

