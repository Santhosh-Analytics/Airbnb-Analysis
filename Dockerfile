# Use Miniconda as the base image

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Performance hints for uv in containers

ENV UV_COMPILE_BYTECODE=1 \

    UV_LINK_MODE=copy \

    UV_PYTHON_DOWNLOADS=0

# Copy the environment.yml file to the working directory
COPY pyproject.toml uv.lock* .python-version* ./

# Create the Conda environment using Mamba
RUN uv sync --frozen --no-group dev

# Now copy the app source
COPY . .

# Check if Streamlit is installed in the activated environment
ENV PATH="/app/.airbnb/bin:$PATH"
# Copy the rest of your app files into the working directory

RUN streamlit --version
# Expose the port that Streamlit uses
EXPOSE 8501

# Set the entry point to run your Streamlit app
CMD ["uv", "run", "streamlit", "run", "Main_mod.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
