FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m appuser

# Set up composable_app package structure (do NOT move prompts or pages)
RUN mkdir -p composable_app && \
    mv agents utils evals composable_app/ && \
    find composable_app -type d -exec touch {}/__init__.py \;

# Set permissions and create log files
RUN touch *.log 2>/dev/null || true && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]