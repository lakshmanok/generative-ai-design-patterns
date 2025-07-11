FROM python:3.13-slim

# Set working directory
WORKDIR /composable_app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create non-root user
RUN useradd -m appuser

# Set permissions and create log files
RUN touch *.log 2>/dev/null || true && \
    chown -R appuser:appuser /composable_app

# Set environment variables
ENV PYTHONPATH=/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]