FROM python:3.13-slim

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends git

# Clone the repository
RUN git clone https://github.com/lakshmanok/generative-ai-design-patterns /generative-ai-design-patterns

# Set the working directory to the application folder
WORKDIR /generative-ai-design-patterns/composable_app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Command to run the application
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
