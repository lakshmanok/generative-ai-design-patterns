import os
import statistics
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from locust import User, task, between

if os.path.exists("../keys.env"):
    load_dotenv("../keys.env")
else:
    raise FileNotFoundError("keys.env not found")

class AnthropicUser(User):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        # Initialize Anthropic client
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @task
    def measure_inference_metrics(self):
        # Record start time for time to first token
        start_time = time.time()

        # Make the API call
        response = self.client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": "Write a short story about a robot learning to paint."}
            ],
            stream=True
        )

        # Track metrics
        first_token_time = None
        total_tokens = 0
        response_times = []

        # Process the stream
        for chunk in response:
            if first_token_time is None:
                first_token_time = time.time() - start_time

            if chunk.type == "content_block_delta":
                total_tokens += len(chunk.delta.text.split())
                response_times.append(time.time() - start_time)

        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        # Record metrics
        self.environment.events.request.fire(
            request_type="POST",
            name="anthropic_inference",
            response_time=total_time * 1000,  # Convert to milliseconds
            response_length=total_tokens,
            exception=None,
            context={
                "tokens_per_second": tokens_per_second,
                "time_to_first_token": first_token_time * 1000,  # Convert to milliseconds
                "response_time_distribution": {
                    "mean": statistics.mean(response_times) * 1000,
                    "median": statistics.median(response_times) * 1000,
                    "p95": statistics.quantiles(response_times, n=20)[18] * 1000,  # 95th percentile
                    "p99": statistics.quantiles(response_times, n=100)[98] * 1000,  # 99th percentile
                }
            }
        )
