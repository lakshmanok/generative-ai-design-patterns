"""
LLM Inference Performance Benchmark Tool

This module provides a comprehensive benchmarking tool for testing LLM inference
performance across different endpoints (OpenAI, vLLM, etc.) with support for
concurrent users, streaming responses, and detailed performance metrics.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RequestResult:
    """Results from a single LLM inference request."""

    user_id: int
    request_id: int
    ttft: float  # Time to first token in seconds
    total_time: float  # Total request time in seconds
    tokens_generated: int
    tokens_per_second: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration parameters for benchmark execution."""

    endpoint_url: str
    num_users: int = 10
    requests_per_user: int = 5
    prompt: str = "Explain quantum computing in simple terms."
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.7
    api_key: Optional[str] = None
    timeout: float = 60.0
    ramp_up_time: float = 5.0  # Time to gradually add users (seconds)


@dataclass
class BenchmarkResults:
    """Aggregated results from a complete benchmark run."""

    config: BenchmarkConfig
    individual_results: List[RequestResult] = field(default_factory=list)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_ttft: float = 0.0
    p95_ttft: float = 0.0
    avg_tokens_per_sec: float = 0.0
    p95_tokens_per_sec: float = 0.0
    total_duration: float = 0.0
    overall_throughput: float = 0.0


class LLMBenchmark:
    """
    Main benchmarking class for LLM inference performance testing.

    Supports multiple API formats (OpenAI, vLLM) and provides detailed
    performance metrics including TTFT, throughput, and latency distributions.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.results = BenchmarkResults(config=config)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "LLMBenchmark":
        """Initialize HTTP session with optimized connection settings."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up HTTP session."""
        if self.session:
            await self.session.close()

    def _prepare_openai_request(self) -> Dict:
        """Prepare request payload for OpenAI-compatible API."""
        return {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": self.config.prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True
        }

    def _prepare_vllm_request(self) -> Dict:
        """Prepare request payload for vLLM API."""
        return {
            "prompt": self.config.prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True
        }

    def _get_headers(self) -> Dict[str, str]:
        """Generate HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _is_openai_endpoint(self) -> bool:
        """Check if the endpoint is OpenAI-compatible."""
        return "openai" in self.config.endpoint_url.lower()

    def _is_lmstudio_endpoint(self) -> bool:
        """Check if the endpoint is LM Studio-compatible."""
        return "127.0.0.1" in self.config.endpoint_url.lower()


    def _extract_content_from_response(self, data: Dict) -> str:
        """Extract content from streaming response data."""
        # Handle OpenAI format
        if 'choices' in data and data['choices']:
            delta = data['choices'][0].get('delta', {})
            return delta.get('content', '')

        # Handle vLLM format
        if 'text' in data:
            return data['text']

        return ''

    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count by splitting on whitespace."""
        return len(text.split()) if text else 0

    async def _make_streaming_request(self, user_id: int, request_id: int) -> RequestResult:
        """
        Execute a single streaming request and measure performance metrics.

        Args:
            user_id: Identifier for the simulated user
            request_id: Identifier for the request within the user's session

        Returns:
            RequestResult containing performance metrics and success status
        """
        start_time = time.time()
        ttft = None
        tokens_generated = 0

        try:
            # Prepare request based on endpoint type
            payload = (self._prepare_openai_request() if (self._is_openai_endpoint() or self._is_lmstudio_endpoint())
                      else self._prepare_vllm_request())
            headers = self._get_headers()

            print(f"Sending request to {self.config.endpoint_url}")
            print(f"Payload: {payload}")
            print(f"Headers: {headers}")

            async with self.session.post(
                self.config.endpoint_url,
                json=payload,
                headers=headers
            ) as response:

                if response.status != 200:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    return self._create_error_result(
                        user_id, request_id, start_time, error_msg
                    )

                async for line in response.content:
                    line_str = line.decode('utf-8').strip()

                    if not line_str.startswith('data: '):
                        continue

                    data_str = line_str[6:]  # Remove 'data: ' prefix

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)
                        content = self._extract_content_from_response(data)

                        if content:
                            # Record time to first token
                            if ttft is None:
                                ttft = time.time() - start_time

                            tokens_generated += self._estimate_token_count(content)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            return self._create_error_result(
                user_id, request_id, start_time, str(e)
            )

        return self._create_success_result(
            user_id, request_id, start_time, ttft, tokens_generated
        )

    def _create_error_result(
        self,
        user_id: int,
        request_id: int,
        start_time: float,
        error_message: str
    ) -> RequestResult:
        """Create a RequestResult for a failed request."""
        return RequestResult(
            user_id=user_id,
            request_id=request_id,
            ttft=0,
            total_time=time.time() - start_time,
            tokens_generated=0,
            tokens_per_second=0,
            success=False,
            error_message=error_message
        )

    def _create_success_result(
        self,
        user_id: int,
        request_id: int,
        start_time: float,
        ttft: Optional[float],
        tokens_generated: int
    ) -> RequestResult:
        """Create a RequestResult for a successful request."""
        total_time = time.time() - start_time

        # Use total time as fallback if no tokens were detected
        if ttft is None:
            ttft = total_time

        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0

        return RequestResult(
            user_id=user_id,
            request_id=request_id,
            ttft=ttft,
            total_time=total_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            success=True
        )

    async def _simulate_user(self, user_id: int) -> List[RequestResult]:
        """
        Simulate a single user making multiple sequential requests.

        Args:
            user_id: Identifier for the simulated user

        Returns:
            List of RequestResult objects for all requests made by this user
        """
        results = []

        for request_id in range(self.config.requests_per_user):
            result = await self._make_streaming_request(user_id, request_id)
            results.append(result)

            # Small delay between requests from the same user
            await asyncio.sleep(0.1)

        return results

    async def _create_delayed_user_task(self, user_id: int, delay: float) -> List[RequestResult]:
        """Create a delayed user simulation task."""
        await asyncio.sleep(delay)
        return await self._simulate_user(user_id)

    def _calculate_aggregate_statistics(self, all_results: List[RequestResult]) -> None:
        """Calculate and store aggregate performance statistics."""
        successful_results = [r for r in all_results if r.success]
        failed_results = [r for r in all_results if not r.success]

        self.results.individual_results = all_results
        self.results.total_requests = len(all_results)
        self.results.successful_requests = len(successful_results)
        self.results.failed_requests = len(failed_results)

        if successful_results:
            ttfts = [r.ttft for r in successful_results]
            tokens_per_secs = [r.tokens_per_second for r in successful_results]
            total_tokens = sum(r.tokens_generated for r in successful_results)

            self.results.avg_ttft = statistics.mean(ttfts)
            self.results.p95_ttft = np.percentile(ttfts, 95)
            self.results.avg_tokens_per_sec = statistics.mean(tokens_per_secs)
            self.results.p95_tokens_per_sec = np.percentile(tokens_per_secs, 95)
            self.results.overall_throughput = total_tokens / self.results.total_duration

    async def run_benchmark(self) -> BenchmarkResults:
        """
        Execute the complete benchmark with multiple concurrent users.

        Returns:
            BenchmarkResults containing all performance metrics and individual results
        """
        print(f"Starting benchmark with {self.config.num_users} users, "
              f"{self.config.requests_per_user} requests per user...")

        start_time = time.time()

        # Create tasks for all users with staggered start times
        tasks = []
        for user_id in range(self.config.num_users):
            delay = (user_id / self.config.num_users) * self.config.ramp_up_time
            task = asyncio.create_task(
                self._create_delayed_user_task(user_id, delay)
            )
            tasks.append(task)

        # Execute all user simulations concurrently
        user_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and process results
        all_results = []
        for user_result in user_results:
            if isinstance(user_result, list):
                all_results.extend(user_result)
            else:
                print(f"User simulation failed with exception: {user_result}")

        # Calculate timing and statistics
        self.results.total_duration = time.time() - start_time
        self._calculate_aggregate_statistics(all_results)

        # Print summary
        print(f"Benchmark completed in {self.results.total_duration:.2f} seconds")
        print(f"Successful requests: {self.results.successful_requests}")
        print(f"Failed requests: {self.results.failed_requests}")

        return self.results

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Generate and display performance visualization plots.

        Args:
            save_path: Optional path to save the plot image
        """
        successful_results = [r for r in self.results.individual_results if r.success]

        if not successful_results:
            print("No successful results to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        self._plot_ttft_distribution(ax1, successful_results)
        self._plot_tokens_per_sec_distribution(ax2, successful_results)
        self._plot_ttft_over_time(ax3, successful_results)
        self._plot_user_performance_comparison(ax4, successful_results)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def _plot_ttft_distribution(self, ax, successful_results: List[RequestResult]) -> None:
        """Plot Time to First Token distribution."""
        ttfts = [r.ttft for r in successful_results]
        ax.hist(ttfts, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Time to First Token (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('TTFT Distribution')
        ax.axvline(self.results.avg_ttft, color='red', linestyle='--',
                   label=f'Mean: {self.results.avg_ttft:.3f}s')
        ax.axvline(self.results.p95_ttft, color='orange', linestyle='--',
                   label=f'P95: {self.results.p95_ttft:.3f}s')
        ax.legend()

    def _plot_tokens_per_sec_distribution(self, ax, successful_results: List[RequestResult]) -> None:
        """Plot tokens per second distribution."""
        tps = [r.tokens_per_second for r in successful_results]
        ax.hist(tps, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Tokens per Second')
        ax.set_ylabel('Frequency')
        ax.set_title('Tokens/Sec Distribution')
        ax.axvline(self.results.avg_tokens_per_sec, color='red', linestyle='--',
                   label=f'Mean: {self.results.avg_tokens_per_sec:.1f}')
        ax.axvline(self.results.p95_tokens_per_sec, color='orange', linestyle='--',
                   label=f'P95: {self.results.p95_tokens_per_sec:.1f}')
        ax.legend()

    def _plot_ttft_over_time(self, ax, successful_results: List[RequestResult]) -> None:
        """Plot TTFT over request sequence."""
        request_times = [
            (r.user_id * self.config.requests_per_user + r.request_id, r.ttft)
            for r in successful_results
        ]
        request_times.sort()
        x_vals, y_vals = zip(*request_times)
        ax.scatter(x_vals, y_vals, alpha=0.6, color='blue')
        ax.set_xlabel('Request Number')
        ax.set_ylabel('TTFT (seconds)')
        ax.set_title('TTFT Over Time')

    def _plot_user_performance_comparison(self, ax, successful_results: List[RequestResult]) -> None:
        """Plot average TTFT comparison across users."""
        user_avg_ttft = {}
        for r in successful_results:
            if r.user_id not in user_avg_ttft:
                user_avg_ttft[r.user_id] = []
            user_avg_ttft[r.user_id].append(r.ttft)

        user_ids = list(user_avg_ttft.keys())
        avg_ttfts = [statistics.mean(user_avg_ttft[uid]) for uid in user_ids]

        ax.bar(user_ids, avg_ttfts, alpha=0.7, color='purple')
        ax.set_xlabel('User ID')
        ax.set_ylabel('Average TTFT (seconds)')
        ax.set_title('Average TTFT per User')

    def print_summary(self) -> None:
        """Print a comprehensive summary of benchmark results."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Endpoint: {self.config.endpoint_url}")
        print(f"Model: {self.config.model}")
        print(f"Users: {self.config.num_users}")
        print(f"Requests per user: {self.config.requests_per_user}")
        print(f"Total requests: {self.results.total_requests}")
        print(f"Successful: {self.results.successful_requests}")
        print(f"Failed: {self.results.failed_requests}")

        if self.results.total_requests > 0:
            success_rate = (self.results.successful_requests / self.results.total_requests) * 100
            print(f"Success rate: {success_rate:.1f}%")

        print(f"Total duration: {self.results.total_duration:.2f}s")
        print()
        print("PERFORMANCE METRICS:")
        print(f"Average TTFT: {self.results.avg_ttft:.3f}s")
        print(f"95th percentile TTFT: {self.results.p95_ttft:.3f}s")
        print(f"Average tokens/sec: {self.results.avg_tokens_per_sec:.1f}")
        print(f"95th percentile tokens/sec: {self.results.p95_tokens_per_sec:.1f}")
        print(f"Overall throughput: {self.results.overall_throughput:.1f} tokens/sec")
        print("="*60)
