## Usage

Documented uses of inference optimization in the wild:
* [AWS Inferentia2](https://aws.amazon.com/blogs/machine-learning/faster-llms-with-speculative-decoding-and-aws-inferentia2/) demonstrates speculative decoding with Llama-2-70B/7B models, using a smaller draft model to accelerate inference while maintaining accuracy on their custom AI chips.
* [NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) provides comprehensive inference optimization through TensorRT-LLM, including continuous batching, speculative inference, attention optimizations, and model compression techniques for enterprise deployment.
* [Anthropic Claude](https://latitude-blog.ghost.io/blog/scaling-llms-with-batch-processing-ultimate-guide/) implemented dynamic batching resulting in 37% increased throughput, 28% reduced latency, and processing 1.2 million more queries per day through intelligent batch size management.

-------
To contribute a use case, submit a pull request. Make sure to link to a publicly accessible blog post or article that has the relevant technical details.
