import os
import time
from openai import OpenAI
from dotenv import load_dotenv

if os.path.exists("examples/keys.env"):
    load_dotenv("examples/keys.env")
else:
    raise FileNotFoundError("examples/keys.env not found")

def main():
    client = OpenAI()

    # Create a long system prompt that exceeds 1024 tokens
    long_system_prompt = """You are an expert AI assistant with deep knowledge across multiple domains.
    Your responses should be comprehensive, accurate, and well-structured. You should always:
    1. Provide detailed explanations
    2. Include relevant examples
    3. Cite sources when possible
    4. Consider multiple perspectives
    5. Highlight key points
    6. Use clear and concise language
    7. Maintain a professional tone
    8. Be helpful and informative
    9. Acknowledge limitations
    10. Suggest further reading

    Additional guidelines for your responses:
    - Always start with a clear introduction
    - Use bullet points for key information
    - Include relevant statistics when available
    - Explain complex concepts in simple terms
    - Provide historical context when relevant
    - Consider cultural implications
    - Address potential misconceptions
    - Include practical applications
    - Suggest related topics
    - End with a clear conclusion

    Your expertise covers:
    - Science and Technology
    - History and Culture
    - Arts and Literature
    - Business and Economics
    - Health and Medicine
    - Education and Learning
    - Environment and Sustainability
    - Politics and Society
    - Philosophy and Ethics
    - Sports and Recreation

    For each response, you should:
    1. Analyze the question thoroughly
    2. Research the topic if needed
    3. Structure your response logically
    4. Include relevant examples
    5. Provide supporting evidence
    6. Consider different viewpoints
    7. Address potential questions
    8. Suggest related topics
    9. Include practical applications
    10. End with a clear conclusion

    Remember to:
    - Be accurate and precise
    - Use clear language
    - Provide context
    - Include examples
    - Cite sources
    - Consider implications
    - Address limitations
    - Suggest next steps
    - Be helpful and informative
    - Maintain professionalism

    Your responses should demonstrate:
    - Deep understanding
    - Clear communication
    - Logical structure
    - Comprehensive coverage
    - Practical relevance
    - Cultural awareness
    - Ethical consideration
    - Scientific accuracy
    - Historical context
    - Future implications

    When providing information, ensure:
    - Accuracy of facts
    - Clarity of explanation
    - Relevance of examples
    - Depth of analysis
    - Breadth of coverage
    - Practical applicability
    - Cultural sensitivity
    - Ethical consideration
    - Scientific validity
    - Historical accuracy

    Your communication style should be:
    - Professional
    - Clear
    - Concise
    - Engaging
    - Informative
    - Helpful
    - Respectful
    - Accurate
    - Comprehensive
    - Well-structured

    Remember to always:
    - Verify information
    - Provide context
    - Include examples
    - Cite sources
    - Consider implications
    - Address limitations
    - Suggest next steps
    - Be helpful
    - Maintain professionalism
    - Ensure accuracy"""

    # Common parameters for all requests
    common_params = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": long_system_prompt},
            {"role": "user", "content": "What is the capital of Monaco?"}
        ],
        "temperature": 0.0,  # Set to 0 for deterministic responses
        "seed": 42,  # Fixed seed for deterministic responses
        "response_format": {"type": "text"},  # Ensure consistent response format
        "cache_control": "force-cache"  # Force cache usage
    }

    # Run 5 identical requests and measure time
    timings = []
    response_hashes = []

    print("Making 5 identical requests to test server-side caching...")
    print("Note: OpenAI's server-side cache typically has a TTL of a few minutes")

    for i in range(5):
        print(f"\nRun {i+1}:")
        start_time = time.time()
        try:
            response = client.chat.completions.create(**common_params)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timings.append(elapsed_time)

            # Store response hash for comparison
            response_hash = hash(response.choices[0].message.content)
            response_hashes.append(response_hash)

            print(f"Time: {elapsed_time:.2f} seconds")
            print(f"Response length: {len(response.choices[0].message.content)} characters")
            print(f"Response hash: {response_hash}")

            # Add a small delay between requests to avoid rate limits
            if i < 4:  # Don't delay after the last request
                time.sleep(1)
        except Exception as e:
            print(f"Error on run {i+1}: {str(e)}")
            if "cache_control" in str(e):
                print("Removing cache_control parameter and retrying...")
                del common_params["cache_control"]
                response = client.chat.completions.create(**common_params)
                end_time = time.time()
                elapsed_time = end_time - start_time
                timings.append(elapsed_time)
                response_hash = hash(response.choices[0].message.content)
                response_hashes.append(response_hash)
                print(f"Time: {elapsed_time:.2f} seconds")
                print(f"Response length: {len(response.choices[0].message.content)} characters")
                print(f"Response hash: {response_hash}")

    # Print summary statistics
    print("\nSummary:")
    print(f"Average time: {sum(timings)/len(timings):.2f} seconds")
    print(f"Min time: {min(timings):.2f} seconds")
    print(f"Max time: {max(timings):.2f} seconds")
    print(f"Time difference between fastest and slowest: {max(timings) - min(timings):.2f} seconds")

    # Check if all responses were identical
    all_identical = all(h == response_hashes[0] for h in response_hashes)
    print(f"\nAll responses identical: {all_identical}")

    if not all_identical:
        print("Note: Different response hashes indicate that server-side caching might not be working.")
        print("This could be due to:")
        print("1. Cache TTL expiration")
        print("2. Server load balancing")
        print("3. Rate limiting")
        print("4. Different server instances handling the requests")

if __name__ == "__main__":
    main()
