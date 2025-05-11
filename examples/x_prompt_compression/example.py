#!/usr/bin/env python3
# Project Gutenberg Book Analysis with Prompt Compression Techniques
# This example shows how to handle a large text like a book that exceeds LLM context windows

import re
import json
import asyncio
import aiohttp
import math
from typing import List, Dict, Any, Optional, Union

# In a real implementation, you would use an LLM API client here
import openai  # Hypothetical import

from dotenv import load_dotenv
import os

if os.path.exists("examples/saved_keys.env"):
    load_dotenv("examples/saved_keys.env")
else:
    raise FileNotFoundError("examples/saved_keys.env not found")


openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. LOADING AND PREPROCESSING
async def load_gutenberg_book(title: str, book_id: int = None, local_path: str = None) -> Optional[str]:
    """
    Load a book from Project Gutenberg (either from API or local cache)
    """
    print(f"Loading {title}...")

    try:
        if local_path:
            # Load from local file
            with open(local_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif book_id:
            # Fetch from Project Gutenberg
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        # Try alternative URL format
                        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                        async with session.get(alt_url) as alt_response:
                            if alt_response.status != 200:
                                print(f"Failed to fetch book (ID: {book_id})")
                                return None
                            text = await alt_response.text()
                    else:
                        text = await response.text()
        else:
            print("Error: Either book_id or local_path must be provided")
            return None

        # Clean up Gutenberg header and footer
        text = re.sub(r'^[\s\S]*?START OF (THIS|THE) PROJECT GUTENBERG EBOOK[\s\S]*?\*\*\*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK[\s\S]*$', '', text, flags=re.IGNORECASE)
        return text.strip()

    except Exception as e:
        print(f"Error loading book: {e}")
        return None

# 2. TOKENIZATION AND CHUNKING
def estimate_token_count(text: str) -> int:
    """
    Estimate token count (rough approximation)
    1 token ≈ 0.75 words in English
    """
    return math.ceil(len(re.findall(r'\S+', text)) * 1.33)

def split_into_chunks(text: str, max_chunk_tokens: int = 4000) -> List[str]:
    """
    Split text into chunks based on estimated token count
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for paragraph in paragraphs:
        paragraph_tokens = estimate_token_count(paragraph)

        # If adding this paragraph would exceed the limit, start a new chunk
        if current_token_count + paragraph_tokens > max_chunk_tokens and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_token_count = paragraph_tokens
        else:
            current_chunk.append(paragraph)
            current_token_count += paragraph_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

# 3. PROMPT COMPRESSION TECHNIQUES
# 3.1 Semantic Compression - Using an LLM to compress text
async def semantic_compression(text: str, target_tokens: int = 1000) -> str:
    """
    Use an LLM to compress text while preserving key information
    """
    prompt = f"""
    Compress the following text while preserving key information, themes, characters, and plot elements.
    Your output should be approximately {target_tokens} tokens.

    TEXT TO COMPRESS:
    {text}

    COMPRESSED VERSION:
    """

    try:
        response = await openai.Completion.acreate(
            model="gpt-4",  # Use appropriate model
            prompt=prompt,
            max_tokens=target_tokens,
            temperature=0.3
        )

        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in semantic compression: {e}")
        # Fallback to simple extractive compression if the API call fails
        return simple_extractive_summary(text, target_tokens)

# 3.2 Extractive Summarization (fallback method)
def simple_extractive_summary(text: str, target_tokens: int) -> str:
    """
    Simple extractive summarization as a fallback method
    """
    paragraphs = re.split(r'\n\s*\n', text)
    compression_ratio = target_tokens / estimate_token_count(text)

    # Take first sentence from each paragraph, preserving more important paragraphs
    # (beginning, end, and those with key character names)
    important_keywords = ['Elizabeth', 'Darcy', 'Bennet', 'marriage', 'pride', 'prejudice']

    # Score paragraphs by importance
    scored_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        score = 0

        # Prioritize beginning and end
        if i < len(paragraphs) * 0.1:
            score += 2
        if i > len(paragraphs) * 0.9:
            score += 2

        # Check for important keywords
        for keyword in important_keywords:
            if keyword in paragraph:
                score += 1

        scored_paragraphs.append({"paragraph": paragraph, "score": score})

    # Sort by score and take top paragraphs
    sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x["score"], reverse=True)
    selected_paragraphs = [item["paragraph"] for item in sorted_paragraphs[:int(len(paragraphs) * compression_ratio)]]

    return '\n\n'.join(selected_paragraphs)

# 3.3 Hierarchical Compression
async def hierarchical_compression(chunks: List[str], depth: int = 1, max_depth: int = 3) -> List[str]:
    """
    Apply hierarchical compression to chunks of text
    """
    if len(chunks) <= 1 or depth > max_depth:
        return chunks

    # Compress each chunk
    compressed_chunks = []
    for chunk in chunks:
        compressed = await semantic_compression(chunk)
        compressed_chunks.append(compressed)

    # If the total is still too large, recursively compress
    total_size = sum(estimate_token_count(chunk) for chunk in compressed_chunks)

    if total_size > 8000 and len(chunks) > 1:  # If still too large for context window
        # Combine into larger chunks and compress again
        combined_chunks = []
        for i in range(0, len(compressed_chunks), 2):
            if i + 1 < len(compressed_chunks):
                combined_chunks.append(compressed_chunks[i] + '\n\n' + compressed_chunks[i + 1])
            else:
                combined_chunks.append(compressed_chunks[i])

        return await hierarchical_compression(combined_chunks, depth + 1, max_depth)

    return compressed_chunks

# 4. ANALYSIS FUNCTIONS
# 4.1 Extract Key Themes
async def extract_key_themes(compressed_text: str) -> List[Dict[str, str]]:
    """
    Extract key themes from compressed text
    """
    prompt = f"""
    Analyze the following text and extract the top 10 themes or motifs.
    For each theme, provide a brief explanation of its significance.

    TEXT:
    {compressed_text}

    FORMAT YOUR RESPONSE AS JSON:
    [
      {{"theme": "Theme name", "significance": "Brief explanation"}},
      ...
    ]
    """

    try:
        response = await openai.Completion.acreate(
            model="gpt-4",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.3
        )

        return json.loads(response.choices[0].text.strip())
    except Exception as e:
        print(f"Error extracting themes: {e}")
        return [{"theme": "Error", "significance": "Could not extract themes"}]

# 4.2 Character Analysis
async def analyze_characters(compressed_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze characters from compressed text
    """
    prompt = f"""
    Analyze the following text and identify the main characters.
    For each character, provide:
    1. A brief description
    2. Their key traits
    3. Their role in the story
    4. Their relationships with other characters

    TEXT:
    {compressed_text}

    FORMAT YOUR RESPONSE AS JSON:
    {{
      "characters": [
        {{
          "name": "Character name",
          "description": "Brief description",
          "traits": ["trait1", "trait2", ...],
          "role": "Role in story",
          "relationships": [{{"with": "Other character", "nature": "Nature of relationship"}}]
        }},
        ...
      ]
    }}
    """

    try:
        response = await openai.Completion.acreate(
            model="gpt-4",
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3
        )

        return json.loads(response.choices[0].text.strip())
    except Exception as e:
        print(f"Error analyzing characters: {e}")
        return {"characters": [{"name": "Error", "description": "Could not analyze characters"}]}

# 4.3 Plot Analysis
async def analyze_plot(compressed_text: str) -> Dict[str, Any]:
    """
    Analyze plot structure from compressed text
    """
    prompt = f"""
    Analyze the following text and provide a structured analysis of the plot.
    Include:
    1. Exposition/setup
    2. Rising action
    3. Climax
    4. Falling action
    5. Resolution

    Also identify key turning points and their significance.

    TEXT:
    {compressed_text}

    FORMAT YOUR RESPONSE AS JSON:
    {{
      "structure": {{
        "exposition": "Description",
        "risingAction": "Description",
        "climax": "Description",
        "fallingAction": "Description",
        "resolution": "Description"
      }},
      "turningPoints": [
        {{"event": "Event description", "significance": "Significance explanation"}}
      ]
    }}
    """

    try:
        response = await openai.Completion.acreate(
            model="gpt-4",
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3
        )

        return json.loads(response.choices[0].text.strip())
    except Exception as e:
        print(f"Error analyzing plot: {e}")
        return {
            "structure": {"exposition": "Error analyzing plot"},
            "turningPoints": []
        }

# 5. MAIN ANALYSIS FUNCTION
async def analyze_gutenberg_book(title: str, book_id: int = None, local_path: str = None) -> Dict[str, Any]:
    """
    Analyze a book from Project Gutenberg
    """
    print(f"Starting analysis of {title}...")

    # Load and preprocess the book
    full_text = await load_gutenberg_book(title, book_id, local_path)
    if not full_text:
        return {"error": "Failed to load book"}

    total_tokens = estimate_token_count(full_text)
    print(f"Book loaded: ~{total_tokens} tokens (exceeds standard context windows)")

    # Split into manageable chunks
    chunks = split_into_chunks(full_text)
    print(f"Split into {len(chunks)} chunks for processing")

    # Apply hierarchical compression
    print("Applying hierarchical compression...")
    compressed_chunks = await hierarchical_compression(chunks)
    compressed_text = "\n\n---\n\n".join(compressed_chunks)
    compressed_tokens = estimate_token_count(compressed_text)
    print(f"Compressed to ~{compressed_tokens} tokens ({compressed_tokens/total_tokens*100:.1f}% of original)")

    # Perform analyses
    print("Extracting key themes...")
    themes = await extract_key_themes(compressed_text)

    print("Analyzing characters...")
    characters = await analyze_characters(compressed_text)

    print("Analyzing plot structure...")
    plot = await analyze_plot(compressed_text)

    # Return comprehensive analysis
    return {
        "title": title,
        "originalSize": total_tokens,
        "compressedSize": compressed_tokens,
        "compressionRatio": f"{compressed_tokens/total_tokens*100:.1f}%",
        "themes": themes,
        "characters": characters,
        "plot": plot
    }

# 6. DEMO EXECUTION
async def run_demo() -> None:
    """
    Run a demonstration of the book analysis
    """
    book_title = "Pride and Prejudice"
    book_id = 1342  # Project Gutenberg ID for Pride and Prejudice
    local_path = "./pride_and_prejudice.txt"  # Alternative local path

    print("===== GUTENBERG BOOK ANALYSIS WITH PROMPT COMPRESSION =====")
    print("This demonstrates how to analyze a book that exceeds LLM context windows")
    print("using prompt compression techniques\n")

    # In a real implementation, this would perform the actual analysis
    # analysis = await analyze_gutenberg_book(book_title, book_id=book_id)

    # For this demo, we'll show a simulated analysis result
    simulated_analysis = {
        "title": "Pride and Prejudice",
        "originalSize": 124500,
        "compressedSize": 7800,
        "compressionRatio": "6.3%",
        "themes": [
            {
                "theme": "Pride",
                "significance": "Represented by Darcy, whose excessive pride causes him to look down on those he considers socially inferior"
            },
            {
                "theme": "Prejudice",
                "significance": "Embodied by Elizabeth, whose initial prejudice against Darcy blinds her to his true character"
            },
            {
                "theme": "Social Class and Mobility",
                "significance": "Explores the rigid class structure of Regency England and the limitations it imposes"
            },
            {
                "theme": "Marriage and Economic Security",
                "significance": "Illustrates how marriage was often an economic necessity for women of the period"
            },
            {
                "theme": "Personal Growth and Self-Discovery",
                "significance": "Both Elizabeth and Darcy undergo significant character development through confronting their flaws"
            }
            # Additional themes would be included here
        ],
        "characters": {
            "characters": [
                {
                    "name": "Elizabeth Bennet",
                    "description": "The intelligent and lively second daughter of the Bennet family",
                    "traits": ["witty", "intelligent", "prejudiced initially", "independent-minded"],
                    "role": "Protagonist and heroine",
                    "relationships": [
                        {
                            "with": "Mr. Darcy",
                            "nature": "Love interest and eventual husband, initially dislikes him due to prejudice"
                        },
                        {
                            "with": "Jane Bennet",
                            "nature": "Sister and closest confidante"
                        }
                    ]
                },
                {
                    "name": "Fitzwilliam Darcy",
                    "description": "Wealthy, intelligent gentleman with a large estate called Pemberley",
                    "traits": ["proud", "reserved", "honorable", "loyal"],
                    "role": "Male protagonist and hero",
                    "relationships": [
                        {
                            "with": "Elizabeth Bennet",
                            "nature": "Love interest and eventual wife, initially offends her with his pride"
                        },
                        {
                            "with": "Charles Bingley",
                            "nature": "Close friend whom he initially dissuades from pursuing Jane"
                        }
                    ]
                }
                # Additional characters would be included here
            ]
        },
        "plot": {
            "structure": {
                "exposition": "The Bennet family, with five unmarried daughters, becomes acquainted with Mr. Bingley, a wealthy bachelor who moves to Netherfield near their home. His friend Mr. Darcy initially snubs Elizabeth.",
                "risingAction": "Elizabeth forms a prejudice against Darcy while being courted by Mr. Collins and meeting the charming Mr. Wickham, who falsely claims to have been wronged by Darcy. Meanwhile, Jane's relationship with Bingley deteriorates due to Darcy's interference.",
                "climax": "Darcy unexpectedly proposes to Elizabeth, declaring his love despite her family's low connections. She firmly rejects him, confronting him about his pride and his role in separating Jane and Bingley.",
                "fallingAction": "Darcy writes a letter explaining his actions regarding Wickham and Bingley. Elizabeth gradually reassesses her opinions. Later, while visiting Pemberley, she sees Darcy's changed behavior. When Lydia elopes with Wickham, Darcy secretly helps resolve the situation.",
                "resolution": "Bingley returns and proposes to Jane. Darcy proposes to Elizabeth again, now with humility, and she accepts. Both couples marry, demonstrating how they have overcome the initial obstacles of pride and prejudice."
            },
            "turningPoints": [
                {
                    "event": "Darcy's first proposal to Elizabeth",
                    "significance": "Marks the height of their mutual misunderstanding and forces both to confront their flaws"
                },
                {
                    "event": "Elizabeth reading Darcy's letter",
                    "significance": "Begins Elizabeth's reassessment of Darcy and her own prejudices"
                },
                {
                    "event": "Elizabeth's visit to Pemberley",
                    "significance": "Allows Elizabeth to see Darcy in a new light and witness his changed behavior"
                },
                {
                    "event": "Lydia's elopement with Wickham",
                    "significance": "Gives Darcy an opportunity to demonstrate his true character by helping the Bennet family"
                }
            ]
        }
    }

    print("Analysis complete! Results:")
    print(json.dumps(simulated_analysis, indent=2))

    print("\nThis demonstrates how we can use context window management and prompt compression")
    print("to analyze books that would otherwise exceed LLM context windows.")

# Run the demo when executed directly
if __name__ == "__main__":
    asyncio.run(run_demo())
