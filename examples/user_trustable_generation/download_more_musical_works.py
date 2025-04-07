"""
Download Additional Musical Works Information

This script downloads additional information about composers' musical works
(more operas, symphonies, and cantatas) from Wikipedia.
"""

import os
import requests
from bs4 import BeautifulSoup

def get_wikipedia_content(title):
    """Get content from Wikipedia article."""
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get main content
    content = soup.find(id="mw-content-text").find(class_="mw-parser-output")

    # Extract text from paragraphs
    text = ""
    for p in content.find_all("p"):
        text += p.get_text()

    return text

def create_additional_musical_works_texts():
    """Create text files for additional musical works from Wikipedia."""
    # Define additional works to download
    additional_works = {
        "mozart": [
            "Symphony_No._40_(Mozart)",
            "Piano_Concerto_No._21_(Mozart)",
            "Requiem_(Mozart)",
            "Le_nozze_di_Figaro"
        ],
        "beethoven": [
            "Symphony_No._3_(Beethoven)",
            "Symphony_No._6_(Beethoven)",
            "Piano_Sonata_No._14_(Beethoven)",
            "String_Quartet_No._14_(Beethoven)"
        ],
        "bach": [
            "The_Well-Tempered_Clavier",
            "Goldberg_Variations",
            "St_John_Passion",
            "Cello_Suites_(Bach)"
        ]
    }

    # Create directory if it doesn't exist
    if not os.path.exists("raw_texts"):
        os.makedirs("raw_texts")

    # Get content for each musical work and save to file
    for composer, works in additional_works.items():
        print(f"Downloading works for {composer}...")
        for work in works:
            try:
                # Ensure work is a string before using replace
                if not isinstance(work, str):
                    print(f"Error: Expected string but got {type(work)} for {work}")
                    continue

                text = get_wikipedia_content(work)

                # Create a safe filename
                safe_work = work.lower().replace('(', '').replace(')', '').replace('_', '-')
                filename = f"raw_texts/{composer}_{safe_work}.txt"

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Created {filename}")
            except Exception as e:
                print(f"Error creating file for {work}: {e}")

if __name__ == "__main__":
    print("Starting download of additional musical works information...")
    create_additional_musical_works_texts()
    print("Additional musical works content download complete!")
