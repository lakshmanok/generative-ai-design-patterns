from dataclasses import dataclass
from pydantic import Field
from typing import List

@dataclass
class Article:
    full_text: str = Field("Full text of article in Markdown format.")
    title: str = Field("Title of article suitable for audience.")
    summary: str = Field("One sentence summarizing the key learning point in article.")
    index_keywords: List[str] = Field("List of keywords or phrases by which this answer can be indexed.")

    def to_markdown(self):
        newline = "\n"
        return (
f"""## {self.title}
{self.summary}

### Details
{self.full_text}

### Keywords
{newline.join(self.index_keywords)}
""")