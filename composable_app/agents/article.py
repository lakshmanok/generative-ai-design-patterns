from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import List

@dataclass
class Article:
    full_text: str = Field("Full text of article in Markdown format.")
    title: str = Field("Title of article suitable for audience.")
    key_lesson: str = Field("One sentence summarizing the key learning point in article.")
    index_keywords: List[str] = Field("List of keywords or phrases by which this answer can be indexed.")

    def to_markdown(self):
        star = "\n* "
        return (
f"""## {self.title}
{self.key_lesson}

### Details
{self.full_text}

### Keywords
{star.join(["", *self.index_keywords])}
""")