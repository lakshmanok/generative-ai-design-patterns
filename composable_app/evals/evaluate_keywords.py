# Demonstrates how you can use the evals.log to carry out evaluations
# This is a rather simple evaluation because it doesn't need external (outcomes) data

import json
import os
from agents.article import Article # needed for eval() to work
from scipy import stats
from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def get_records(target: str = "initial_draft"):
    records = []
    source_dir = os.path.dirname(os.path.abspath(__file__))
    evals_file = os.path.join(source_dir, "../evals.log")
    with open(evals_file) as ifp:
        for line in ifp.readlines():
            obj = json.loads(line)
            if obj['target'] == target:
                article = eval(obj['ai_response'])
                records.append(article)

    return records


def evaluate(keywords: List[str], embedding_model) -> float:
    # if we have 5 keywords, it is ideal. anything more or less is penalized
    score = 1.0 - (np.abs(len(keywords) - 5) / 5.0)
    score = np.min([1.0, np.max([0.0, score])])

    # the more diverse the set of keywords, the better
    # we calculate diversity as variance of the embeddings
    embeds = [np.mean(embedding_model.encode(keyword)) for keyword in keywords]
    score += np.var(embeds)

    return score

if __name__ == "__main__":
    articles = get_records()

    logger.warning("Downloading model from Huggingface ... will take a while the first time")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    scores = []
    for article in articles:
        article_score = evaluate(article.index_keywords, embed_model)
        print(article.title, article_score)
        scores.append(article_score)

    stats = stats.describe(scores)
    print(stats)
