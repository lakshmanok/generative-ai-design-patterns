# generative-ai-design-patterns
A [catalog](./catalog.html) of design patterns when building generative AI applications



Please see:
https://medium.com/@lakshmanok/generative-ai-design-patterns-8eb1d937fccc

## Book on Gen AI Design Patterns?
Design Patterns are common solutions to recurring problems. The solutions are not perfect - each of them comes with tradeoffs and a choice of one often brings about new problems that need to be addressed. So, an experienced practitioner will have to be pragmatic when choosing among these solutions.

Because I co-authored the O'Reilly book [Machine Learning Design Patterns](https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783), I frequently get asked whether I plan to update the book to add GenAI design patterns. I believe it's too early for a book, but we do know enough to create an online, easily editable catalog. I've gotten started on one, think of it as a community catalog, and welcome your contributions.

## It's too early for a book
On one hand, I believe it's too early for a patterns book on Gen AI because the technology is still evolving and no one truly knows best practices yet:
1. It may be tempting to position today's hot framework as "best practice", but we are still in the phase where frameworks are improving across the board. For example, prompt templates and few shot learning used to be the best way to build applications a few months ago, but now it's agent frameworks and chain-of-thought. These are not alternatives to one another; the latter are across-the-board better generalizations of the older approach. 
2. Even the leaders in the space are not able to launch without public relations, issues or scalability or availability concerns.
3. Research advances continue to make dramatic changes to GenAI practices. Human Feedback through Reinforcement Learning (RLHF), the key innovation that OpenAI brought to the process with GPT-2 has been made unnecessary by Direct Preference Optimization (DPO), and the ability to avoid reinforcement learning has enabled open-source models to be trained at a fraction of the cost.

Can we not write a book that consists only of the things we are sure about? But, but … we are not sure about anything! Consider what is probably the key architectural design pattern in this space, and the one that countless startups are being built around - Retrieval Augmented Generation (RAG).

We can already glimpse two potential reasons why even RAG may not stand the test of time:
1. A body of recent research suggests that an all-in model that can be created that incorporates a RAG. In other words, you can train an LLM to do RAG, and then just as feature engineering went away with deep learning, RAG will go away in GenAI.
2. Today, large context windows are expensive, take too long, and so are not widely applicable. But inefficient things have a tendency to get optimized; over time, the bar might get low enough that the RAG pattern changes from identifying the most relevant document chunks to just identifying all related documents.

## But just right for a living catalog on GitHub
Even though it's too early to make it a book that needs to remain relevant for a few years, it's also clear that there is a need to capture what today's best practices are. Perhaps a living catalog on GitHub can help cut through the noise and point to where improvements are happening. Then, once the catalog stabilizes (whether in a few months or in a few years), we can freeze it into a book.

Please do feel free to file issues, create a pull-request, etc. When it comes time to create a book, I'll reach out to the top contributors about collaborating on making it a book.
