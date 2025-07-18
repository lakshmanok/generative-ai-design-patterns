# Generative AI Design Patterns
|    |    |
| -- | -- |
| <a href="https://www.oreilly.com/library/view/generative-ai-design/9798341622654/"><img src="diagrams/cover.png" width="500"></a> | Code repo for in-press O'Reilly book on GenAI design patterns by Valliappa Lakshmanan and Hannes Hapke. https://www.oreilly.com/library/view/generative-ai-design/9798341622654/ <br/><br/> 

## Summary of patterns
These are the 32 design patterns covered in the book:

<details>
<summary>Chapter 2: Controlling Style (Patterns 1-5)</summary>

| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example |
| -------------: | :----------- | :------ | :------- | :-------------- | :----------- |
| 1 | Logits Masking | Need to ensure generated text conforms to specific style rules for brand, accuracy, or compliance reasons. | Intercept the generation at the sampling stage to zero out probabilities of continuations that don't meet the rules | Use words associated with specific brand; avoid repeating factual information; make content compliant with style book | [examples/01_logits_masking](examples/01_logits_masking)|
| 2 | Grammar | Need text to conform to a specific format or data schema for downstream processing. | Specify rules as a formal grammar (e.g., BNF) or schema that the model framework applies to constrain token generation. | Generating valid SQL timestamps; extracting structured data in a specific format; ensuring output conforms to JSON schema. | [examples/02_grammar](examples/02_grammar) |
| 3 | Style Transfer | Need to convert content into a form that mimics specific tone and style that is difficult to express through rules, but can be shown through example conversions. | Use few-shot learning or model fine-tuning to teach the model how to convert content to the desired style. | Rewriting generic content to match brand guidelines; converting academic papers to blog posts; transforming image and text content for different social media platforms or audiences. | [examples/03_style_transfer](examples/03_style_transfer) |
| 4 | Reverse Neutralization | Need to generate content in a specific style that can be shown through example content. | Use an LLM to generate content in an intermediate neutral form, and a fine-tuned LLM to convert that neutral form into the desired style. | Generating letters in region-specific legalese; generating emails in personal style. | [examples/04_reverse_neutralization](examples/04_reverse_neutralization) |
| 5 | Content Optimization | Need to determine optimal style for content without knowing which factors matter. | Generate pairs of content, compare them using an evaluator, create a preference dataset, and perform preference tuning. | Optimizing ad copy, marketing content, or educational materials where effective style factors are unknown. | [examples/05_content_optimization](examples/05_content_optimization) |

</details>

<details>
<summary>Chapters 3 and 4: Adding Knowledge (Patterns 6-12) </summary>
  
| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example |
| -------------: | :----------- | :------ | :------- | :-------------- | :----------- |
| 6 | Basic RAG | Knowledge cutoff, confidential data, and hallucinations pose problems for zero-shot generation by LLMs. | Ground the response generated by the LLM by adding relevant information from a knowledge base into the prompt context. | The applications of RAG are constantly expanding as the technology evolves. | [examples/06_basic_rag](examples/06_basic_rag) |
| 7 | Semantic Indexing | Traditional keyword indexing/lookup approaches fail when documents get more complex, contain different media types like images or tables, or bridge multiple domains. | Use embeddings to capture the meaning of texts, images, and other media types. Find relevant chunks by comparing the embedding of the chunk to that of the query. | | [examples/07_semantic_indexing](examples/07_semantic_indexing) |
| 8 | Indexing at Scale | Dealing with outdated or contradictory information in your knowledge base. | Using metadata, query filtering, and result reranking. | | [examples/08_indexing_at_scale](examples/08_indexing_at_scale) |
| 9 | Index-aware Retrieval | Comparing questions to chunks is problematic because the question itself will not appear in the knowledge base, may use synonyms or jargon, or may require holistic interpretation. | Hypothetical answers, query expansion, hybrid search, GraphRAG | | [examples/09_index_aware_retrieval](examples/09_index_aware_retrieval) |
| 10 | Node Postprocessing | Irrelevant content, ambiguous entities, generic answers. | Reranking offer the ability to bring in a lot of other neat ideas: hybrid search, query expansion, filtering, contextual compression, disambiguation, personalization | | [examples/10_node_postprocessing](examples/10_node_postprocessing) |
| 11 | Trustworthy Generation | How to retain users’ trust given that there is no way to completely avoid errors. | Out-of-domain detection, citations, guardrails, human feedback, corrective RAG, UX design can all help. | | [examples/11_trustworthy_generation](examples/11_trustworthy_generation) |
| 12 | Deep Search | RAG systems are less effective for complex information retrieval tasks because of context window constraints, query ambiguity, information verification, shallow reasoning, and multi-hop query challenges. | Iterative process of searching, reading, and reasoning to provide comprehensive answers to complex queries. | | [examples/12_deep_search](examples/12_deep_search) |

</details>

<details>
<summary>Chapter 5: Extending Model Capability (Patterns 13-16) </summary>
  
| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example |
| -------------: | :----------- | :------ | :------- | :-------------- | :----------- |
| 13 | Chain of Thought (CoT) | Foundational models often struggle with multi-step reasoning tasks, leading to incorrect or fabricated answers. | CoT prompts the model to break down complex problems into intermediate reasoning steps before providing the final answer. | Complex mathematical problems, logical deductions, and sequential reasoning tasks where step-by-step thinking is required. | [examples/13_chain_of_thought](examples/13_chain_of_thought) |
| 14 | Tree of Thoughts (ToT) | Many strategic or logical tasks cannot be solved by a single linear reasoning path, requiring exploration of multiple alternatives. | ToT treats problem-solving as a tree search, generating multiple reasoning paths, evaluating them, and backtracking as needed | Complex tasks involving strategic thinking, planning, or creative writing that require exploring multiple solution paths. | [examples/14_tree_of_thoughts](examples/14_tree_of_thoughts) |
| 15 | Adapter Tuning | Fully fine-tuning large foundational models for specialized tasks is computationally expensive and requires significant data.nt. | Adapter Tuning trains small add-on neural network layers, leaving the original model weights frozen, making it efficient for specialized adaptation. | Adapting models for specific tasks like classification, summarization, or specialized chatbots with a small (100-10k) dataset of examples. | [examples/15_adapter_tuning](examples/15_adapter_tuning) |
| 16 | Evol-Instruct | Creating high-quality datasets for instruction tuning models on new and complex enterprise tasks is difficult and time-consuming. | Evol-Instruct efficiently generates instruction-tuning datasets by evolving instructions through multiple iterations of LLM-generated tasks and answers. | Teaching models new, domain-specific tasks that are not covered by their pre-training data, particularly in enterprise settings. | [examples/16_evol_instruct](examples/16_evol_instruct) |

</details>

<details>
<summary>Chapter 6: Increasing Reliability (Patterns 17-20) </summary>

| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example |
| -------------: | :----------- | :------ | :------- | :-------------- | :----------- |  
| 17 | LLM-as-Judge | Evaluation of GenAI capabilities is hard because the tasks that GenAI performs are open-ended. | Provide detailed, multi-dimensional feedback that can be used to compare models, track improvements, and guide further development. | Evaluation is core to many of the other patterns and to building AI applications effectively. | [examples/17_llm_as_judge](examples/17_llm_as_judge) |
| 18 | Reflection | How to get the LLM to correct an earlier response in response to feedback or criticism. | The feedback is used to modify the prompt that is sent to the LLM a second time. | Reliable performance in most complex tasks where the approach can not be predetermined. | [examples/18_reflection](examples/18_reflection) |
| 19 | Dependency Injection | Need to independently develop and test each component of an LLM chain. | When you build chains of LLM calls, build them such that it is easy to inject a mock implementation to replace any step of the chain. | In any situation where you chain LLM calls or use external tools. | [examples/19_dependency_injection](examples/19_dependency_injection) |
| 20 | Prompt Optimization | Need to easily update prompts when dependencies change to maintain level of performance | Systematically set the prompts used in a GenAI pipeline by optimizing them on a dataset of examples | In any situation where you have to reduce the maintenance overhead associated with LLM version changes (and other dependencies). | [examples/20_prompt_optimiation](examples/20_prompt_optimization) |

</details>

<details>
<summary>Chapter 7: Enabling Action (Patterns 21-23) </summary>

| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example |
| -------------: | :----------- | :------ | :------- | :-------------- | :----------- |  
| 21 | Tool Calling | How can you bridge the LLM and a software API so that the LLM is able to invoke the API and get the job done? | The LLM emits special tokens when it determines that a function needs to be called and also emits the parameters to pass to that function. A client-side postprocessor invokes the function with those parameters, and sends the results back to the LLM. The LLM incorporates the function results in its response. | Whenever you want the LLM to not just state the steps needed, but to execute those steps. Also allows you to incorporate up-to-date knowledge from real-time sources, connect to transactional enterprise systems, perform calculations, and use optimization solvers. | [examples/21_tool_calling](examples/21_tool_calling) |
| 22 | Code Execution | You have a software system that can do the task, but invoking it involves a DSL. | LLMs generate code that is then executed by an external system. | Creating graphs, annotating images, updating databases. | [examples/22_code_execution](examples/22_code_execution) |
| 23 | Multi-agent Collaboration | Handle multi-step tasks that require different tools, maintain content over extended interactions, evaluate situations and take appropriate actions without human intervention, and adapt to user preferences. | Multi-agent architectures allow you to solve real-world problems using specialized single-purpose agents and organizing them in ways that mimic human organizational structures. | Complex reasoning, multi-step problem solving, collaborative content creation, adversarial verification, specialized domain integration, self-improving systems | [examples/23_multi_agent](examples/23_multi_agent) |
    
</details>

<details>
<summary>Chapters 8: Meeting Constraints (Patterns 24-28) </summary>

| Pattern Number | Pattern Name               | Problem                                                                                                                   | Solution                                                                                                                                                                                                                                                                     | Usage Scenarios                                                                                                                           | Code Example                                                             |
| -------------: |:---------------------------|:--------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------|
| 24 | Small Language Model (SLM) | The foundational model you are using is introducing too much latency or cost.                                             | Use a small foundational model to fit within cost and latency constraints without compromising unduly on quality by employing quantization (reduce precision of model parameters), distillation (narrow knowledge scope), or speculative coding (backstop with larger model) | Narrow-scoped knowledge applications, cost reduction, edge device deployment, faster inference requirements, GPU-constrained environments | [examples/24_small_language_model](examples/24_small_language_model)     |
| 25 | Prompt Caching             | User requests follow patterns with repeated queries. Recomputing the same responses wastes resources and increases costs. | Reuse previously generated responses (in the case of client-side caching) and/or model internal states (in the case of server-side caching) for the same or similar prompts. The similarity can be based on prompt meaning (semantic cache) or overlap (prefix caching).     | Applications with repeated queries, cost optimization, interactive applications requiring fast responses, multi-tenant systems            | [examples/25_prompt_caching](examples/25_prompt_caching)                 |
| 26 | Inference Optimization     | Self-hosting LLMs brings with it GPU constraints and hardware utilization challenges. Real-time applications need faster response times. | Improves the efficiency of model inference by employing continuous batching (requests are pulled from a queue and slotted into GPU cores as soon as they become available), speculative decoding (efficiently compute the next set of tokens whenever the smaller model is able to do so, backstopping this with a large model), and/or prompt compression (preprocess prompts to make them shorter). | Self-hosted LLM deployments, real-time applications, GPU memory-constrained environments, high-throughput serving scenarios               | [examples/26_inference_optimization](examples/26_inference_optimization) |
| 27 | Degradation Testing        |  Need metrics to help identify when service quality degrades and the constraint under which the application is bounded. | A set of core metrics — Time-to-First-Token (TTFT), End-to-End Request Latency (EERL), Tokens per Second (TPS) — and a variety of scalability and resilience metrics can help identify degradation of service quality; targeted interventions can help improve specific metrics. | Pre-production testing, performance validation, bottleneck identification, capacity planning, ongoing monitoring and optimization.        | [examples/27_degradation_testing](examples/27_degradation_testing)       |
| 28 | Long-Term Memory | LLM applications need to simulate memory of past interactions by prepending relevant history to each prompt, but this approach can become costly and inefficient with long conversations due to context window limitations. | LLM applications use various types of memory – working, episodic, procedural, and semantic – to maintain context, recall past interactions, personalize responses, and retain key facts, respectively. | Chatbots, multi-step workflows, personalization, processing large documents | [examples/28_long_term_memory](examples/28_long_term_memory)             |

</details>
<details>
<summary>Chapters 9: Setting Safeguards (Patterns 29-32) </summary>

| Pattern Number | Pattern Name | Problem | Solution | Usage Scenarios | Code Example                                                       |
|---------------:| :----------- | :------ | :------- | :-------------- |:-------------------------------------------------------------------|
|             29 | Template Generation | The risk of sending content without human review is very high, but human review will not scale to the volume of communications. | Pregenerate templates that are reviewed beforehand. Inference time requires only deterministic string replacement, and is therefore safe to directly send to consumers. | Personalized communications in business to consumer settings. | [examples/29_template_generation](examples/29_template_generation) |
|             30 | Assembled Reformat | Content needs to be presented in an appealing way, but the risk posed by dynamically generated content is too high. | Reduce the risk of inaccurate or hallucinated content by separating out the task of content creation into two low-risk steps — first, assembling data in low-risk ways and second, formatting the content based on that data. | Situations where accurate content needs to be presented in appealing ways, such as in product catalogs. | [examples/30_assembled_reformat](examples/30_assembled_reformat)   |
|             31 | Self-Check | Identify potential hallucinations cost-effectively | Use token probabilities to detect hallucination in LLM responses | In any situation where factual (as opposed to creative) responses are needed. | [examples/31_self_check](examples/31_self_check)                   |
|             32 | Guardails |  Require safeguards for security, data privacy, content moderation, hallucination, and alignment to ensure that AI applications operate within ethical, legal, and functional parameters. | Wrap the LLM calls with a layer of code that preprocesses the information going into the model and/or post-processes the output of the model. Knowledge retrieval and tool use will also need to be protected. | Anytime your application could be subject to attacks by malicious adversaries. | [examples/32_guardrails](examples/32_guardrails)                   |

</details>

## Want to be cited in future versions of the book?
* If you have implemented any of the patterns in the book in production, submit a PR to update the USAGE.md in the folder corresponding to the pattern.
See [examples/15_adapter_tuning/USAGE.md](examples/15_adapter_tuning/USAGE.md) for an example.

## Further reading
The GenAI Design Patterns book is a companion book to the O'Reilly book [Machine Learning Design Patterns](https://www.amazon.com/Machine-Learning-Design-Patterns-Preparation/dp/1098115783).
