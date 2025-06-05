# Topic 3. Context

**The course is under construction, with new materials appearing regularly.**

## Contents

* **3.1. The concept of RAG** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic3/3.1_the_concept_of_rag.ipynb)

  Check why adding context is a often crucial thing to do in LLM-powered applications. Explore Retrieval Augmented Generation based on web search and examine a Deep Reseach demo.

* **3.2. Database search and vector stores** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic3/3.2_database_search_and_vector_stores.ipynb)
  
  Learn how to augment LLM generation with context from relational and vector databases

  Additional reading: [**Text encoders**](https://nebius-academy.github.io/knowledge-base/text-encoders/)

  Additional reading: [**Databases in AI development: a practical guide**](https://nebius-academy.github.io/knowledge-base/databases-a-practical-guide/)

* **3.3. Advanced RAG components** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic3/3.3_advanced_rag_components.ipynb)

  Implement two types of two-stage RAG pipelines - one with a reranking step after retrieval and another compining vector and graph databases. Explore Hierarchical Navigable Small World - the nearest neighbour search algorithm used in most of today's vector databases.

* **3.4. Problems with RAG and how to solve them**

  [**Hallucinations in Retrieval Augmented LMs**](https://nebius-academy.github.io/knowledge-base/hallucinations-in-ralms/)

  [**Advanced RAG issues**](https://nebius-academy.github.io/knowledge-base/advanced-rag-issues/)

  **RAPTOR demo** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic3/3.4_RAPTOR_demo.ipynb)


## Project part: deploying an RAG service

The project materials are in the [rag_service](https://github.com/Nebius-Academy/LLMOps-Essentials/tree/rag_service) branch of the project repo. For deployment guide, see the usual [Deployment manual](https://github.com/Nebius-Academy/LLMOps-Essentials/blob/main/DEPLOYMENT_MANUAL.md).

The following video will help you to understand how the code is structured and how the service works:

1. [Creating a RAG gradio service](https://youtu.be/ep-IOGHnrqg)

**Tasks for you:** see [topic_3_project_task.md](https://github.com/Nebius-Academy/LLM-Engineering-Essentials/new/main/topic3/topic_3_project_task.md)
