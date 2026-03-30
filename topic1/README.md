# Topic 1. LLM API Basics

**The course is under construction, with new materials appearing regularly.**

**Subscribe for updates and make sure you don’t miss anything: [Stay updated](https://academy.nebius.com/llm-engineering-essentials/update/)**

## Contents

* **1.1. Intro to LLM APIs** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.1_intro_to_llm_apis.ipynb)

  Experiment with text-to-image and image-to-text generation, send API requests, and analyze responses using OpenAI API and [Nebius Token Factory](https://tokenfactory.nebius.com/), which provides open-source models like Llama, Mistral, DeepSeek, and Qwen. 
  This section will ensure you can make API calls, handle responses, and integrate LLMs into applications.

* **1.2. Tokenization** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.2_tokenization.ipynb)

  For LLMs, text is composed of tokens, which can be words, subwords, individual characters, or even byte sequences.
  In this chapter, you'll learn what tokens are and how tokenization affects the way LLMs process text.

* **1.3. Basic prompting guidelines** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.3_basic_prompting_guidelines.ipynb)

  In this chapter, we'll share some basic practical hints for prompting LLMs.

* **1.4. What can possibly go wrong with an LLM** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.4_what_can_possibly_go_wrong_with_an_llm.ipynb)

  Learn about some of the blights that plague LLM users: hallucinations, bias, mode collapsing, and jailbreaks.

* **1.5. Choosing an LLM** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.5_how_to_choose_an_llm.ipynb)

  Learn about various cosiderations that you can use to choose LLMs for your projects, including size tiers, benchmarks, reasoning capabilities, etc.

* **1.6. LLM Inference Parameters** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.6_llm_inference_parameters.ipynb)

  Understand token probabilities, uncertainty, and the LLM generation process. 
  Learn how inference parameters influence randomness in responses and how to balance reproducibility and creativity.


* **1.7. Creating an LLM-powered Character** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.7_creating_an_llm-powered_character.ipynb)

  Build a chatbot based on a fantasy character and explore advanced features like giving it a scratchpad — a space for its own "thoughts."

## Project part: deploying an LLM-powered service

The project materials are in the [LLMOps Essentials repo](https://github.com/Nebius-Academy/LLMOps-Essentials). For now, you only need the `main` branch which deploys the simple API-based service (we'll proceed to RAG and open-source LLM deployment in the later topics).

The following videos will help you to understand how the repo and the service that it implements work:

1. [Designing a chat service using LLMs: A simple way](https://youtu.be/pOXA7ZuB-98)
2. [Designing a chat service using LLMs: Running a service](https://youtu.be/Ry0nXts6B0o)
3. [Designing a chat service using LLMs: An advanced way](https://youtu.be/N6okNbcGjY8)
4. [Running the chat service](https://youtu.be/pPFWefazyAQ)
5. (Optional) [Advanced topics: Kubernetes deployment](https://youtu.be/uVEP4doSGQ4) - we prepared this one for those who already know about Kubernetes and want to understand how to deploy the chat service in it

Please check the [deployment manual](https://github.com/Nebius-Academy/LLMOps-Essentials/blob/main/DEPLOYMENT_MANUAL.md) to learn how to deploy the service on your VM.

**A task for you**. To start wrapping your mind around the repo:

1. Update the range of available LLMs. Add something new and fashionable
2. Locate where chat history is stored and find out how to get it by querying the service from colab
3. Try adding scratchpad support

## Additional reading: 

* [**LLM training overview**](https://nebius-academy.github.io/knowledge-base/llm-training-overview/)
