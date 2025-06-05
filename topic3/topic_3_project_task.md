# Topic 3 project task

1. First, you will need a database to experiment with. The RAG project uses LanceDB, as described in the `config` file. We suggest you to work with the documentation of the `transformers` library, but you can choose a different corpus. Just make sure that your chunking is suitable for the chosen document type.
    
    Download the [markdown_to_text.py](https://drive.google.com/file/d/1Q6wtX9Ldu7P1BadGROW-kxeCB66fr8Y0/view) file to your VM.
    
    Clone `https://github.com/huggingface/transformers` to your VM and run `markdown_to_text.py` script to extract raw text from `transformers/docs/source/en/`. This is the script you need to run:
    
    `python prep_scripts/markdown_to_text.py --input-dir transformers/docs/source/en/ --output-dir docs`
    
    This will be your knowledge base, you don't need it to be a part of your repository.
    
    Use the `add_to_rag_db` endpoint to load every text into the database. Make several RAG API calls to check that the pipeline is working.
    
2. Try using a reranker. 
    
    The reranker is defined in the
    
    ```
    rerank:
        container_name: rerank_service
        image: [ghcr.io/huggingface/text-embeddings-inference:cpu-1.5](http://ghcr.io/huggingface/text-embeddings-inference:cpu-1.5)
        volumes:
            - ./data:/data
        command: ["--model-id", "mixedbread-ai/mxbai-rerank-base-v1"]
    ```
    
    part of `rag_service/docker-compose.yaml`. As you can see, the defalt reranker is [mixedbread-ai/mxbai-rerank-base-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1). However, by default it is not used, which is established by the parameter `use_reranker: bool = False` of the constructor `class RAGRequest(BaseModel)` in the file `rag_service/src/main.py`. You need to make it `True` to switch on reranking.
    
    Also, you can change the parameters `top_k_retrieve`, `top_k_rank` , if necessary.
    
    - Compare how the retrieved context changes after adding a reranker. For that, try at least 10 different prompts. If you're generous with your time and API budget, try LLM-as-a-Judge.
    - Measure the responce time for RAG with and without a reranker for at least 10 different prompts and for at least 3 different values of `top_k_rank`.
    - Analyze pros and cons of using a reranker, based on relevance of the top-k documents and the response time.
    
3. Try at least three different LLMs and compare the results.
    
    As in the previous task, try at least 10 different prompts for each LLM or try LLM-as-a-Judge. 
    
4. Put together a simple evaluation dataset of 20 questions (+optionally answers) - you could do it manually or generate with an LLM. Use the [LLM-as-a-Judge](https://huggingface.co/learn/cookbook/en/llm_judge) approach to quantitatively evaluate your best setup. 
    
    [Bonus] Explore the [Ragas](https://docs.ragas.io/en/stable/) docs for possible evaluation setups.

5. Try a different embedding model; for example one from the top of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and justify your choice. If things are getting slow, switch to a gpu - don’t forget to switch to a corresponding [tei container](https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images).
    
    The encoder is defined in this part
    
    ```
    embed:
        container_name: embed_service
        image: [ghcr.io/huggingface/text-embeddings-inference:cpu-1.5](http://ghcr.io/huggingface/text-embeddings-inference:cpu-1.5)
        volumes:
            - ./data:/data
        command: ["--model-id", "BAAI/bge-small-en-v1.5"]
    ```
    
    of `rag_service/docker-compose.yaml`. As you can see, the defalt encoder is [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
    
    Analyze the relevance of the retrieved documents and how the embedding time differs between models.

6. [Bonus] Adjust the RAG setup to work smoothly for a multi-turn conversation.
