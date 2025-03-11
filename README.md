# Research Paper Retrieval System

## Overview
This project is a Retrieval-Augmented Generation (RAG) system designed to assist users in retrieving and generating responses based on research papers. The system utilizes various components to ensure efficient document retrieval, context-aware generation, and user-friendly interaction.

## Components

### Data Source
- Dataset: [Research Papers Short](https://huggingface.co/datasets/pt-sk/research_papers_short)
- The dataset consists of research papers that have been processed into smaller chunks.
- Chunking parameters:
  - Tokens: 256
  - Dimension: 384
- Additionally, chunks that complete a paper but do not meet the size requirement are stored separately.

### Language Model (LLM)
- Model: **Cohere command-r-plus**
- Documentation: [Cohere API Docs](https://docs.cohere.com/v2/docs/command-r-plus)
- Used for generating responses based on retrieved documents or directly answering queries without context.

### Retriever
- The system supports multiple retrieval methods:
  - **BM25** from the `rank_bm25` library
  - **Dense retrieval** using a vector database
- Users can choose to:
  - Use BM25 retrieval
  - Use dense retrieval
  - Generate responses without retrieval (no document context used)

### Reranker
- No reranker is implemented in this version.

### Citations
- References to retrieved documents are displayed in **parentheses** within the generated text.
- Citations point to the articles used in the response.

### User Interface (UI)
- Built using **Gradio** to provide an easy-to-use web interface.
- API key is pre-configured, allowing users to access the system without additional setup.

### Additional Feature
- Users can request a **random research paper** from the dataset to test the system’s functionality quickly.
