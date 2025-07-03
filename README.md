## 📦Problem Statement
Most enterprises have large unstructured documents (PDFs, Word files, etc.). Searching through them is inefficient. The goal was to create a chatbot that could understand, search, and answer user questions from these documents in real-time.


# 📄 Document Q&A Chatbot using Amazon Bedrock + LangChain + OpenSearch

A Generative AI-powered chatbot that allows users to ask questions from internal company documents (PDFs) using Retrieval-Augmented Generation (RAG) and Amazon Bedrock's large language models.

## 🔧 Tech Stack

- **Language**: Python
- **Cloud**: AWS
  - Amazon S3 – File storage
  - Amazon Bedrock – Claude / Titan LLMs
  - Amazon OpenSearch – Vector DB for RAG
  - AWS Lambda – Backend API
  - API Gateway – Public access
  - AWS Cognito – Authentication
- **Libraries**: LangChain, Boto3, Bedrock SDK

---

## 📐 Architecture

```plaintext
1. Upload PDF ➜ S3
2. Extract & Chunk ➜ LangChain
3. Generate Embeddings ➜ Titan or HuggingFace
4. Store Vectors ➜ OpenSearch
5. Query ➜ LangChain → Retrieve + Prompt Claude
6. Respond ➜ Lambda → API Gateway → Frontend
