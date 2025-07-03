# Gen AI Document Q&A Chatbot using Amazon Bedrock + LangChain + OpenSearch

import boto3
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms import BedrockLLM
from langchain.chains import RetrievalQA

# ----------- Step 1: Upload PDF to S3 -----------
s3 = boto3.client("s3")
bucket_name = "your-s3-bucket-name"
file_path = "your-local-pdf.pdf"
s3_key = "documents/your-local-pdf.pdf"
s3.upload_file(file_path, bucket_name, s3_key)
print("PDF uploaded to S3")

# ----------- Step 2: Load & Chunk Document -----------
loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# ----------- Step 3: Generate Embeddings -----------
embedding = BedrockEmbeddings(
    region_name="us-east-1",
    credentials_profile_name="default"
)

# ----------- Step 4: Store Embeddings in OpenSearch -----------
opensearch_vector_store = OpenSearchVectorSearch(
    opensearch_url="https://your-opensearch-domain.amazonaws.com",
    index_name="genai-docs",
    embedding_function=embedding
)

opensearch_vector_store.add_documents(chunks)
print("Chunks embedded and stored in OpenSearch")

# ----------- Step 5: Build Q&A Chain with Bedrock -----------
llm = BedrockLLM(
    region_name="us-east-1",
    credentials_profile_name="default",
    model_id="anthropic.claude-v2"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=opensearch_vector_store.as_retriever()
)

# ----------- Step 6: Ask a Question -----------
query = "What is the refund policy mentioned in the document?"
answer = qa_chain.run(query)
print("Answer:", answer)


# Example Lambda handler structure:
def lambda_handler(event, context):
    query = event['queryStringParameters']['q']
    answer = qa_chain.run(query)
    return {
        'statusCode': 200,
        'body': json.dumps({'answer': answer})
    }
