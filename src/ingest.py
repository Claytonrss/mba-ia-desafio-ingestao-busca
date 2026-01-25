import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

from dotenv import load_dotenv

load_dotenv()

for key in ("PDF_PATH", "GOOGLE_API_KEY", "DATABASE_URL", "GOOGLE_EMBEDDING_MODEL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(key):
        raise ValueError(f"Environment variable {key} is not set")

PDF_PATH = os.getenv("PDF_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

def enrich_chunks(chunks):
    enriched_chunks = []
    for chunk in chunks:
        enriched_chunk = Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)}
        )
        enriched_chunks.append(enriched_chunk)
    return enriched_chunks

def ingest_pdf(max_chunks=None):
    docs = PyPDFLoader(str(PDF_PATH)).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False).split_documents(docs)
    
    if not splits:
        raise ValueError("No splits found")
    
    if max_chunks:
        splits = splits[:max_chunks]
    
    enriched_chunks = enrich_chunks(splits)
    
    ids = [f"doc-{i}" for i in range(len(enriched_chunks))]
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    
    try:
        pg_vector = PGVector(
            embeddings=embeddings,
            collection_name=PG_VECTOR_COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True
        )
        
        pg_vector.add_documents(enriched_chunks, ids=ids)
    except Exception as e:
        print(f"Erro ao conectar ao banco de dados: {e}")
        print(f"DATABASE_URL (primeiros 50 chars): {DATABASE_URL[:50]}...")
        raise
    
    return enriched_chunks

if __name__ == "__main__":
    MAX_CHUNKS = os.getenv("MAX_CHUNKS")
    max_chunks = int(MAX_CHUNKS) if MAX_CHUNKS else None
    
    chunks = ingest_pdf(max_chunks=max_chunks)
    print(f"✅ Processados {len(chunks)} chunks com sucesso!")
    if chunks:
        print("\n📝 Primeiro chunk:")
        print(f"Conteúdo (primeiros 200 chars): {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    print("-" * 100)