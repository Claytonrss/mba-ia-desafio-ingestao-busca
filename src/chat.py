import os
from langchain_community.document_loaders import PyPDFLoader
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
    
query = "Qual é o faturamento da Cobalto Gás Indústria?"

embeddings = GoogleGenerativeAIEmbeddings(
    model=GOOGLE_EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY
)

pg_vector = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True
    )

results = pg_vector.similarity_search_with_score(query, k=10)

for i, (doc, score) in enumerate(results, start=1):
    print("="*50)
    print(f"Resultado {i} (score: {score:.2f}):")
    print("="*50)
    
    print("\nTexto:\n")
    print(doc.page_content.strip())
    
    print("\nMetadados:\n")
    for k, v in doc.metadata.items():
        print(f"{k}: {v}")
    
# from search import search_prompt

# def main():
#     chain = search_prompt()

#     if not chain:
#         print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
#         return
    
#     pass

# if __name__ == "__main__":
#     main()
