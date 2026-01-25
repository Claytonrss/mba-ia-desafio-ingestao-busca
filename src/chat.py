import os
import sys
from dotenv import load_dotenv
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from search import search_prompt

load_dotenv()

def main():
    # Load configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL")
    GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
    GOOGLE_GENERATIVE_MODEL = os.getenv("GOOGLE_GENERATIVE_MODEL")
    PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
    
    if not GOOGLE_API_KEY:
        print("Erro: GOOGLE_API_KEY não definida.")
        sys.exit(1)

    print("Inicializando sistema de chat...")

    try:
        # Initialize components
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

        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_GENERATIVE_MODEL,
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e:
        print(f"Erro na inicialização: {e}")
        sys.exit(1)
    
    print("\nSistema pronto! Digite 'sair' para encerrar.\n")
    print("-" * 50)

    while True:
        try:
            question = input("\nFaça sua pergunta: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ["sair", "exit", "quit"]:
                print("Encerrando...")
                break
                
            print("\nBuscando resposta...\n")
            
            response = search_prompt(question, pg_vector, llm)
            
            print("-" * 50)
            print(f"PERGUNTA: {question}")
            print(f"RESPOSTA: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}")

if __name__ == "__main__":
    main()
