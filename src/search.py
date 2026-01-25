from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question, pg_vector, llm):
    """
    Realiza busca semântica e gera resposta usando LLM.
    
    Args:
        question (str): Pergunta do usuário
        pg_vector (PGVector): Instância do banco vetorial
        llm (ChatGoogleGenerativeAI): Instância do LLM
        
    Returns:
        str: Resposta gerada
    """
    # Buscar documentos relevantes
    results = pg_vector.similarity_search_with_score(question, k=10)
    
    # Concatenar contexto
    contexto = "\n\n".join([doc.page_content for doc, score in results])
    
    # Criar prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(contexto=contexto, pergunta=question)
    
    # Gerar resposta
    response = llm.invoke(formatted_prompt)
    
    return response.content