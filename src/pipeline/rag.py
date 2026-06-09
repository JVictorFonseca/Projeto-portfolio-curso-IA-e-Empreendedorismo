"""Pipeline RAG usando ChromaDB e a API do Gemini via OpenAI Client."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from src.observability.trace import trace
from src.pipeline.tools import TOOL_REGISTRY, TOOLS


def _make_client() -> tuple[OpenAI, str]:
    """Cria cliente OpenAI-compatible para o Gemini."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ), "https://generativelanguage.googleapis.com/v1beta/openai/"
    raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")


class RAGPipeline:
    def __init__(self, corpus_dir: str, db_dir: str = ".chromadb"):
        self.corpus_dir = Path(corpus_dir)
        self.db_dir = Path(db_dir)
        
        self.client, _ = _make_client()
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Usando a função default do Chroma para embeddings locais leves
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="concursos_corpus",
            embedding_function=self.emb_fn
        )
        
        # Ingestão inicial se estiver vazio
        if self.collection.count() == 0:
            self.ingest_and_index()

    # ============================================================================
    # TODO 1 — Ingestao (Leitura, Chunking e Indexacao)
    # ============================================================================
    def ingest_and_index(self) -> None:
        """Lê os PDFs do diretório corpus, divide em pedaços e indexa no ChromaDB."""
        if not self.corpus_dir.exists():
            return

        documents = []
        metadatas = []
        ids = []
        counter = 0

        for pdf_path in self.corpus_dir.glob("*.pdf"):
            try:
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    
                    # Chunking simples por tamanho (fatias de ~800 caracteres)
                    chunk_size = 800
                    for i in range(0, len(text), chunk_size - 100):
                        chunk = text[i:i + chunk_size].strip()
                        if len(chunk) > 50:
                            documents.append(chunk)
                            metadatas.append({
                                "source": pdf_path.name,
                                "page": page_num
                            })
                            ids.append(f"id_{pdf_path.name}_{counter}")
                            counter += 1
            except Exception as e:
                print(f"Erro ao ler {pdf_path.name}: {e}")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    # ============================================================================
    # TODO 2 — Recuperacao (Retrieve)
    # ============================================================================
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Busca os trechos de manuais/provas mais relevantes para a dúvida."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                retrieved.append({
                    "text": doc,
                    "source": meta.get("source", "Desconhecido"),
                    "page": meta.get("page", 0)
                })
        return retrieved

    # ============================================================================
    # TODO 3 — Geracao Ancorada (Generate)
    # ============================================================================
    def answer(self, query: str) -> dict[str, Any]:
        """Orquestra o RAG: recupera contexto, chama ferramentas e responde via LLM."""
        # 1. Recupera o contexto dos PDFs
        context_blocks = self.retrieve(query)
        context_text = "\n\n".join([f"--- Fonte: {b['source']} (p. {b['page']}) ---\n{b['text']}" for b in context_blocks])
        
        # 2. Configura as mensagens do sistema
        system_prompt = (
            "Você é um Mentor de Concursos Públicos experiente. Responda à dúvida do candidato "
            "com base estrita nos documentos fornecidos no contexto abaixo. Se usar informações do contexto, "
            "cite a fonte e a página no corpo do texto. Caso o usuário pergunte sobre o estilo de uma banca específica, "
            "use a ferramenta de domínio apropriada disponível."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto extraído das provas/manuais:\n{context_text}\n\nPergunta do Candidato: {query}"}
        ]
        
        model_name = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
        
        # Primeira chamada ao LLM para verificar se ele quer acionar a Tool de Bancas (TODO 4)
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS if TOOLS else None
        )
        
        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        
        # Executa a ferramenta caso a IA decida usá-la
        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                if func_name in TOOL_REGISTRY:
                    import json
                    args = json.loads(tool_call.function.arguments)
                    tool_output = TOOL_REGISTRY[func_name](**args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_output
                    })
            
            # Segunda chamada para consolidar o relatório final
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            final_answer = response.choices[0].message.content
        else:
            final_answer = response_message.content

        # Estrutura o retorno com as referências das páginas
        sources = [(b["source"], b["page"]) for b in context_blocks]
        return {
            "answer": final_answer,
            "sources": list(set(sources))  # Remove duplicatas de páginas idênticas
        }


def build_rag_pipeline(corpus_dir: str) -> RAGPipeline:
    return RAGPipeline(corpus_dir=corpus_dir)