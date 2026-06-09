"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional)."""

    def get_stats(self) -> dict:
        """Retorna o tamanho atual do cache."""
        return {"size": len(self._answers)}
    
    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        # Inicializa cliente para embeddings (mesmo provider do RAG)
        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
        else:
            self._client = OpenAI()
            self._embed_model = "text-embedding-3-small"

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        # 1. Obter o embedding da nova query
        q_emb = self._embed(query)
        q_norm = np.linalg.norm(q_emb)
        
        best_sim = -1.0
        best_idx = -1
        
        # 2. Calcular similaridade cosseno contra todas as queries anteriores
        for i, em in enumerate(self._embeddings):
            em_norm = np.linalg.norm(em)
            cos_sim = np.dot(q_emb, em) / (q_norm * em_norm)
            
            # 3. Guardar o índice do mais similar
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_idx = i
                
        # 4. Se a similaridade for maior ou igual ao threshold (0.93 por defeito), devolvemos o cache
        if best_sim >= self.threshold:
            return self._answers[best_idx]
            
        return None