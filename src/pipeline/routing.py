"""Model routing cheap-first com fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium)."""
    
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.5-flash-lite")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    query_lower = query.lower()
    
    palavras_complexas = [
        "analise", "compare", "diferença", "explique", 
        "resuma", "tendência", "perfil", "pegadinha", 
        "armadilha", "jurisprudência", "profundo"
    ]
    
    if any(palavra in query_lower for palavra in palavras_complexas):
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="A pergunta exige capacidade de análise, comparação ou síntese profunda."
        )
    
    if len(query) > 100:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="A pergunta é longa e possui um contexto extenso."
        )
        
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="A pergunta é direta e focada, ideal para o modelo rápido."
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()