"""Streamlit UI — entrada principal do app. Pronta para deploy 1-click no Streamlit Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Adiciona o root do projeto no path para imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402


# ---------------------------------------------------------------- Streamlit UI
st.set_page_config(page_title="Mentor Concurso", page_icon=":shield:", layout="centered")

st.title("🛡️ Mentor Concurso: Analisador de Bancas")
st.caption("Domine o estilo de cobrança das bancas organizadoras com análise baseada em evidências.")

# Inicializacao lazy de pipeline + caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Sidebar — metricas e debug
with st.sidebar:
    st.header("Métricas do Sistema")
    st.metric("Chunks indexados", pipeline.collection.count())
    
    # Tratamento seguro para evitar quebra caso o método stats mude
    try:
        exact_size = exact_cache.get_stats()["size"]
    except AttributeError:
        exact_size = len(exact_cache._store) if hasattr(exact_cache, "_store") else 0
        
    st.metric("Exact cache", exact_size)
    st.metric("Semantic cache", semantic_cache.get_stats()["size"])

    if st.button("Limpar caches"):
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos. Recarregue a página.")


# Main — chat interface
query = st.text_input("Sua pergunta:", placeholder="Ex: Qual o perfil de pegadinhas do CEBRASPE?")

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        # 1. Exact cache
        cached = exact_cache.get(query)
        if cached:
            st.success("Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        # 2. Semantic cache
        try:
            cached = semantic_cache.get(query)
        except NotImplementedError:
            cached = None
            st.warning("Semantic cache nao implementado (TODO 5). Caindo no LLM real.")

        if cached:
            st.success("Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        # 3. Pipeline RAG + Routing
        try:
            decision = classify_complexity(query)
            st.info(f"Routing Inteligente: Complexidade '{decision.complexity}' -> Usando {decision.model}")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except NotImplementedError:
            st.warning("Routing nao implementado (TODO 6). Usando modelo default.")

        try:
            result = pipeline.answer(query)
        except NotImplementedError as e:
            st.error(f"Pipeline nao implementado: {e}")
            st.info("Implemente TODOs 1-3 em `src/pipeline/rag.py` para destravar.")
            st.stop()

        # 4. Renderiza + cacheia
        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("Fontes citadas no corpus"):
                for source, page in result["sources"]:
                    st.write(f"- `{source}:p{page}`")

        exact_cache.put(query, result["answer"])
        semantic_cache.put(query, result["answer"])
        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))

st.divider()