"""Streamlit UI — entrada principal do app com Upload de PDFs dinâmico."""

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

# Garante a existência do diretório do Corpus
CORPUS_DIR = _ROOT / "data" / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Streamlit UI
st.set_page_config(page_title="Mentor Concurso", page_icon=":shield:", layout="centered")

st.title("🛡️ Mentor Concurso: Analisador de Bancas")
st.caption("Domine o estilo de cobrança das bancas organizadoras com análise baseada em evidências.")

# Inicializacao lazy de pipeline + caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(CORPUS_DIR))


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


# Sidebar — upload de arquivos, metricas e debug
with st.sidebar:
    st.header("Alimentar o Mentor")
    
    # Upload dinâmico via interface gráfica
    uploaded_files = st.file_uploader(
        "Suba novos PDFs de bancas/provas:", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        novos_arquivos = False
        for uploaded_file in uploaded_files:
            file_path = CORPUS_DIR / uploaded_file.name
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Salvo: {uploaded_file.name}")
                novos_arquivos = True
        
        if novos_arquivos:
            with st.spinner("Fatiando e indexando novos documentos..."):
                if hasattr(pipeline, "collection") and pipeline.collection.count() > 0:
                    try:
                        pipeline.collection.delete(where={})
                    except Exception:
                        pass
                pipeline.ingest_and_index()
            st.rerun()

    st.header("Métricas do Sistema")
    st.metric("Chunks indexados", pipeline.collection.count())
    
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
    # Desacoplado do context manager de trace para evitar colisões em caso de erro da API
    cached = exact_cache.get(query)
    if cached:
        st.success("Cache hit (exact)")
        st.write(cached)
        st.stop()

    try:
        cached = semantic_cache.get(query)
    except NotImplementedError:
        cached = None

    if cached:
        st.success("Cache hit (semantic)")
        st.write(cached)
        st.stop()

    try:
        decision = classify_complexity(query)
        st.info(f"Routing Inteligente: Complexidade '{decision.complexity}' -> Usando {decision.model}")
    except NotImplementedError:
        st.warning("Routing não configurado.")

    try:
        result = pipeline.answer(query)
        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("Fontes citadas no corpus"):
                for source, page in result["sources"]:
                    st.write(f"- `{source}:p{page}`")
                    
        exact_cache.put(query, result["answer"])
        semantic_cache.put(query, result["answer"])
        
    except Exception as e:
        st.error(f"Erro ao processar resposta da pipeline: {e}")
        st.info("Verifique se sua GEMINI_API_KEY no arquivo .env está correta e completa.")

st.divider()