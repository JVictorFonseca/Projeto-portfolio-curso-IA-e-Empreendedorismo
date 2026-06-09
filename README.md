# 🛡️ Mentor Concurso: Analisador de Bancas com Arquitetura RAG

Este projeto consiste em um assistente de inteligência artificial especializado e disponível 24/7, projetado para atuar como um mentor de alto nível para concurseiros. O sistema utiliza técnicas avançadas de Engenharia de Prompt e RAG (Retrieval-Augmented Generation) para analisar manuais de bancas organizadoras, perfis de cobrança e histórico de pegadinhas, garantindo respostas ancoradas em evidências e livres de alucinações.

---

## 🎯 1. Problem Statement (Declaração do Problema)

Candidatos a concursos públicos de alto nível enfrentam grandes dificuldades para mapear o perfil de cobrança específico de bancas organizadoras (como CEBRASPE, FGV e FCC). As análises disponíveis no mercado costumam ser superficiais ou dispersas em longos vídeos e manuais em PDF complexos. Além disso, modelos de linguagem genéricos tendem a alucinar sobre os estilos de cobrança ou misturar as regras de uma banca com outra (como o sistema de penalização do CEBRASPE).

**A Solução:** Um pipeline inteligente que ingere os manuais de critérios e provas oficiais, permitindo ao estudante interagir com um mentor especialista que extrai métricas exatas, mapeia pegadinhas recorrentes e fundamenta cada orientação diretamente nas páginas dos documentos oficiais.

---

## 🏗️ 2. Arquitetura do Sistema

O sistema foi construído seguindo os padrões modernos de aplicações de IA Generativa corporativas, dividindo-se em camadas bem definidas:

```text
[ Usuário (Interface Streamlit) ]
               │
               ▼
   [ Cache Semântico (Chroma) ] ──(Hit)──> [ Retorna Resposta Salva ]
               │
             (Miss)
               ▼
[ Router Inteligente (Routing) ]
        ├── Escolha: "Simple" ──> [ Gemini 2.5 Flash Lite ] (Econômico/Rápido)
        └── Escolha: "Complex" ──> [ Gemini 2.5 Pro ] (Análise Avançada)
               │
               ▼
[ Orquestrador RAG (LangChain + Tool Registry) ]
        ├── Contexto Interno: Vector DB (Chunks de PDFs de Manuais/Provas)
        └── Tool Externa: `buscar_armadilhas_banca` (Regras de Negócio Críticas)
               │
               ▼
    [ Resposta Ancorada + Fontes ]