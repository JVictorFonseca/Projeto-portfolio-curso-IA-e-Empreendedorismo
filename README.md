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
```

---

## 🧠 3. Decisões de Design

* **Abordagem Híbrida (RAG + Ferramenta Estática):** Em vez de confiar apenas na busca semântica para conceitos altamente estruturados (como o fato de uma questão errada anular uma certa no CEBRASPE), combinamos o RAG para os PDFs com uma `Tool` dedicada. Isso garante 100% de precisão em regras cruciais de negócio.
* **Chunking Baseado em Páginas e Caracteres:** A estruturação dos chunks mantém metadados detalhados de `source` (nome do PDF) e `page` (página exata). Isso permite a "Rastreabilidade Baseada em Evidências", exibindo ao usuário de onde aquela regra foi extraída.
* **Cache Semântico com Similaridade de Cosseno:** Implementado diretamente sobre a camada de embeddings. Ao fixar o limiar (*threshold*) em `0.93`, o sistema consegue interceptar variações na forma de perguntar do usuário, reduzindo o consumo de tokens repetidos a zero.

---

## ⚡ 4. Otimização de Custo e Latência

Para garantir a viabilidade econômica do projeto mantendo o desempenho de um modelo "premium", aplicamos duas estratégias fundamentais:

1. **Roteamento Dinâmico de Modelos (Routing):**
   * **Modelo Econômico (`gemini-2.5-flash-lite`):** Ativado para interações curtas, perguntas diretas ou consultas simples de baixa complexidade. Reduz drasticamente o custo por milhão de tokens.
   * **Modelo Premium (`gemini-2.5-pro`):** Invocado de forma cirúrgica apenas quando a pergunta exige capacidade analítica profunda ou quando o contexto inserido pelo usuário ultrapassa 100 caracteres.
2. **Camada Dupla de Cache:** O cache exato poupa processamento computacional imediato, enquanto o cache semântico evita chamadas repetidas à API externa do Gemini, diminuindo a latência da resposta de ~2.5 segundos para menos de 100 milissegundos nas perguntas recorrentes.

---

## 🛠️ 5. Como Executar o Projeto Localmente

1. **Instalar o Gerenciador de Pacotes (UV):**
   ```bash
   curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
   source $HOME/.local/bin/env
   ```
2. **Sincronizar as Dependências:**
   ```bash
   uv sync
   ```
3. **Configurar as Variáveis de Ambiente:**
   Crie um arquivo `.env` na raiz do projeto e adicione a sua chave da API:
   ```env
   GEMINI_API_KEY=SUA_CHAVE_AQUI
   ```
4. **Executar a Aplicação:**
   ```bash
   uv run streamlit run src/ui/streamlit_app.py
   ```

---

## 🚀 6. Guia de Uso (Como interagir com o Mentor)

A interface foi desenhada para ser intuitiva e simular a mesa de estudos de um candidato. Siga os passos abaixo para extrair o máximo do pipeline RAG:

### Passo 1: Alimentar a Base de Conhecimento (Upload)
O Mentor precisa de material base para analisar. 
1. Na barra lateral esquerda, localize a seção **"Alimentar o Mentor"**.
2. Arraste e solte arquivos PDF (como o edital do TCE/RN, provas anteriores do CEBRASPE ou manuais de critérios da banca) na área de upload.
3. O sistema fará a ingestão, dividirá o texto em fragmentos (chunks) e atualizará automaticamente o contador **"Chunks indexados"**.

### Passo 2: Consultar o Mentor
Com os PDFs carregados, utilize a caixa de texto principal no centro da tela. 
* **Dica de Ouro:** Seja específico para acionar a inteligência analítica do sistema.
* *Exemplo de pergunta:* "De acordo com os editais carregados, como o CEBRASPE penaliza o chute nas provas de conhecimentos específicos para o TCE/RN?"

### Passo 3: Auditoria de Fontes (Evidências)
O grande diferencial deste sistema é a mitigação de alucinações.
1. Após receber a resposta do Mentor, role logo abaixo do texto gerado.
2. Clique no menu expansível **"Fontes citadas no corpus"**.
3. O sistema listará exatamente qual documento PDF e qual página foram utilizados para embasar a resposta (ex: `- cespe-prova.pdf:p4`).

### Passo 4: Monitoramento e Cache
Na barra lateral, você pode acompanhar a eficiência da arquitetura em tempo real:
* **Exact cache:** Conta quantas perguntas foram respondidas instantaneamente por serem idênticas a consultas anteriores.
* **Semantic cache:** Mostra respostas recuperadas por similaridade de intenção (ex: "estilo da banca" vs. "como a banca cobra").
* Caso precise resetar a memória para um novo ciclo de estudos, basta clicar no botão **"Limpar caches"**.