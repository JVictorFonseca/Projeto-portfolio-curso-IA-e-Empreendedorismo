"""Function-calling / tool-use — registro de tools usadas pelo agente.

Reaproveita o LAB-001. Voce vai preencher 1 TODO aqui (sua tool especifica).
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ============================================================================
# TODO 4 — Sua tool especifica do dominio
# ============================================================================
# Cada projeto precisa de UMA tool customizada que faca sentido para o problema.
# Exemplos por dominio:
#   - Livro tecnico:    lookup_chapter(chapter: int) -> str
#   - Changelog:        check_compat(lib: str, version: str) -> dict
#   - Podcast:          get_timestamp(quote: str) -> str
#   - Codigo:           run_snippet(code: str) -> str  (sandboxed)
#   - Documentos legais: cite_article(law: str, article: int) -> str
#
# 1. Implemente a funcao Python real abaixo (substitua o exemplo)
# 2. Adicione o schema JSON em TOOLS abaixo
# 3. Registre em TOOL_REGISTRY
# ============================================================================


# SEU CODIGO AQUI — TODO 4
def buscar_armadilhas_banca(banca: str) -> str:
    """Retorna o histórico de armadilhas e o estilo de cobrança de uma banca."""
    banca = banca.upper().strip()
    
    base_conhecimento = {
        "CEBRASPE": "Estilo Certo/Errado. Uma errada anula uma certa. Foca muito em jurisprudência (STF/STJ) e interpretação profunda. Armadilhas comuns: trocar 'pode' por 'deve', 'prescindível' por 'imprescindível', e usar palavras restritivas como 'apenas', 'somente', 'nunca'.",
        "FGV": "Textos longos e casos concretos (situações hipotéticas). Em Português, a interpretação de texto e semântica são o foco principal, não a gramática pura. Em Direito, cobra muita literalidade misturada com casos práticos.",
        "FCC": "Conhecida como 'Fundação Copia e Cola', cobra muita literalidade da lei (lei seca). Armadilhas comuns: trocar prazos, quóruns e autoridades competentes. Em Português, foca bastante em gramática normativa e reescritura de frases.",
        "VUNESP": "Questões diretas e objetivas. Cobra muita lei seca, mas de forma menos complexa que a FGV. Cuidado com armadilhas nas questões de raciocínio lógico e matemática."
    }
    
    for chave in base_conhecimento:
        if chave in banca:
            return base_conhecimento[chave]
            
    return f"Não encontrei dados específicos de armadilhas para a banca {banca}. Analise os ficheiros PDF indexados para obter mais detalhes."


TOOLS: list[dict[str, Any]] = [
    # SEU CODIGO AQUI — TODO 4 (continuacao)
    {
        "type": "function",
        "function": {
            "name": "buscar_armadilhas_banca",
            "description": "Busca o histórico de armadilhas, pegadinhas e o estilo de cobrança de uma banca organizadora de concursos (ex: CEBRASPE, FGV, FCC). Use esta ferramenta sempre que o utilizador perguntar sobre o perfil, truques ou estilo de uma banca.",
            "parameters": {
                "type": "object",
                "properties": {
                    "banca": {
                        "type": "string", 
                        "description": "Nome da banca organizadora do concurso (ex: CEBRASPE, FGV, FCC)."
                    },
                },
                "required": ["banca"],
            },
        },
    },
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "buscar_armadilhas_banca": buscar_armadilhas_banca,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
