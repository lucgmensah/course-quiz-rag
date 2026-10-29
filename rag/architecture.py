from rag.config import Settings
from rag.ingest import EMBEDDING_MODEL


def pipeline_stages(s: Settings) -> list[dict]:
    """The real pipeline, with parameters read from the live settings."""
    return [
        {"name": "PDF", "description": "Le cours déposé par le visiteur, conservé en mémoire uniquement.",
         "params": {"taille max (Mo)": s.max_pdf_mb, "pages max": s.max_pages}},
        {"name": "Extraction", "description": "Le texte est extrait page par page avec pypdf. "
         "Les PDF scannés, protégés ou corrompus sont refusés.", "params": {}},
        {"name": "Découpage", "description": "Chaque page est découpée en passages qui se chevauchent "
         "pour ne pas couper une idée en deux.",
         "params": {"taille": s.chunk_size, "chevauchement": s.chunk_overlap}},
        {"name": "Embeddings", "description": "Chaque passage devient un vecteur qui représente son sens.",
         "params": {"modèle": EMBEDDING_MODEL}},
        {"name": "Index Chroma", "description": "Les vecteurs sont stockés dans une base vectorielle "
         "en mémoire, propre à la session.", "params": {}},
        {"name": "Récupération", "description": "Le thème choisi est converti en vecteur et les passages "
         "les plus proches sont retrouvés.", "params": {"top-k": s.top_k}},
        {"name": "Prompt", "description": "Les passages retrouvés, le niveau et le type d'exercice "
         "forment la consigne envoyée au modèle.", "params": {}},
        {"name": "LLM", "description": "Le modèle génère les questions à partir des seuls passages fournis.",
         "params": {"modèle": s.llm_model}},
        {"name": "JSON validé", "description": "La réponse doit être du JSON conforme (4 choix, bonne "
         "réponse 0 à 3). Une seule nouvelle tentative est faite si elle ne l'est pas.", "params": {}},
        {"name": "Quiz", "description": "Quiz interactif, chaque question affiche les passages sources.",
         "params": {}},
    ]


def _label(stage: dict) -> str:
    params = "\\n".join(f"{k}: {v}" for k, v in stage["params"].items())
    return f"{stage['name']}\\n{params}" if params else stage["name"]


def pipeline_dot(s: Settings) -> str:
    stages = pipeline_stages(s)
    lines = ["digraph rag {", "  rankdir=TB;", "  node [shape=box, style=rounded];"]
    lines += [f'  n{i} [label="{_label(st)}"];' for i, st in enumerate(stages)]
    # theme query joins the flow at the retrieval stage, the rest is a straight line
    lines.append('  theme [label="Thème", shape=ellipse];')
    names = [st["name"] for st in stages]
    retrieval = names.index("Récupération")
    lines += [f"  n{i} -> n{i + 1};" for i in range(len(stages) - 1)]
    lines.append(f"  theme -> n{retrieval};")
    lines.append("}")
    return "\n".join(lines)
