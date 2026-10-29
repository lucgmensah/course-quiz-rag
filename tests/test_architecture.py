import dataclasses

from rag.architecture import pipeline_dot, pipeline_stages
from rag.config import load_settings


def test_stage_params_come_from_settings():
    s = dataclasses.replace(load_settings(), chunk_size=512, top_k=9, llm_model="my-model")
    stages = {st["name"]: st for st in pipeline_stages(s)}
    assert stages["Découpage"]["params"]["taille"] == 512
    assert stages["Récupération"]["params"]["top-k"] == 9
    assert "512" in pipeline_dot(s) and "my-model" in pipeline_dot(s)


def test_dot_contains_every_stage():
    s = load_settings()
    dot = pipeline_dot(s)
    for stage in pipeline_stages(s):
        assert stage["name"] in dot
    assert dot.startswith("digraph")
