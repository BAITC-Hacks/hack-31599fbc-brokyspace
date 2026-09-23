from __future__ import annotations

import os

import agents

from src.agent import AMLAnalystAgent


def test_agent_answers_core_question_from_pipeline(pipeline_output):
    agent = AMLAnalystAgent(pipeline_output)
    answer = agent.ask("Кого из 2248 клиентов смотреть первым и почему?")
    assert "Кого смотреть первым" in answer.text
    assert "priority=" in answer.text
    assert "node_features.parquet" in answer.sources
    assert answer.mode == "local-evidence"


def test_agent_inspects_exact_gid_without_hallucination(pipeline_output):
    agent = AMLAnalystAgent(pipeline_output)
    gid = str(agent.features.nlargest(1, "priority_score").iloc[0]["gid"])
    answer = agent.ask(f"Почему нужно проверить GID {gid}?")
    assert gid in answer.text
    assert "Рекомендация" in answer.text
    assert "не доказательство" in answer.text


def test_agent_rejects_unknown_gid(pipeline_output):
    answer = AMLAnalystAgent(pipeline_output).ask("Проверь GID 999999999999999999")
    assert "отсутствует" in answer.text


def test_openai_agent_wiring_without_network(pipeline_output, monkeypatch):
    class Result:
        final_output = "Проверяемый ответ"

    def fake_run(agent, question, max_turns):
        assert agent.name == "AML Graph Analyst"
        assert len(agent.tools) == 4
        assert question == "Кого проверить?"
        assert max_turns == 8
        return Result()

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agents.Runner, "run_sync", staticmethod(fake_run))
    answer = AMLAnalystAgent(pipeline_output).ask(
        "Кого проверить?", use_openai=True, api_key="test-key", model="gpt-6-astra"
    )
    assert answer.text == "Проверяемый ответ"
    assert answer.mode == "openai:gpt-6-astra"
    assert "OPENAI_API_KEY" not in os.environ
