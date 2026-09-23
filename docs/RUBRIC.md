# Матрица критериев жюри

Эта таблица не обещает оценку жюри; она показывает, где проверяется каждый критерий.

| Критерий | Доказательство в проекте |
|---|---|
| Соответствие задаче и работоспособность — 25 | `run_pipeline.py`, три обязательных CSV, dashboard, AI-аналитик, тест на 2248 узлов |
| Техническая реализация — 25 | Модульный `src/`, directed flow, PageRank/HITS/betweenness, Louvain, temporal patterns, OpenAI Agents SDK tools, validation |
| README и воспроизводимость — 25 | `README.md`, `docs/ARCHITECTURE.md`, `docs/DEMO.md`, pinned ranges, one-command pipeline, pytest, GitHub Actions |
| Ценность и применимость — 15 | TOP review queue, evidence, GID card, cluster explorer, cycles/routes/anomalies, limitations, audit log |
| Потенциал и оригинальность — 10 | role ≠ priority, depth/seed-aware logic, resilience, read-only evidence-grounded agent, million-node migration path |

## Проверяемые acceptance gates

- ровно 2248 уникальных GID;
- все обязательные поля заполнены;
- scores находятся в `[0,1]`;
- каждый evidence содержит число и не длиннее 200 символов;
- 444 depth-boundary узла не получают роль terminal только из-за отсутствия выхода;
- seed pass-through отключён;
- TOP отсортирован;
- cycles не усечены лимитом;
- pipeline работает менее пяти минут;
- локальный агент называет существующие GID и источники ответа;
- неизвестный GID не выдумывается.

Все gates исполняются командой `pytest -q` на реальных данных из репозитория.
