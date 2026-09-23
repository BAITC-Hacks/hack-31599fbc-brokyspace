# Матрица критериев жюри

Эта таблица не обещает оценку жюри; она показывает, где проверяется каждый критерий.

| Критерий | Доказательство в проекте |
|---|---|
| Соответствие задаче и работоспособность — 25 | `run_pipeline.py`, три обязательных CSV, dashboard, AI-аналитик, тест на 2248 узлов |
| Техническая реализация — 25 | Модульный `src/`, directed flow, PageRank/HITS/betweenness, Louvain, temporal patterns, recurring A→B→C, structuring/depth-peer anomalies, OpenAI Agents SDK tools, validation |
| README и воспроизводимость — 25 | `README.md`, `docs/ARCHITECTURE.md`, `docs/DEMO.md`, pinned ranges, one-command pipeline, pytest, GitHub Actions |
| Ценность и применимость — 15 | TOP review queue, evidence, GID card, cluster explorer, cycles/routes/anomalies, completeness + next request, audit log |
| Потенциал и оригинальность — 10 | role ≠ priority, depth/seed-aware logic, recurring chains, structuring, resilience, read-only evidence-grounded agent, million-node migration path |

## Проверяемые acceptance gates

- ровно 2248 уникальных GID;
- все обязательные поля заполнены;
- scores находятся в `[0,1]`;
- каждый evidence содержит число и не длиннее 200 символов;
- 444 depth-boundary узла не получают роль terminal только из-за отсутствия выхода;
- seed pass-through отключён;
- TOP отсортирован;
- cycles не усечены лимитом;
- recurring A→B→C, синхронные входы и structuring-сигналы непусты;
- depth-peer anomaly находится в `[0,1]`;
- completeness покрывает каждый GID и содержит следующий запрос;
- pipeline работает менее пяти минут;
- локальный агент называет существующие GID и источники ответа;
- неизвестный GID не выдумывается.

Все gates исполняются командой `pytest -q` на реальных данных из репозитория.
