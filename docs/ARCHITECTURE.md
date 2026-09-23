# Архитектура AML Graph Intelligence

```mermaid
flowchart LR
    A[3 parquet files] --> B[Schema adapter + validation]
    B --> C[Directed weighted graph]
    C --> D[Graph features]
    B --> E[Temporal features]
    C --> F[Cycles + bridges]
    E --> G[Recurring routes + anomalies]
    D --> H[Explainable role engine]
    C --> I[Louvain projection]
    F --> J[Priority score]
    G --> J
    H --> J
    I --> J
    J --> K[Evidence + validation]
    K --> L[CSV / Parquet / metadata]
    L --> M[Streamlit dashboard]
    L --> N[Read-only AML agent]
    N --> O[Local evidence answer]
    N -. explicit opt-in .-> P[OpenAI Agents SDK]
```

## Границы компонентов

| Компонент | Ответственность |
|---|---|
| `src/data.py` | Schema aliases, типы, ссылки на GID, отрицательные значения |
| `src/graph.py` | Направленный взвешенный `DiGraph`; undirected projection только для communities |
| `src/features.py` | Flow metrics, PageRank, HITS, betweenness, components, seed proximity |
| `src/temporal.py` | Active days, forwarding delay, rapid pass-through, burst |
| `src/patterns.py` | Cycles 2–6, recurring routes, composite temporal anomaly |
| `src/roles.py` | Пять независимых explainable scores и fallback `peripheral` |
| `src/clustering.py` | Louvain, bridge score, cluster summaries |
| `src/priority.py` | AML review priority, отдельно от уверенности в роли |
| `src/evidence.py` | Числовое объяснение роли ≤200 символов |
| `src/agent.py` | Read-only инструменты, ответы аналитику, audit log |
| `src/validation.py` | Машинные контракты обязательных и bonus outputs |
| `app.py` | Demo-first интерфейс без бизнес-логики расчётов |

## Agentic AI

`AMLAnalystAgent` имеет четыре read-only инструмента: рейтинг узлов, карточка GID, карточка кластера и AML-паттерны GID. Локальный evidence-agent всегда доступен и не передаёт данные наружу. OpenAI Agents SDK — опциональный режим с явным согласием пользователя; модель получает только строки, возвращённые выбранным инструментом, а не весь датасет.

Защита от галлюцинаций:

- фактические ответы строятся только из pipeline artifacts;
- неизвестный GID явно отклоняется;
- инструкции требуют разделять факт, гипотезу и рекомендацию;
- агент напоминает, что score не является доказательством нарушения;
- вопрос, режим, источники и latency пишутся в локальный `agent_audit.jsonl`.

## Масштабирование

Контракты признаков и выходов не зависят от NetworkX. Для ~1 млн узлов вычислительный слой заменяется на Polars/DuckDB + igraph/GraphFrames, exact betweenness — на sampling, а UI и агент продолжают читать те же агрегированные артефакты.
