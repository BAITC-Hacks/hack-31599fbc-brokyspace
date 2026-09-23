# AML Graph Intelligence

Локальный, воспроизводимый и объяснимый инструмент восстановления финансовой структуры организованной группы по транзакционной сети. Решение строит направленный граф движения денег, назначает функциональные роли всем узлам, ранжирует их для AML-проверки, выделяет сообщества и формирует доказательную базу в понятной аналитику форме.

## 1. Задача и решение

Оборот сам по себе редко показывает, кто управляет сетью. Координатор может переводить меньше денег, но связывать критические части графа; узел на границе четырёх шагов может выглядеть терминальным только из-за усечения выборки. Поэтому решение сочетает направленные потоки, центральности, связь с seed, временное поведение и структуру сообществ. Black-box классификатор не используется: каждая роль и позиция в рейтинге выводятся из формальных признаков.

## 2. Архитектура

```text
data/*.parquet
  → src/data.py           чтение, приведение схемы, проверки
  → src/graph.py          directed weighted graph
  → src/features.py       flow + PageRank/HITS/betweenness/seed
  → src/temporal.py       скорость перенаправления и всплески
  → src/patterns.py       cycles + recurring routes + anomalies
  → src/clustering.py     Louvain и межкластерные мосты
  → src/roles.py          пять role scores + peripheral
  → src/priority.py       AML priority score
  → src/evidence.py       объяснения с фактическими числами
  → src/validation.py     контракт выходных данных
  → out/*                 CSV, parquet, metadata
  → app.py                Streamlit dashboard
```

## 3. Quick Start

Требуется Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --data ./data --out ./out
streamlit run app.py
```

Полный pipeline выполняется одной командой. Код возврата ненулевой при нарушении контракта. Для набора другого размера проверку `2248` можно изменить через `--expected-nodes N` или отключить значением `0`.

Если командная строка неудобна, можно сразу открыть `streamlit run app.py`: при отсутствии результатов стартовый экран предложит загрузить три parquet-файла и запустить pipeline одной кнопкой. Обработка остаётся локальной.

## 4. Входные данные

В `data/` должны находиться:

- `nodes.parquet`: уникальный `gid`, а также `depth` и `is_seed`;
- `edges.parquet`: `source`, `target`, `sum_kzt`, опционально `n_tx`;
- `transactions.parquet`: `source`, `target`, сумма и timestamp.

Загрузчик понимает распространённые эквиваленты (`src/dst`, `sender_gid/receiver_gid`, `amount_kzt`, `transaction_date` и др.), но после чтения приводит всё к единой схеме. Проверяются уникальность GID, ссылки на существующие узлы, типы и неотрицательность сумм. Если timestamps отсутствуют, temporal-признаки честно принимают нулевые значения, а не имитируются.

## 5. Pipeline

Последовательность: load → validate → graph → network features → temporal features → Louvain → role scores → cluster importance → priority → evidence → validation → export. В консоли печатается короткий EDA, время каждого этапа и общий runtime; подробности сохраняются в `out/metadata.json`.

## 6. Feature engineering

На направленном графе вычисляются `in/out_degree`, `in/out_kzt`, `in/out_tx`, PageRank, betweenness, HITS hub/authority, weak component, расстояние/близость к seed, число соседей и непосредственных источников-seed. `pass_through = min(in_kzt,out_kzt)/max(in_kzt,out_kzt)` лежит в `[0,1]`.

Для основных величин используются percentile ranks. Они сохраняют относительную редкость сигнала при тяжёлых хвостах денежных распределений и устойчивее абсолютных порогов при смене периода или размера сети. Например, `in_deg_pct=0.99` означает, что узел превосходит 99% сети по числу входящих контрагентов.

## 7. Temporal analysis

По транзакциям считаются активные входящие/исходящие дни, медианная задержка между наблюдаемым входом и следующим исходящим событием, доля перенаправлений до 24/48 часов и всплеск дневной активности. Поиск предыдущего входа сделан через `merge_asof`, без квадратичного сопоставления транзакций.

Дополнительный `temporal_anomaly_score` объединяет percentile всплеска, rapid forwarding, участие в повторяющихся маршрутах и циклических потоках. Это приоритизация для расследования, а не вероятность нарушения.

## 8. Методология ролей

Для каждого узла независимо считаются пять scores в `[0,1]`, затем выбирается самый сильный. Если максимум меньше `0.38`, узел относится к `peripheral`. В формулах `P(x)` — percentile, `R` — observed retention, `B` — наличие входа и выхода, `S` — seed connectivity, `X` — bridge score:

```text
consolidator = .23P(in_deg)+.20P(in_kzt)+.18P(authority)+.15P(seed_sources)+.12fan_in+.12R
distributor  = .27P(out_deg)+.20P(out_tx)+.18P(hub)+.18fan_out+.10P(out_kzt)+.07S
transit      = .27B+.30pass_through+.18P(rapid)+.12same_next_day+.13(1-R)
terminal     = [.26P(in_kzt)+.18P(in_deg)+.24no_out+.20R+.12(1-P(out_kzt))] × has_input
coordinator  = .29P(betweenness)+.17P(PageRank)+.12max(P(hub),P(authority))+.22X+.12S+.08P(degree)
```

`role_score` — сила соответствия выбранной роли, не риск и не вероятность преступления. Для seed баланс вход/выход ненадёжен из-за неполного входящего контекста: их transit score считается отдельно без обычного pass-through.

## 9. Ограничение depth=4

Комбинация `depth == max(depth)` и `out_degree == 0` помечается `truncated_by_depth`. Она означает границу обхода, а не доказанную остановку средств. Поэтому terminal score такого узла умножается на `0.35`, а `evidence` предупреждает об усечении.

## 10. Priority scoring

`priority_score` отвечает на другой вопрос: насколько важно проверить узел AML-аналитику.

```text
.30 structural importance + .20 non-peripheral role strength
+.18 seed connectivity + .12 temporal anomaly
+.10 cluster importance + .10 transaction volume
```

Структура включает betweenness, PageRank, bridge и HITS. Денежный объём ограничен весом 10%, поэтому крупнейшие клиенты не вытесняют структурно значимые узлы.

## 11. Кластеризация

Louvain применяется к взвешенной неориентированной проекции, где встречные потоки суммируются по `sum_kzt`. Community detection ищет плотные группы связности, тогда как flow features, роли и транзитная логика остаются направленными. Для каждого кластера рассчитываются размер, seed, внутренний оборот, TOP GID и детерминированная hypothesis.

## 12. Explainability

Каждый `evidence` непустой, не длиннее 200 символов и содержит фактические числа: контрагентов, суммы, percentile, долю перенаправления или межкластерные связи. Это краткая причина назначения роли, а не шаблон «высокий score».

## 13. Выходные схемы

`out/nodes_roles.csv`: `gid, role, role_score, cluster_id, priority_score, evidence`.

`out/clusters.csv`: `cluster_id, n_nodes, n_seed, sum_kzt_internal, top_gids, hypothesis`.

`out/top_nodes.csv`: `rank, gid, role, priority_score, why`; минимум 20 строк, убывание priority.

Дополнительно создаются `node_features.parquet`, `graph_edges.parquet`, `resilience.csv`, `cycles.csv`, `recurring_routes.csv`, `anomalies.csv` и `metadata.json`; они питают dashboard и аудит расчётов.

## 14. Dashboard и demo-сценарий

Dashboard рассчитан на пятиминутную демонстрацию:

1. «Обзор»: KPI, роли, структурная значимость и устойчивость сети.
2. «Приоритет»: TOP-20 и объяснение, почему turnover не доминирует.
3. «Поиск GID»: карточка узла с потоками, ролью и evidence.
4. «Граф»: top-risk, ego-network и кластер; стрелки показывают направление денег, цвет — роль.
5. «Кластеры»: seed, оборот, роли и автоматически созданная hypothesis.
6. «Методология»: ограничения данных и отсутствие ground truth.

## 15. Автоматическая валидация

Перед экспортом проверяются число узлов, обязательные поля и отсутствие null, уникальность gid, допустимые роли, диапазоны scores, evidence/cluster, схема и непустота кластеров, размер и сортировка TOP. Ошибка прерывает pipeline до выдачи внешне корректных, но логически неверных CSV.

## 16. Bonus: AML patterns и network resilience

Из графа последовательно удаляются TOP-5/10/20 узлов по priority. `resilience.csv` показывает число weak components, размер/долю крупнейшей компоненты и fragmentation. График доступен на обзорной странице dashboard.

Также реализованы:

- направленные циклы длиной 2–6 с bottleneck-суммой, наличием seed и `cycle_score`;
- recurring routes, наблюдаемые в несколько дней, с оценкой cadence regularity;
- rapid transit по задержке до 24/48 часов;
- temporal anomalies по всплескам, скорости перенаправления, циклам и повторным маршрутам.

Порог recurring routes определяется 75-м percentile среди многодневных связей, а список anomalies — 95-м percentile composite score. Результаты доступны на вкладке «AML-паттерны».

## 17. Ограничения

- выборка ограничена четырьмя поколениями обхода;
- наблюдение преимущественно outgoing и входы seed неполны;
- роли без ground truth являются проверяемыми аналитическими гипотезами;
- отсутствие timestamps снижает силу temporal-части;
- PageRank и HITS отражают только наблюдаемую сеть.

## 18. Масштабирование до ~1 млн узлов

Текущая NetworkX-реализация оптимальна для локального кейса на 2248 узлах. Для миллиона узлов контракт и формулы сохраняются, но exact betweenness заменяется sampling approximation, граф переводится в `igraph`/`graph-tool` или Spark GraphFrames, parquet обрабатывается Polars/DuckDB, Louvain — пакетным Leiden/Louvain, а dashboard читает агрегаты и заранее подготовленные ego-графы. Расчёт остаётся explainable; меняется только backend.
