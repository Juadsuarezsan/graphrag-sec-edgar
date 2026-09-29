# Rendimiento y perfilado

Fuente de las cifras: `eval/runs/2026-09-29-fixture-deterministic.json` (100 preguntas × 4 sistemas, grafo de 279 nodos / 451 aristas, Python 3.11, CPU sin GPU). Regenerable con `python -m eval.run`.

## Latencia por sistema (100 preguntas, sin LLM)

| Sistema | p50 | p95 | media |
|---|---|---|---|
| GraphRAG (planner + traversal) | 0 ms | 0 ms | 0,01 ms |
| Hybrid (linking → vectores sólo si falla el linking) | 0 ms | 0 ms | 0,19 ms |
| Vector RAG (TF-IDF exacto, 279 pasajes, 1024 d) | 22 ms | 23 ms | 22,8 ms |
| Ablation GraphRAG `max_hops=1` | 0 ms | 0 ms | 0,00 ms |

Las latencias de GraphRAG son sub-milisegundo porque el linking es una búsqueda de subcadenas sobre ~600 alias y el traversal recorre listas de adyacencia en memoria. La demo reporta 0–4 ms por pregunta en `metrics.latency_ms`.

## Dónde está el cuello de botella

1. **Con LLM: la red.** Cada síntesis con `claude-sonnet-4-5-20250929` añade 2–5 s y ~1 200/250 tokens; cada chunk de extracción, 1–3 s. Todo lo demás es ruido frente a eso. Por eso el motor corre en `asyncio.to_thread` y la API es `async`: el hilo del event loop queda libre mientras el SDK espera.
2. **Sin LLM: el baseline vectorial.** `VectorIndex.query` calcula el coseno contra todos los pasajes en Python puro (O(n·d) = 279 × 1024 ≈ 286 k multiplicaciones → 22 ms). Es el 99 % del tiempo de la corrida vectorial. A 5 000 nodos serían ~400 ms por consulta: el punto en que conviene NumPy o el índice HNSW de Neo4j.
3. **Arranque.** `populate` + ingesta por reglas del excerpt real + `staleness.mark` + construcción del índice de alias tardan ~1 s. El índice vectorial se construye de forma perezosa en la primera consulta `vector`/`hybrid`.

## Perfil de una pregunta de tres saltos

`Which CEOs lead companies that share a supplier with Apple?` (3h-01):

| Fase | Trabajo | Tiempo aprox. |
|---|---|---|
| `EntityLinker.link` | normalizar, recorrer alias ordenados por longitud | ~0,3 ms |
| `build_plan` | 10 regex de cues, colapso, tipo de respuesta por firmas | ~0,2 ms |
| `execute` | 3 expansiones: 5 proveedores → 9 clientes → 9 CEOs; 35 aristas registradas | ~0,5 ms |
| `TemplateSynthesizer` | formatear 9 nombres + 4 evidencias | ~0,05 ms |
| Serialización Pydantic (`GraphAnswer` con subgrafo) | ~40 nodos, 35 aristas | ~1–2 ms |

La serialización de la respuesta (subgrafo para la visualización) pesa más que el razonamiento; `subgraph` se recorta a 150 nodos / 300 aristas.

## Ingesta

| Paso | Coste | Nota |
|---|---|---|
| `html_to_text` | ~50 ms por 10-K de 1 MB con lxml | recorrido DOM único |
| `split_sections` | ~5 ms | regex multilinea + elección del cuerpo más largo por Item |
| `chunk_text` | < 1 ms por sección | cortes en fin de oración |
| `RuleBasedExtractor` | ~1 ms por chunk | regex + léxico |
| `ClaudeExtractor` | 1–3 s por chunk, ~700/400 tokens | estimación; se mide en `IngestReport.latency_ms` cuando hay llave |
| `merge_result` | embeddings en un lote por filing + MERGE O(entidades + aristas) | sin llamadas por nodo |

## Caché y batching implementados

* `get_settings()` con `lru_cache`: la configuración se lee una vez.
* Índice de alias y índice vectorial se construyen una vez por proceso (`GraphRAGEngine.refresh()` los invalida tras una ingesta).
* Embeddings por lote (`Embedder.embed(list)`) en ingesta y en la construcción del índice; `VoyageEmbedder` agrupa hasta 64 textos por llamada.
* `TfidfHashingEmbedder.fit` una sola vez sobre el corpus.

## Pendiente de medir (requiere llave o Docker)

* p95 de extremo a extremo con `ClaudeSynthesizer` (objetivo del DoD: < 10 s, ideal < 5 s).
* Latencia de `db.index.vector.queryNodes` y del traversal variable en Neo4j 5.25 con ≥ 5 000 nodos.
* Coste real por request (`cost_usd` ya se calcula por tokens; hoy es 0 porque no hay llamadas).
