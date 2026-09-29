# Arquitectura

![Arquitectura](architecture.svg)

## Resumen

Dos fases sobre un mismo grafo de conocimiento. **Ingesta**: HTML de SEC EDGAR → texto → secciones del 10-K → chunks → extracción (reglas o Claude) → MERGE en el grafo con fecha de evidencia. **Consulta**: pregunta → entity linking → plan de relaciones → traversal → agregación → síntesis con citas → subgrafo para la demo. Un baseline de Vector RAG sobre el mismo corpus permite medir qué aporta el grafo.

## Módulos y responsabilidades

| Capa | Módulo | Responsabilidad | No sabe de |
|---|---|---|---|
| Configuración | `src/config.py` | `Settings` desde variables de entorno (modelo pinneado, backend, embedder, CORS, rate limit, staleness) | dominio, HTTP |
| Observabilidad | `src/observability.py` | `trace_id` (contextvar), `RequestMetrics` (latencia, tokens, `cost_usd`), tabla de precios, exportador LangSmith REST | dominio |
| Embeddings | `src/embeddings/` | `Embedder` Protocol: hashing, TF-IDF hashing, Voyage (HTTP), sentence-transformers (extra `[ml]`, import perezoso) | grafo |
| Grafo | `src/graph/` | `GraphStore` Protocol; `InMemoryGraph`, `NetworkXGraphStore` (JSON), `Neo4jGraphStore` (Cypher, índice vectorial); `staleness`; `fixture` | HTTP, LLM |
| Ingesta | `src/ingestion/` | `html_to_text` (DOM, oculta XBRL), `split_sections` (Items 1/1A/7…), `chunk_text`, parsers de Exhibit 21 y DEF 14A, `pipeline` (extracción → MERGE con `source_filing_date`) | HTTP |
| Extracción | `src/extraction/` | Esquemas cerrados (`NodeType`, `EdgeType`, `EDGE_SIGNATURES`), `EntityExtractor` Protocol, `RuleBasedExtractor`, `ClaudeExtractor` (tool use forzado, tenacity, coste) | grafo, HTTP |
| Motor | `src/engine/` | `EntityLinker`, `planner` (cues → pasos, dirección por firmas, intersección multi-seed), `GraphRAGEngine` (graph/vector/hybrid), `VectorIndex` (baseline), `TemplateSynthesizer`/`ClaudeSynthesizer`, `sanitize_text` | HTTP |
| API | `src/api/` | FastAPI: validación Pydantic (422), CORS por env, slowapi (429), `X-Trace-Id`, errores tipados, `/health`, `/api/query`, `/api/graph`, `/api/stale`, `/api/node/{id}`, `/api/schema`, `/demo` | detalles del grafo |
| Evaluación | `src/evaluation/`, `eval/` | gold (100 preguntas), métricas (hit, P/R, F1 de extracción, percentiles, rúbrica de faithfulness), runner que escribe `eval/runs/*.json` y `eval/RESULTS.md` | HTTP |

Regla de dependencias: `api → engine → (graph, extraction, embeddings) → config`; `ingestion → extraction → graph`. Ningún módulo de dominio importa FastAPI ni lee `os.environ`.

## Flujo de ingesta

1. `download_data.py` (o los excerpts de `data/sec_edgar_sample/`) aporta HTML + metadatos (`Filing`).
2. `html_to_text` recorre el DOM con lxml: elimina `ix:header` y `display:none`, mantiene los runs `<span><font>` en la misma línea, convierte `&nbsp;` y separa celdas con ` | `.
3. `split_sections` localiza los `Item N` y, para cada uno, conserva la ocurrencia con el cuerpo más largo (la del índice es corta). Devuelve `Section` para Items 1, 1A, 1B, 1C, 2, 3, 7, 7A, 8.
4. `chunk_text` corta en fin de oración con solape; `Chunk.id` es determinista (accession + item + índice + hash).
5. El extractor recibe cada chunk y la entidad focal (el filer, reutilizando el nodo existente por ticker/CIK). `RuleBasedExtractor` para EX-21/DEF14A y cues de prosa; `ClaudeExtractor` cuando hay `ANTHROPIC_API_KEY`.
6. `merge_result` calcula embeddings **por lote** y hace MERGE de nodos y aristas con `source_filing_date`, `source_accession` y `evidence`.
7. `staleness.mark` marca `stale=True` en nodos cuya evidencia supera `STALE_AFTER_DAYS` (365).

## Flujo de consulta

1. `EntityLinker.link` encuentra menciones por alias (más largo primero; posesivos, plurales, tickers).
2. `build_plan` extrae cues de relación, los asigna a la mención vecina (los anteriores se invierten; los posteriores van en orden), colapsa duplicados, detecta `share/same`, `how many`, filtros `S&P 500` / sector / propiedad y deriva el tipo de respuesta de la firma del último paso.
3. `GraphRAGEngine.execute` expande cada constraint sobre `GraphStore.neighbors` con la dirección resuelta en tiempo de ejecución; varias menciones se intersectan; sin menciones se aplica un plan *seedless* (filtro por propiedad o grado).
4. Aggregation devuelve el conteo del frontier final; los lookups de propiedad leen el nodo semilla.
5. El sintetizador produce el texto (plantilla determinista o Claude con citas `[id]`), saneado.
6. La respuesta incluye `answer_ids`, `plan`, `seeds`, `cited_nodes`, `subgraph` y `metrics` (`trace_id`, latencia, tokens, `cost_usd`).

Modos: `graph` (lo anterior), `vector` (TF-IDF sobre pasajes de nodo + aristas; la respuesta son las entidades nombradas en los `k` pasajes) e `hybrid` (seeds vectoriales sólo si el linking falla, luego el plan).

## Backends del grafo

| Backend | Uso | Persistencia | Búsqueda vectorial | Traversal |
|---|---|---|---|---|
| `memory` | tests, demo, eval | ninguna (fixture + muestra al arrancar) | coseno exacto | listas de adyacencia |
| `networkx` | ejecución local sin Docker con miles de nodos | JSON atómico (`GRAPH_PATH`) | coseno exacto | `MultiDiGraph` |
| `neo4j` | producción | Neo4j 5.25 | `db.index.vector.queryNodes` | `MATCH p=(s)-[*1..k]-(m)` |

## Observabilidad

Cada request lleva `X-Trace-Id`; el middleware loguea método, ruta, estado y latencia; `RequestMetrics.finish()` emite tokens y coste. Con `LANGSMITH_API_KEY` el `LangSmithTracer` envía cada consulta como *run* a la API REST de LangSmith (con tenacity); sin llave es inerte y lo dice en `/health`.

## Seguridad

Validación Pydantic con límites (`question` 3–1000 caracteres, `k` 1–25, `max_hops` 1–4, `mode` cerrado) → 422; `slowapi` → 429; CORS sólo para `CORS_ORIGINS`; salida saneada; secretos sólo por entorno (`.env.example` sin valores); `gitleaks` en CI y ejecutado localmente sin hallazgos.
