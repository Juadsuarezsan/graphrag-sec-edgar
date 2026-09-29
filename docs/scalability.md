# Escalabilidad

Cómo se comporta el sistema al crecer el corpus (documentos) y la carga (consultas), y dónde se rompe. Las latencias provienen de `eval/runs/*.json` (medidas sin LLM); las cifras que dependen de la API de Anthropic son estimaciones con supuestos explícitos.

## Estado actual

| Dimensión | Valor medido |
|---|---|
| Grafo | 279 nodos / 452 aristas (fixture + Item 1 real de Alphabet) |
| Latencia GraphRAG | p50 ≈ 0 ms, p95 ≤ 4 ms por pregunta (planner + traversal en memoria) |
| Latencia Vector RAG | p50 ≈ 22 ms (TF-IDF exacto sobre 279 pasajes, Python puro) |
| Arranque de la API | ~1 s (fixture + ingesta por reglas + marcado de staleness) |
| Memoria | < 150 MB (sin torch) |

## Eje 1 — Corpus: de 280 a 5 000 / 20 000 y más allá

**Ingesta.** El coste dominante es la extracción con Claude. Supuestos: 300 filings (100 empresas × 3 años), ~60 000 caracteres útiles en Items 1/1A/7 por filing, chunks de 2 000 caracteres → ~30 chunks por filing → **~9 000 llamadas**. Con ~700 tokens de entrada y ~400 de salida por chunk y el precio del modelo pinneado (3 / 15 USD por millón), la ingesta completa cuesta **≈ 73 USD** y, a 5 req/s, unos 30 minutos en paralelo. Exhibit 21 y DEF 14A no consumen tokens (reglas). Los embeddings se calculan por lote por filing (`merge_result`): Voyage suma ~300 llamadas.

**Almacenamiento.** `NetworkXGraphStore` mantiene el grafo en memoria y lo vuelca a JSON: a 20 000 aristas el archivo pesa ~30 MB con embeddings de 256 d; funciona hasta ~1e5 nodos. Por encima, `GRAPH_BACKEND=neo4j`: el índice vectorial nativo sustituye la búsqueda exacta O(n) y el traversal variable `[*1..3]` se ejecuta en el servidor. El código está escrito y probado con un driver falso; la validación contra Neo4j real requiere Docker (ítem [B]).

**Búsqueda vectorial.** El mixin local hace coseno exacto contra todos los nodos: O(n·d). A 5 000 nodos × 256 d son 1,3 M multiplicaciones por consulta (~10 ms en Python puro); a 50 000 conviene Neo4j (HNSW) o NumPy. El planner sólo recurre a vectores cuando el linking léxico falla.

**Traversal.** BFS con filtro por tipo de arista: el tamaño de la frontera crece con el grado. Los nodos "hub" (TSMC, el riesgo *cybersecurity* con 16 empresas) están acotados por el esquema; con `max_hops=4` y 20 000 aristas el peor caso son unos miles de nodos visitados, todavía milisegundos.

## Eje 2 — Carga: 1 000 usuarios mensuales y 100× consultas

Con 1 000 usuarios mensuales y 20 consultas cada uno (20 000 consultas/mes):

| Componente | Sin LLM (modo determinista) | Con síntesis Claude |
|---|---|---|
| CPU | una réplica de uvicorn absorbe > 500 req/s de GraphRAG | igual; el trabajo corre en `asyncio.to_thread` |
| Tokens | 0 | ~1 200 in / 250 out por respuesta → 20 000 × (0,0036 + 0,00375) ≈ **150 USD/mes** |
| Latencia p95 | < 10 ms | 2–5 s (dominada por la API de Anthropic) |
| Rate limit | `RATE_LIMIT=30/minute` por IP con slowapi | subir el límite y cachear respuestas por `(pregunta normalizada, modo)` |

A 100× (2 M consultas/mes) el cuello de botella es la cuota de Anthropic y el coste (~15 000 USD/mes de síntesis). Mitigaciones en orden: (1) responder lookups y aggregations con la plantilla (exactas y citables) y reservar Claude para multi-hop; (2) caché de síntesis por hash de pregunta + subgrafo; (3) *prompt caching* del system prompt; (4) réplicas detrás de un balanceador: el grafo es de sólo lectura en consulta, así que escalar horizontalmente no requiere coordinación.

## Eje 3 — Frescura

Cada filing nuevo se ingiere de forma incremental: MERGE por id, `source_filing_date` actualizado, `staleness.mark` recalcula el flag. Un cron diario sobre `data.sec.gov/submissions` para las 100 CIK detecta nuevos 10-K/DEF 14A y sólo procesa esos (~30 llamadas al modelo por filing).

## Dónde se rompe

1. **Planner de reglas.** Preguntas fuera del sub-lenguaje (comparaciones numéricas, temporalidad, negación) devuelven plan vacío. Solución: fallback text-to-Cypher validado contra el esquema.
2. **Entity linking léxico.** Alias no vistos ("the iPhone maker") no enlazan; `hybrid` usa vectores como red de seguridad pero con hashing la recuperación es pobre. Solución: Voyage/bge + diccionario de alias extraído de los 10-K.
3. **Búsqueda vectorial exacta** por encima de ~5e4 nodos.
4. **JSON de NetworkX** por encima de ~1e5 nodos: pasar a Neo4j.
5. **Directores sintéticos**: el fixture no responde preguntas reales de gobierno corporativo hasta ingerir DEF 14A reales (`ingest_def14a` ya existe).
