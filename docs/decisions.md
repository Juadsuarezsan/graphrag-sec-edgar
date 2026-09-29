# Decisiones técnicas

Cada entrada registra qué se eligió, qué alternativa se descartó y con qué criterio. Las decisiones pendientes de validar con llaves o Docker se marcan.

## 1. Neo4j como destino, `GraphStore` como contrato y NetworkX como backend sin Docker

**Elegido:** un `Protocol` `GraphStore` con tres implementaciones (`InMemoryGraph`, `NetworkXGraphStore` con JSON, `Neo4jGraphStore` con Cypher). **Descartado:** acoplar el motor al driver de Neo4j o usar Kùzu/DuckDB-PGQ.

Criterios: (a) motor, tests y evaluación deben correr sin demonio Docker; (b) Neo4j 5.13+ aporta índice vectorial nativo y APOC para el despliegue real; (c) NetworkX cubre decenas de miles de nodos en memoria con persistencia trivial, que es el orden del DoD (≥5 000 nodos). Los tests de contrato (`tests/test_graph_store.py`, `tests/test_neo4j_store.py`) garantizan la misma semántica de MERGE y traversal en las tres. Kùzu se descartó por añadir una dependencia binaria sin ganancia a este tamaño.

## 2. GraphRAG planificado frente a Vector RAG puro

**Elegido:** entity linking + planner de relaciones + traversal dirigido. **Descartado:** recuperar pasajes por embedding y dejar que el LLM "infiera" los saltos.

El baseline de Vector RAG se implementó sobre el **mismo corpus** (cada nodo con sus aristas se vuelve un pasaje) para que la comparación sea justa. En `eval/RESULTS.md` el vector puro cae de 86,7 % (lookup) a 30 % (dos saltos) y 0 % (tres saltos): un pasaje contiene los vecinos de un nodo, no los vecinos de los vecinos. El grafo convierte la pregunta en un camino verificable y devuelve ids citables, que es lo que un analista de M&A necesita auditar.

## 3. Planner determinista en vez de "Claude escribe Cypher"

**Elegido:** planner de reglas (cues de relación → tipos de arista, dirección resuelta por las firmas del esquema). **Descartado por ahora:** text-to-Cypher con el LLM.

Criterios: reproducibilidad sin llave, plan inspeccionable en la respuesta (`plan`, `seeds`), cero riesgo de inyección de Cypher y latencia sub-milisegundo. El coste es cobertura lingüística: el planner soporta el sub-lenguaje documentado en `src/engine/planner.py`. La ruta LLM está prevista como *fallback* cuando no se enlaza ninguna entidad (ítem [B]).

## 4. Extracción con Claude vía *tool use* forzado y esquema cerrado

**Elegido:** `ClaudeExtractor` obliga a responder por la herramienta `record_extraction`, cuyo `input_schema` se genera de los modelos Pydantic; los tipos de nodo y arista son `Literal`. **Descartado:** pedir JSON libre y parsear.

Un `ValidationError` descarta el chunk en vez de contaminar el grafo con `ACQUIRED_BY` u `Organization`; `EDGE_SIGNATURES` rechaza relaciones con tipos incompatibles. Se usa el id fechado `claude-sonnet-4-5-20250929` con `timeout` explícito, `max_retries=0` en el SDK y `tenacity` (backoff exponencial, 4 intentos) sólo sobre errores de conexión, 429 y 5xx.

## 5. Reglas deterministas para Exhibit 21 y DEF 14A

**Elegido:** `RuleBasedExtractor` con parsers de tabla para subsidiarias y directores y cues conservadores para la prosa. **Descartado:** mandar también los exhibits al LLM.

Exhibit 21 y la tabla de nominados son estructurados; las reglas alcanzan F1 = 1,0 en los chunks anotados y cuestan cero tokens: para 100 empresas × 3 años son ~300 exhibits y ~300 proxies que no pasan por el modelo. En prosa las reglas priorizan precisión (F1 95,5 % sobre el gold de extracción, recall menor en párrafos densos); Claude cubre el recall.

## 6. Embeddings: hashing/TF-IDF por defecto, Voyage y bge opcionales

**Elegido:** `HashingEmbedder` (SHA-256, 256 d) para nodos y `TfidfHashingEmbedder` (1024 d, Python puro) para el baseline; `VoyageEmbedder` (`voyage-3`, HTTP directo con tenacity) y `SentenceTransformerEmbedder` (`bge-large-en-v1.5`) detrás del extra `[ml]`. **Descartado:** `sentence-transformers` como dependencia base.

Torch pesa ~6 GB y huggingface.co estaba bloqueado; la instalación pasó de minutos a segundos y CI no descarga modelos. El planner no depende de la similitud para preguntas con entidades nombradas; el hashing basta como *entry point* de respaldo. Ambos backends reales quedan cableados y probados con mocks.

## 7. Staleness como propiedad del grafo, no como tabla en Postgres

**Elegido:** `source_filing_date` y `updated_at` en cada nodo; `staleness.mark` fija `stale=True` si la evidencia tiene >365 días; `GET /api/stale`. **Descartado:** PostgreSQL + pgvector para auditoría.

La regla del DoD se resuelve con dos propiedades y una pasada al arrancar; Postgres duplicaba el estado en dos almacenes. La auditoría por request se cubre con `trace_id`, latencia, tokens y `cost_usd` en logs y respuesta. Un nodo sin fecha se considera stale: la procedencia desconocida nunca pasa por fresca.

## 8. Ground truth manual sobre un fixture declarado

**Elegido:** 100 preguntas escritas a mano sobre un grafo curado (`src/graph/fixture.py`) con el alcance declarado en `RESULTS.md`, README y `data_schema.md`. **Descartado:** generar el eval set con un LLM o publicar métricas sobre datos que no están en el repo.

Sin red a sec.gov ni llave de Anthropic no se puede construir ni verificar un grafo de 5 000 nodos; un eval set sobre un grafo pequeño pero verificable es preferible a uno grande irreproducible. Las celdas que exigen llave (`Faithfulness`, F1 de `ClaudeExtractor`) quedan como `pendiente (requiere ANTHROPIC_API_KEY)`.

## 9. Síntesis por plantilla en las corridas guardadas

La respuesta en lenguaje natural con citas la produce `ClaudeSynthesizer`; sin llave, todas las corridas usan `TemplateSynthesizer` y lo declaran (`synthesizer="template"`). Los hit rates se calculan sobre `answer_ids`, que no dependen del sintetizador, así que la comparación Vector/Graph/Hybrid es válida con o sin LLM.

## 10. slowapi + CORS por variable de entorno + salida saneada

`RATE_LIMIT` (por defecto `30/minute`) se aplica a `/api/query`; `CORS_ORIGINS` es una lista explícita (nunca `*` por defecto); `sanitize_text` elimina etiquetas HTML y caracteres de control de cualquier texto que salga del sintetizador. La demo renderiza con `textContent`.

## 11. Visualización propia en vez de `react-force-graph` por CDN

La demo implementa una simulación de fuerzas (~80 líneas, canvas 2D con *grid hashing* para la repulsión) porque el entorno bloquea CDNs y un frontend Next.js era desproporcionado para una página estática. Funciona sin dependencias, con `prefers-color-scheme` y a 375 px. Si el proyecto crece a Next.js, `react-force-graph-2d` sigue siendo la elección natural (documentada, no ejecutada).
