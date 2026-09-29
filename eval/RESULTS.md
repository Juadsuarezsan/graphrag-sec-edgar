# Resultados de evaluación — GraphRAG sobre SEC EDGAR

Generado por `python -m eval.run` a partir de `eval/runs/2026-09-29-fixture-deterministic.json` (2026-09-29T02:38:23+00:00).

> **Alcance declarado.** Corrida **sin LLM**: síntesis por plantilla determinista, extracción por reglas. 
> Grafo evaluado: 279 nodos / 451 aristas construidos a partir del fixture curado a mano 
> (`src/graph/fixture.py`) más el excerpt real del 10-K de Alphabet. Los hit rates miden **recuperación estructural**, no la calidad 
> de la redacción del modelo. Faithfulness: **pendiente (requiere ANTHROPIC_API_KEY)**. Modelo pinneado para la corrida con LLM: `claude-sonnet-4-5-20250929`.

Eval set: `eval/gold.jsonl` — 100 preguntas (lookup 30 / dos saltos 30 / tres+ saltos 20 / aggregation 20), ground truth escrita a mano.

## Tabla comparativa (hit rate por categoría)

| Sistema | Lookup Hit | 2-hop Hit | 3-hop Hit | Aggregation Hit | Faithfulness |
|---|---|---|---|---|---|
| Vector RAG (TF-IDF, mismo corpus) | 86.7 % | 30.0 % | 0.0 % | 35.0 % | pendiente (requiere ANTHROPIC_API_KEY) |
| GraphRAG (este sistema) | 100.0 % | 100.0 % | 100.0 % | 100.0 % | pendiente (requiere ANTHROPIC_API_KEY) |
| Hybrid: Graph + Vector | 100.0 % | 100.0 % | 100.0 % | 85.0 % | pendiente (requiere ANTHROPIC_API_KEY) |
| Ablation: GraphRAG sin traversal multi-hop (max_hops=1) | 100.0 % | 30.0 % | 0.0 % | 95.0 % | pendiente (requiere ANTHROPIC_API_KEY) |

Hit = todos los ids esperados están en la respuesta (lookup / multi-hop) o el conteo coincide exactamente (aggregation).

## Precisión, routing, latencia y costo

| Sistema | Hit global | Precisión media (ids) | Routing acc. | p50 ms | p95 ms | Tokens in/out | Costo USD |
|---|---|---|---|---|---|---|---|
| Vector RAG (TF-IDF, mismo corpus) | 42.0 % | 40.0 % | 20.0 % | 22 | 23 | 0/0 | 0.0000 |
| GraphRAG (este sistema) | 100.0 % | 100.0 % | 96.0 % | 0 | 0 | 0/0 | 0.0000 |
| Hybrid: Graph + Vector | 97.0 % | 97.0 % | 96.0 % | 0 | 0 | 0/0 | 0.0000 |
| Ablation: GraphRAG sin traversal multi-hop (max_hops=1) | 58.0 % | 58.0 % | 96.0 % | 0 | 0 | 0/0 | 0.0000 |

Tokens y costo son 0 porque la corrida no invoca ningún LLM (fallback determinista, sin LLM).

## Extracción de entidades y relaciones (F1 contra gold anotado a mano)

| Extractor | Chunks | Entity P | Entity R | Entity F1 | Rel. P | Rel. R | Rel. F1 |
|---|---|---|---|---|---|---|---|
| `rules-v1` (determinista) | 5 | 97.5 % | 93.6 % | 95.5 % | 97.5 % | 93.6 % | 95.5 % |
| `claude-tool-use` (claude-sonnet-4-5-20250929) | — | pendiente (requiere ANTHROPIC_API_KEY) | | | | | |

Gold de extracción: `eval/extraction_gold.jsonl` (Exhibit 21 y DEF 14A sintéticos + un chunk real del 10-K de Alphabet), anotado a mano.

## Rúbrica de faithfulness (LLM-as-judge, pendiente)

- 1. Every factual claim in the answer is supported by a cited node or edge in the context.
- 2. No entity is named that is absent from the retrieved subgraph.
- 3. Counts and aggregates equal the number of matching nodes in the context.
- 4. The answer does not contradict any relationship in the context.
- 5. Score = supported_claims / total_claims, in [0, 1]; empty answers score 0.

## Diez peores casos de GraphRAG

| id | categoría | pregunta | hit | recall | precisión | esperado | obtenido |
|---|---|---|---|---|---|---|---|
| lk-01 | lookup | Who is the CEO of Apple? | sí | 1.00 | 1.00 | tim_cook | tim_cook |
| lk-02 | lookup | Who is the CEO of Microsoft? | sí | 1.00 | 1.00 | satya_nadella | satya_nadella |
| lk-03 | lookup | Who leads NVIDIA? | sí | 1.00 | 1.00 | jensen_huang | jensen_huang |
| lk-04 | lookup | Who is the CEO of Pfizer? | sí | 1.00 | 1.00 | albert_bourla | albert_bourla |
| lk-05 | lookup | Which company does Lisa Su lead? | sí | 1.00 | 1.00 | amd | amd |
| lk-06 | lookup | Which company does Mary Barra lead? | sí | 1.00 | 1.00 | gm | gm |
| lk-07 | lookup | Who is the CEO of Eli Lilly? | sí | 1.00 | 1.00 | david_ricks | david_ricks |
| lk-08 | lookup | Who is the CEO of TSMC? | sí | 1.00 | 1.00 | cc_wei | cc_wei |
| lk-09 | lookup | Who is the parent company of LinkedIn? | sí | 1.00 | 1.00 | microsoft | microsoft |
| lk-10 | lookup | Which company owns Instagram? | sí | 1.00 | 1.00 | meta | meta |

Análisis detallado en `docs/error_analysis.md`.
