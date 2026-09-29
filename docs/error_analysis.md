# Análisis de errores

Basado en `eval/runs/2026-09-29-fixture-deterministic.json` (regenerable con `python -m eval.run`). La tabla "Diez peores casos" de `eval/RESULTS.md` se genera automáticamente; este documento explica las causas.

## Resumen

| Sistema | Hit global | Fallos |
|---|---|---|
| GraphRAG (`max_hops=3`) | 100 / 100 | 0 |
| Hybrid | 97 / 100 | 3 (aggregation) |
| Ablation `max_hops=1` | 58 / 100 | 42 (todas las de 2 y 3 saltos, más una aggregation de 2 saltos) |
| Vector RAG | 42 / 100 | 58 |

**Advertencia de alcance.** El 100 % de GraphRAG no dice que el sistema resuelva cualquier pregunta sobre 10-K reales. Dice tres cosas más modestas: (1) el gold se escribió sobre el mismo grafo que se consulta, así que no hay error de extracción entre pregunta y respuesta; (2) las 100 preguntas usan el sub-lenguaje que el planner soporta (ver `src/engine/planner.py`); (3) el planner y las tablas del fixture no tienen bugs sobre esas 100 preguntas. Los fallos reales del sistema completo aparecerán en la extracción sobre filings reales y en preguntas fuera del sub-lenguaje; ambos quedan como trabajo [B]/futuro y se listan abajo como riesgos conocidos.

## Fallos de Hybrid (3 casos, todos aggregation)

Hybrid añade seeds vectoriales sólo cuando el linking léxico no encuentra ninguna entidad. En las tres preguntas sin seed (`ag-12` "How many directors serve on more than one board?", `ag-18` "How many subsidiaries are incorporated in Ireland?", `ag-19` "How many products are in the Vaccine category?") el modo `graph` ejecuta un plan *seedless* (filtro por propiedad o por grado) que da el conteo exacto; el modo `hybrid` inyecta tres seeds por similitud de hashing y el plan pasa a expandir desde nodos irrelevantes, de modo que el conteo cambia. **Causa:** la heurística "sin seed → usar vectores" es correcta para preguntas abiertas y perjudicial para preguntas de filtro. **Corrección prevista:** no inyectar seeds vectoriales cuando el plan seedless tiene `answer_type` y algún filtro; medir de nuevo.

## Fallos de la ablación `max_hops=1` (42 casos)

Esperados por construcción: al truncar el plan a un paso, las preguntas de dos y tres saltos devuelven el frontier intermedio (empresas en vez de CEOs, proveedores en vez de clientes), que no coincide con el tipo de respuesta y se filtra a vacío. Es la evidencia cuantitativa de que el traversal multi-hop aporta 42 puntos de hit rate sobre este gold.

## Fallos de Vector RAG (58 casos): las cinco causas

1. **Los saltos no están en ningún pasaje (30 casos de 2-hop, 20 de 3-hop).** El pasaje de LinkedIn dice "is a subsidiary of Microsoft"; el de Microsoft dice "is led by CEO Satya Nadella". Ninguno contiene ambas cosas, y la síntesis por plantilla no encadena pasajes. Es la limitación estructural que motiva el grafo.
2. **Conteos imposibles (13 de 20 aggregation).** El baseline cuenta las entidades del tipo pedido presentes en los `k=5` pasajes recuperados; con 15 empresas que mencionan China no caben en cinco pasajes. Sólo acierta cuando el conjunto esperado es pequeño y todo aparece en el pasaje del nodo semilla.
3. **Ruido de nombres largos (4 lookups).** TF-IDF con hashing pondera tokens raros; "Advanced Micro Devices, Inc." y "Taiwan Semiconductor Manufacturing Company" reparten peso en tokens que también aparecen en otros pasajes, y `Who leads NVIDIA?` recupera pasajes de GPU y no el de Jensen Huang.
4. **Direccionalidad.** "supplied by TSMC" y "suppliers of Apple" comparten vocabulario; el pasaje de Apple contiene ambas frases ("is supplied by"/"is a supplier of") y el baseline no distingue el sentido.
5. **Preguntas de propiedad.** `What is Microsoft's ticker?` recupera el pasaje correcto pero la plantilla vectorial no extrae el valor `MSFT`, así que `expected_text` no aparece.

## Riesgos conocidos del sistema completo (no medibles todavía)

| Riesgo | Dónde aparecerá | Mitigación prevista |
|---|---|---|
| Recall de extracción en prosa densa | `ClaudeExtractor` sobre Items 1/1A reales | gold de extracción ampliado a ≥ 50 chunks anotados; medir F1 por tipo |
| Entidades duplicadas por variantes de nombre ("Alphabet Inc." vs "Google LLC") | MERGE por `id` slug | tabla de alias por CIK + resolución con embeddings antes del MERGE |
| Alucinación de relaciones | síntesis con Claude | juez de faithfulness (rúbrica en `src/evaluation/metrics.py`) y citas obligatorias por id |
| Preguntas fuera del sub-lenguaje | planner | fallback text-to-Cypher validado contra `EDGE_SIGNATURES` |
| Directores sintéticos | fixture | reemplazar por DEF 14A reales (`ingest_def14a` ya existe) |

## Cómo reproducir este análisis

```bash
python -m eval.run                     # regenera eval/runs/*.json y eval/RESULTS.md
python - <<'EOF'
import json, glob
run = json.load(open(sorted(glob.glob("eval/runs/*.json"))[-1]))
for s in run["systems"]:
    misses = [c for c in s["cases"] if not c["hit"]]
    print(s["label"], len(misses))
    for c in misses[:5]:
        print("  ", c["id"], c["question"], "->", c["answer_ids"][:5], c["count"])
EOF
```
