# Muestra de excerpts reales de 10-K (SEC EDGAR)

Seis archivos descargados de la API pública de EDGAR (sin autenticación, sólo `User-Agent`) con el script regex anterior a `scripts/download_data.py`. Se conservan como **evidencia** de los tres formatos de HTML que rompían el regex y como único texto real disponible sin red a sec.gov.

| Ticker | Accession | Chars | Prosa utilizable | Nota |
|---|---|---|---|---|
| AAPL | 0000320193-25-000079 | 1 | no | regex no ancló el título (`<span><font>`) |
| MSFT | 0000950170-25-100235 | 8000 | **no** | el regex casó la cabecera XBRL oculta (`ix:header`): tokens `us-gaap:*Member`, sin prosa |
| NVDA | 0001045810-26-000021 | 8 | no | `&#160;` en el título |
| GOOGL | 0001652044-26-000018 | 7999 | **sí** | Item 1 — Business: AI, Moonshots, Google Services |
| META | 0001628280-26-003942 | 1 | no | título anidado |
| TSLA | 0001628280-26-003952 | 1 | no | título anidado |

`_index.json` repite esta tabla campo por campo (`usable_prose`, `note`). El accession de AAPL es `0000320193-25-000079` en los tres sitios (JSON, índice y esta tabla); la versión anterior de este README citaba otro por error.

## Cómo se usa

* `src/ingestion/pipeline.load_sample_filings()` carga los excerpts como secciones Item 1, desescapa entidades HTML y descarta los que tienen menos de 500 caracteres o una proporción de prosa < 0,5 (`prose_ratio`). Hoy sólo pasa GOOGL.
* El `RuleBasedExtractor` extrae de GOOGL nueve productos reales (Android, Chrome, Gmail, Google Drive, Google Gemini, Google Maps, Google Photos, Google Play, Search) y los fusiona en el nodo `google` del fixture. `tests/test_pipeline_e2e.py` lo verifica de extremo a extremo (ingesta → grafo → consulta).
* Un párrafo de GOOGL está anotado a mano en `eval/extraction_gold.jsonl` para medir el F1 del extractor.

## Cómo obtener filings completos

```bash
python scripts/download_data.py --years 3 --limit 5      # requiere salida a sec.gov
```

El script usa `data.sec.gov/submissions/CIK##########.json`, respeta 10 req/s, descarga el documento primario del 10-K, el Exhibit 21 (vía `index.json` del filing) y el DEF 14A, y escribe `data/MANIFEST.txt` con SHA-256. El parser robusto (`src/ingestion/parser.py`) resuelve los tres formatos que fallaban aquí; los tests lo comprueban con HTML sintético de cada estilo.
