# Datos: fuentes, esquema y alcance

## Fuente original

**SEC EDGAR** — <https://www.sec.gov/edgar/search-and-access>. Los filings son documentos públicos del gobierno de EE. UU. sin restricciones de copyright (dominio público). Condiciones de acceso: `User-Agent` con nombre y correo de contacto y máximo 10 solicitudes por segundo (<https://www.sec.gov/os/accessing-edgar-data>). El script `scripts/download_data.py` cumple ambas y reintenta con backoff ante 429/5xx.

Formularios usados:

| Formulario | Qué aporta al grafo | Sección |
|---|---|---|
| 10-K | Company, Product, Risk, Market, SUPPLIED_BY, COMPETES_WITH, MENTIONS_RISK, OFFERS_PRODUCT, OPERATES_IN | Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A) |
| Exhibit 21 del 10-K | Subsidiary, SUBSIDIARY_OF | Tabla "Subsidiaries of the Registrant" |
| DEF 14A | Person (Director), DIRECTOR_OF, CEO_OF | Tabla de nominados / biografías |

## Lo que hay en este repositorio

| Ruta | Contenido | Estado |
|---|---|---|
| `data/sp500_top100.csv` | `rank,ticker,cik,name` de las 100 mayores empresas del S&P 500 por capitalización aproximada (2025). Es la lista que consume `download_data.py`. | commiteado |
| `data/sec_edgar_sample/*_10K.json` | Seis excerpts descargados de EDGAR con el script anterior (regex). Sólo **GOOGL** contiene prosa utilizable del Item 1 (7 999 caracteres, accession `0001652044-26-000018`). **MSFT** tiene 8 000 caracteres pero son la cabecera XBRL oculta (`ix:header`), no prosa; AAPL, NVDA, META y TSLA tienen 1–8 caracteres porque el regex no ancló el título. `_index.json` lo declara campo por campo (`usable_prose`, `note`). | commiteado, evidencia |
| `data/MANIFEST.txt` | SHA-256 de cada archivo de datos commiteado. | commiteado |
| `data/raw/` | Salida de `download_data.py` (HTML de 10-K, Exhibit 21, DEF 14A). | en `.gitignore`; pendiente de red a sec.gov |
| `data/processed/graph.json` | Grafo persistido por `NetworkXGraphStore` (`GRAPH_BACKEND=networkx`). | en `.gitignore`; se regenera al arrancar |

### Esquema de `*_10K.json`

| Campo | Tipo | Descripción |
|---|---|---|
| `ticker` | str | Símbolo bursátil. |
| `cik` | str (10 dígitos) | Central Index Key. |
| `company_name` | str | Nombre en EDGAR. |
| `accession` | `##########-##-######` | Número de accession del 10-K. |
| `source_url` | URL | Documento primario en `sec.gov/Archives`. |
| `business_section_excerpt` | str | Texto del Item 1 (hasta 8 000 caracteres). Puede contener entidades HTML (`&#8217;`) que el loader desescapa. |
| `business_section_chars` | int | Longitud del excerpt. |

### Esquema de `data/sp500_top100.csv`

| Columna | Tipo | Rango |
|---|---|---|
| `rank` | int | 1–100 |
| `ticker` | str | símbolo (con `.` para clases, p. ej. `BRK.B`) |
| `cik` | str | 10 dígitos con ceros a la izquierda |
| `name` | str | razón social |

### Esquema de `data/MANIFEST.txt`

Una línea por archivo: `sha256  ruta_relativa  bytes  form  accession  filing_date` (los tres últimos campos son `-` para archivos que no son filings). Se regenera con `python scripts/build_manifest.py`.

## El fixture curado a mano (`src/graph/fixture.py`)

Como sec.gov no es alcanzable desde el entorno de desarrollo y no hay `ANTHROPIC_API_KEY`, el grafo evaluado se construye a partir de tablas escritas a mano:

* **42 empresas** (36 del S&P 500 y 6 contrapartes: TSMC, Samsung, ASML, Foxconn, SK hynix, Panasonic) con ticker, CIK, sector y fecha aproximada de su último 10-K.
* **38 CEOs** reales según los 10-K/DEF 14A de 2025-2026 (p. ej. Lip-Bu Tan en Intel desde 2025, Stephen Hemsley en UnitedHealth desde 2025).
* **16 directores sintéticos** (`properties.synthetic=True`) con asientos cruzados deliberados. Sustituyen la información de DEF 14A que el extractor por reglas produce cuando hay proxies reales; no representan a personas reales.
* **77 subsidiarias** tipo Exhibit 21, **79 productos**, **13 riesgos**, **15 mercados**, **31 relaciones proveedor-cliente** y **30 pares de competidores** tomados de conocimiento público de los 10-K.

Este fixture es una **muestra de trabajo declarada**, no el resultado de una extracción sobre el corpus. Los hit rates de `eval/RESULTS.md` miden la recuperación estructural sobre este grafo; la extracción real sobre 300 filings queda como ítem [B] del plan.

## Eval sets

| Archivo | Tamaño | Cómo se construyó |
|---|---|---|
| `eval/gold.jsonl` | 100 preguntas: 30 lookup / 30 dos saltos / 20 tres+ saltos / 20 aggregation | Escritas a mano sobre las tablas del fixture (`scripts/build_gold.py` sólo serializa). Cada ítem lleva `expected_ids` (y `expected_count` en aggregation), `hops`, `notes` y `ground_truth="manual (fixture tables)"`. `tests/test_evaluation.py` verifica que todo id exista en el grafo y que los conteos coincidan con las listas. |
| `eval/extraction_gold.jsonl` | 5 chunks anotados a mano | Exhibit 21 y DEF 14A sintéticos, Item 1 e Item 1A sintéticos y un párrafo **real** del 10-K de Alphabet (Google Services) con sus 10 productos anotados. |

### Esquema de `eval/gold.jsonl`

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | `lk-NN`, `2h-NN`, `3h-NN`, `ag-NN` |
| `category` | `lookup` \| `two_hop` \| `three_hop` \| `aggregation` | Categoría del spec |
| `question` | str | Pregunta en inglés |
| `expected_ids` | list[str] | Ids de nodo que deben aparecer en la respuesta |
| `expected_count` | int \| null | Conteo exacto (sólo aggregation) |
| `expected_text` | str \| null | Texto que debe aparecer (lookups de propiedad) |
| `hops` | int 0–4 | Saltos necesarios |
| `notes`, `ground_truth` | str | Trazabilidad |

## Splits y semillas

No hay entrenamiento: el sistema es determinista y el eval set completo es de test. `tests/conftest.py` fija `random.seed(20260516)` por si algún componente futuro introduce aleatoriedad.
