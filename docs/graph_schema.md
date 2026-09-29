# Esquema del grafo de conocimiento

Fuente de verdad en código: `src/extraction/schemas.py` (`NodeType`, `EdgeType`, `EDGE_SIGNATURES`) y `src/graph/base.py` (`Node`, `Edge`). El endpoint `GET /api/schema` devuelve exactamente lo que se documenta aquí. Los tipos son cerrados: un extractor (reglas o Claude) no puede introducir un tipo nuevo sin cambiar el código y este documento.

## Propiedades comunes a todo nodo

| Propiedad | Tipo | Descripción |
|---|---|---|
| `id` | `str` (slug `^[a-z0-9_]+$`) | Identificador estable. Empresas: alias corto (`apple`, `jpm`); personas: `nombre_apellido`; subsidiarias: `sub_*`; productos: `prod_*`; riesgos: `risk_*`; mercados: `market_*`. |
| `type` | `NodeType` | Uno de los seis tipos de abajo. En Neo4j se materializa como segunda etiqueta (`:Entity:Company`). |
| `name` | `str` | Nombre canónico legible. |
| `properties` | `dict` | Propiedades específicas del tipo (tabla siguiente) más `aliases: list[str]`, `source_accession`, `stale: bool`. |
| `embedding` | `list[float]` | Vector del embedder configurado (`hashing` 256 d, `tfidf` 1024 d, `voyage-3` 1024 d). |
| `updated_at` | `YYYY-MM-DD` | Fecha del último MERGE. |
| `source_filing_date` | `YYYY-MM-DD \| null` | Fecha del filing que aportó la evidencia. El detector de staleness la prefiere sobre `updated_at`. |

## Tipos de nodo

| Tipo | Propiedades específicas | Origen en los filings |
|---|---|---|
| `Company` | `ticker`, `cik`, `sector`, `in_sp500: bool`, `hq_state`, `description` | Portada del 10-K (`dei:` en XBRL), `data.sec.gov/submissions`. Las contrapartes no S&P (TSMC, ASML, Samsung…) se crean cuando un 10-K las menciona como proveedor o competidor. |
| `Person` | `role` (`CEO` \| `Director`), `company`, `since`, `age`, `director_since`, `synthetic: bool` | Item 10 del 10-K y DEF 14A (tabla de nominados). En el fixture los directores son sintéticos y llevan `synthetic=True`. |
| `Subsidiary` | `jurisdiction`, `parent`, `ownership_pct` | Exhibit 21 ("Subsidiaries of the Registrant"). |
| `Product` | `category`, `company` | Item 1 ("our products include …"). |
| `Risk` | — (los alias definen el léxico) | Item 1A. Trece temas normalizados (`risk_china_exposure`, `risk_supply_concentration`, `risk_climate_regulation`, `risk_interest_rate`, `risk_cybersecurity`, `risk_ai_regulation`, `risk_antitrust`, `risk_drug_pricing`, `risk_patent_expiry`, `risk_fx`, `risk_tariffs`, `risk_talent`, `risk_commodity_prices`). |
| `Market` | — | Item 1 (segmentos y mercados declarados). |

## Tipos de relación

Cada relación tiene una firma `(tipo origen → tipo destino)` que `ExtractionResult` valida; una relación con tipos incompatibles se rechaza con `ValidationError`.

| Relación | Firma | Propiedades | Semántica |
|---|---|---|---|
| `CEO_OF` | Person → Company | `since` | La persona es el CEO de la empresa en la fecha del filing. |
| `DIRECTOR_OF` | Person → Company | `since`, `source` | La persona es director/miembro del consejo. |
| `SUBSIDIARY_OF` | Subsidiary → Company | `source` (`Exhibit 21`), `ownership_pct` | Entidad controlada listada en Exhibit 21. |
| `SUPPLIED_BY` | Company → Company | `component`, `evidence` | El origen compra al destino (`apple -SUPPLIED_BY-> tsmc`). Dirección relevante: "clientes de X" son las aristas entrantes a X. |
| `COMPETES_WITH` | Company → Company | `evidence` | Competencia declarada. Se almacena una vez y se recorre en ambos sentidos. |
| `MENTIONS_RISK` | Company → Risk | `section`, `cue`, `evidence` | El 10-K menciona el tema de riesgo (normalmente en Item 1A). |
| `OFFERS_PRODUCT` | Company → Product | `evidence` | Producto o plataforma ofrecida. |
| `OPERATES_IN` | Company → Market | — | Mercado/segmento en el que opera. |

Toda arista puede llevar `source_accession` (accession del filing) y `evidence` (oración de soporte, ≤500 caracteres) cuando proviene de una extracción.

## Semántica de MERGE

* Nodo: clave `id`. Las `properties` se fusionan superficialmente (las nuevas pisan a las viejas), el embedding se reemplaza sólo si se aporta uno y `source_filing_date` conserva el valor existente si el nuevo es `null`.
* Arista: clave `(from_id, to_id, type)`. Un segundo MERGE actualiza propiedades, nunca duplica.
* Idéntico en `InMemoryGraph`, `NetworkXGraphStore` y `Neo4jGraphStore` (tests de contrato en `tests/test_graph_store.py` y `tests/test_neo4j_store.py`).

## Cypher de referencia (Neo4j 5.13+)

```cypher
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 256, `vector.similarity_function`: 'cosine'}};

// MERGE de nodo (las propiedades específicas se aplanan con prefijo p_)
MERGE (n:Entity {id: $id})
SET n:Company, n.type = $type, n.name = $name, n += $props,
    n.updated_at = $updated_at,
    n.source_filing_date = coalesce($sfd, n.source_filing_date);

// ¿Qué empresas del S&P 500 dependen de TSMC?
MATCH (c:Company)-[:SUPPLIED_BY]->(t:Company {id: 'tsmc'}) WHERE c.p_in_sp500 RETURN c.name;

// Tres saltos: directores de competidores de clientes de TSMC
MATCH (t:Company {id:'tsmc'})<-[:SUPPLIED_BY]-(c)-[:COMPETES_WITH]-(k)<-[:DIRECTOR_OF]-(d:Person)
RETURN DISTINCT d.name, k.name;
```

## Tamaño actual y objetivo

| | Fixture + excerpt real (este repo) | Objetivo DoD (pendiente de `ANTHROPIC_API_KEY` y red a sec.gov) |
|---|---|---|
| Nodos | 279 (42 Company, 54 Person, 77 Subsidiary, 78 Product, 13 Risk, 15 Market) | ≥ 5 000 |
| Aristas | 451 | ≥ 20 000 |
| Fuente | Tablas curadas a mano en `src/graph/fixture.py` + Item 1 real de Alphabet | 100 empresas × 3 años de 10-K + Exhibit 21 + DEF 14A extraídos con `ClaudeExtractor` |

Los conteos exactos se leen con `python -c "from src.graph.fixture import fixture_summary; print(fixture_summary())"` y en `GET /health`.
