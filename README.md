# Project 08 — GraphRAG over SEC EDGAR

> Knowledge graph from public 10-K filings of the S&P 500. Answers complex multi-hop questions about the company ecosystem: who supplies whom, who sits on competing boards, which companies share regulatory exposure.

[![Status](https://img.shields.io/badge/status-planned-fbbf24)]()
[![DB](https://img.shields.io/badge/db-Neo4j%205-22d3ee)]()
[![Technique](https://img.shields.io/badge/technique-Microsoft%20GraphRAG-7c5cff)]()

**Industrial use case:** Financial intelligence (Visible Alpha, Tegus, AlphaSense), strategic consulting, M&A.

## What this project does

Ingests 10-K filings (annual reports filed with the SEC) from S&P 500 companies. Extracts entities (companies, persons, subsidiaries, products) and relationships (board memberships, supplier ties, competitive overlap) using Claude. Stores in Neo4j with native vector index for hybrid graph+vector retrieval. Answers questions like "which S&P 500 companies depend on TSMC and what is their combined exposure?"

## Architecture

```
FASE DE INGESTA (one pass over corpus):
   10-K PDF/HTML
   │
   ▼
[Document Chunker] respects 10-K sections (Item 1, Item 1A, etc.)
   │
   ▼
[Entity Extractor] Claude + Pydantic
   → entities: [{id, type, properties}]
   → relationships: [{from, to, type, properties}]
   │
   ▼
[Graph Merger] Neo4j MERGE
   → no duplicates; embedding of name+desc stored as node property
   │
   ▼
[Staleness Tracker] each node has updated_at + source_filing_date

FASE DE CONSULTA:
   Question: "Which S&P 500 companies depend on TSMC?"
   │
   ▼
[Query Classifier] Claude → multi_hop + aggregation
   │
   ▼
[Entry Point Finder] vector search on Neo4j → seed nodes
   │
   ▼
[Graph Traversal] Cypher
   MATCH (tsmc)-[:SUPPLIED_BY]<-(c:Company {in_sp500: true})
   │
   ▼
[Result Aggregator] → ordering, totals, ratios
   │
   ▼
[Synthesizer] Claude → natural-language answer citing nodes
   │
   ▼
[Visualization] subgraph for frontend
```

## Roadmap to v1.0.0

1. [ ] SEC EDGAR API integration — download 10-Ks for top 100 S&P 500 companies, last 3 years
2. [ ] Document chunker that respects 10-K section structure
3. [ ] Entity extraction with Pydantic schemas (`Company`, `Person`, `Product`, `Risk`, `Subsidiary`)
4. [ ] Neo4j with APOC plugin + GDS for community detection
5. [ ] Embedding ingestion (Voyage AI voyage-3 → Neo4j native vector index)
6. [ ] Query classifier + traversal + synthesizer pipeline
7. [ ] Eval set: 100 multi-hop questions across 4 categories with manual ground truth
8. [ ] Comparative table: Vector RAG vs GraphRAG vs Hybrid
9. [ ] Next.js demo with react-force-graph-2d visualization
10. [ ] Staleness detector flags nodes with data >365 days old
11. [ ] Graph schema documented in `docs/graph_schema.md`

## Stack

| Layer | Technology |
|---|---|
| Graph DB | Neo4j 5+ with APOC + GDS plugins |
| Query language | Cypher |
| Vector index | Native in Neo4j 5.13+ |
| LLM | Claude Sonnet 4.5 (entity extraction + synthesis) |
| Schemas | Pydantic v2 |
| Embeddings | Voyage AI `voyage-3` or `bge-large-en-v1.5` |
| Framework | LangChain `neo4j-graphrag` or LlamaIndex KnowledgeGraphIndex |
| Frontend | Next.js + react-force-graph-2d |
| Audit | PostgreSQL for staleness tracking |
| Observability | LangSmith |

## Definition of Done — project-specific

- [ ] Knowledge graph populated with top 100 S&P 500 + 3 years of 10-Ks (≥5K nodes, ≥20K edges)
- [ ] Eval set of 100 multi-hop questions with manual ground truth
- [ ] Vector RAG vs GraphRAG comparison with reproducible numbers
- [ ] Demo with interactive graph visualization
- [ ] Pre-baked queries covering 4 categories (lookup, 2-hop, 3+ hop, aggregation)
- [ ] Staleness detector active
- [ ] Graph schema documented

Plus the 12 universal DoD blocks.

## License

MIT.
