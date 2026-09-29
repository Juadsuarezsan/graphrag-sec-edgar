# Borrador de post para LinkedIn

Un RAG con embeddings responde bien "¿quién es el CEO de Microsoft?" y falla en "¿quién es el CEO de la matriz de LinkedIn?". Ningún pasaje contiene esa respuesta: son dos hechos en dos documentos que hay que encadenar.

Construí GraphRAG sobre SEC EDGAR: un grafo de conocimiento a partir de 10-K, Exhibit 21 y DEF 14A del S&P 500, con traversal multi-hop, staleness por fecha de filing y un baseline de Vector RAG sobre el MISMO corpus para medir qué aporta el grafo.

Lo que medí (100 preguntas con ground truth manual, corrida reproducible con `python -m eval.run`):
• Vector RAG: 86,7 % en lookups → 30 % en dos saltos → 0 % en tres.
• GraphRAG: 100 % en las cuatro categorías; el mismo motor limitado a un salto pierde 42 puntos.
• Extractor por reglas (Exhibit 21, DEF 14A, prosa): F1 95,5 % sobre chunks anotados a mano, incluido un párrafo real del 10-K de Alphabet.

Lo que NO puedo afirmar todavía, y está escrito en el README: el grafo evaluado es un fixture curado de 279 nodos, la síntesis de las corridas guardadas es una plantilla sin LLM y la faithfulness está pendiente de una llave de API. Prefiero un eval set pequeño y reproducible a un número grande que nadie pueda regenerar.

Stack: Python 3.11, FastAPI, Pydantic v2 con tipos cerrados, Claude Sonnet 4.5 vía tool use (mockeado en tests), Neo4j 5 / NetworkX detrás de un Protocol, TF-IDF puro Python como baseline, demo con simulación de fuerzas propia en canvas (sin CDN), mypy --strict, 87 tests, 97 % de cobertura, gitleaks en CI.

Repo: https://github.com/Juadsuarezsan/graphrag-sec-edgar
Write-up completo: docs/blog/post.md

#GraphRAG #KnowledgeGraph #LLM #FinTech #SECEDGAR #Python #Neo4j #AIEngineering
