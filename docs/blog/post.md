# Por qué un grafo y no otro índice vectorial: GraphRAG sobre los 10-K de la SEC

*Borrador para Medium / Dev.to. Autor: Juan David Suárez Sánchez. Código: <https://github.com/Juadsuarezsan/graphrag-sec-edgar>.*

Todos los años, cada empresa cotizada en Estados Unidos deposita en la SEC un informe anual —el 10-K— de cien a trescientas páginas en el que describe su negocio, sus riesgos, sus proveedores críticos, sus competidores y sus filiales. Es la fuente primaria más rica y más aburrida del mundo financiero. Empresas como AlphaSense, Tegus o Visible Alpha cobran cientos de millones de dólares por convertir esos documentos en respuestas. Este artículo cuenta cómo construí un sistema que hace una parte de ese trabajo, qué decisiones tomé, qué medí y —sobre todo— qué no puedo afirmar todavía.

## El problema con las preguntas de dos saltos

La receta estándar de RAG funciona así: se parte el documento en pasajes, se calcula un embedding de cada uno, se recuperan los más parecidos a la pregunta y se le pide al modelo que responda con ellos. Para "¿quién es el CEO de Microsoft?" funciona de maravilla: el pasaje que contiene la frase "Satya Nadella, Chairman and Chief Executive Officer" está a un coseno de distancia.

Ahora prueben con "¿quién es el CEO de la empresa matriz de LinkedIn?". Ningún pasaje de ningún 10-K contiene esa respuesta. El Exhibit 21 de Microsoft dice que LinkedIn Corporation es una filial en Delaware; el Item 10 dice quién es el CEO de Microsoft. Son dos hechos en dos documentos distintos, y el segundo sólo se vuelve relevante después de haber leído el primero. La recuperación por similitud no encadena: recupera lo que se parece a la pregunta, y la pregunta se parece a "LinkedIn", no a "Satya Nadella".

Si el sistema tiene que responder "¿qué directores sirven en consejos de competidores directos de una empresa que depende de TSMC?", los saltos son tres y la probabilidad de que un top-k de pasajes contenga toda la cadena es prácticamente cero.

La solución es antigua: un grafo. Si extraigo los hechos una vez —Microsoft *es dueña de* LinkedIn, Nadella *es CEO de* Microsoft— y los guardo como aristas tipadas, la pregunta se convierte en un camino: `LinkedIn -SUBSIDIARY_OF-> Microsoft <-CEO_OF- Nadella`. Dos saltos, dos milisegundos, y la respuesta trae consigo la prueba.

## Lo que construí

El repositorio implementa las dos fases completas.

**Ingesta.** Un script descarga de EDGAR los 10-K de tres años, el Exhibit 21 y el DEF 14A (el proxy statement con la lista de directores) para las cien mayores empresas del S&P 500, respetando las condiciones de la SEC (User-Agent con contacto, diez peticiones por segundo). Un parser recorre el DOM con lxml y hace tres cosas que el regex que tenía antes no hacía: descarta la cabecera XBRL oculta (que en el 10-K de Microsoft ocupa los primeros ocho mil caracteres de "texto" sin una sola oración), mantiene los títulos partidos en `<span><font>Item</font><font> 1.</font></span>` en una sola línea y trata `&nbsp;` como un espacio. Después localiza los Items 1, 1A y 7, distinguiendo la mención del índice de la sección real (la real es la que tiene cuerpo largo), y corta en chunks de dos mil caracteres en fin de oración.

Cada chunk pasa por un extractor. Para Exhibit 21 y DEF 14A —que son tablas— uso reglas deterministas: nombre y jurisdicción de cada filial, nombre y edad y año de cada director. Para la prosa hay dos extractores detrás de la misma interfaz: uno de reglas, conservador, que dispara sólo con frases explícitas ("compete with", "supplied by", "products include") y un léxico de riesgos; y `ClaudeExtractor`, que llama a `claude-sonnet-4-5-20250929` obligándolo a responder a través de una herramienta cuyo esquema JSON se genera de los modelos Pydantic. Los tipos de nodo y de arista son literales cerrados: si el modelo devuelve `ACQUIRED_BY`, la validación falla y el chunk se descarta en vez de contaminar el grafo. El SDK lleva timeout explícito y `tenacity` reintenta con backoff exponencial sólo ante errores de red, 429 y 5xx.

Los resultados se funden en el grafo con semántica MERGE: un nodo por id, una arista por `(origen, destino, tipo)`, y cada uno con `source_filing_date` y `updated_at`. Un detector marca `stale=True` cuando la evidencia tiene más de 365 días; la API lo expone en `/api/stale`.

El grafo vive detrás de un `Protocol` con tres implementaciones: en memoria, NetworkX con persistencia a JSON, y Neo4j 5 con MERGE en Cypher, índice vectorial nativo y traversal de longitud variable. Los tests de contrato garantizan que las tres hacen lo mismo; Neo4j se prueba contra un driver falso porque en el entorno de desarrollo no había demonio Docker.

**Consulta.** Aquí tomé la decisión menos obvia: no hay LLM en el camino de la pregunta a los nodos. Un *entity linker* encuentra las entidades mencionadas por alias (el más largo primero, con posesivos y plurales), y un *planner* de reglas convierte las palabras de relación en pasos: "CEO" es `CEO_OF`, "supplied by" es `SUPPLIED_BY` entrante, "competitors" es `COMPETES_WITH` en ambos sentidos. Los cues que preceden a una entidad se aplican en orden inverso ("CEO of the parent of LinkedIn" = LinkedIn → padre → CEO); los que la siguen, en orden directo ("Apple's suppliers"). Si hay dos entidades, sus conjuntos se intersectan ("directors of both Pfizer and Merck"). "Share a supplier" es ida y vuelta por la misma relación. "How many" cuenta el frontier final. La dirección de cada paso no la adivina el texto: la resuelven las firmas del esquema (`CEO_OF` va de Person a Company; si el frontier son empresas, la arista se recorre hacia atrás).

El plan se devuelve en la respuesta. Cuando el sistema contesta "Jensen Huang, Lisa Su, Cristiano Amon…" a la pregunta de los CEOs que comparten proveedor con Apple, el cliente ve `SUPPLIED_BY (out) → SUPPLIED_BY (in) → CEO_OF (in)`, las 35 aristas recorridas y el subgrafo para dibujarlo. Eso es lo que un analista puede auditar; un párrafo fluido sin camino, no.

La síntesis en lenguaje natural con citas la hace `ClaudeSynthesizer` cuando hay llave. Sin llave, una plantilla determinista lista los nodos con sus ids y las aristas usadas, y lo declara en cada respuesta.

## Medir contra el mismo corpus

Comparar GraphRAG con "Vector RAG" es fácil de hacer mal: basta darle al baseline peores datos. Para evitarlo, el baseline vectorial indexa exactamente los mismos hechos que el grafo: cada nodo se convierte en un pasaje que incluye sus propiedades y una oración por arista ("Apple Inc. is supplied by Taiwan Semiconductor…"). Se recuperan los cinco pasajes más parecidos (TF-IDF con hashing, puro Python, reproducible en cualquier máquina) y la respuesta son las entidades nombradas en ellos. Es una versión generosa del baseline: recibe crédito por cualquier entidad que aparezca en los pasajes, sin exigir que las combine.

El eval set son cien preguntas escritas a mano —treinta de un salto, treinta de dos, veinte de tres o más y veinte de conteo— con sus respuestas verificadas sobre las tablas del grafo. Los resultados, generados por `python -m eval.run` y guardados en el repositorio:

| Sistema | Lookup | 2 saltos | 3 saltos | Aggregation |
|---|---|---|---|---|
| Vector RAG (mismo corpus) | 86,7 % | 30,0 % | 0,0 % | 35,0 % |
| GraphRAG | 100 % | 100 % | 100 % | 100 % |
| Hybrid | 100 % | 100 % | 100 % | 85,0 % |
| GraphRAG limitado a un salto (ablación) | 100 % | 30,0 % | 0,0 % | 95,0 % |

La columna que importa no es la de GraphRAG; es la caída del baseline de 86,7 % a 30 % a 0 % conforme crecen los saltos, y la ablación que muestra que el mismo motor sin traversal multi-hop pierde 42 puntos. Ese es el argumento cuantitativo a favor del grafo. El extractor por reglas alcanza un F1 de 95,5 % en entidades y relaciones sobre cinco chunks anotados a mano, uno de ellos un párrafo real del 10-K de Alphabet del que recupera nueve de sus diez productos.

## Lo que ese 100 % no dice

Aquí es donde hay que ser honesto, porque un 100 % en una tabla invita a dejar de leer.

El grafo evaluado tiene 279 nodos y 451 aristas. No es el resultado de extraer trescientos filings: es un *fixture* curado a mano —42 empresas, sus CEOs reales, filiales del Exhibit 21, productos, trece temas de riesgo, mercados, 31 relaciones de suministro y 30 pares de competidores— más el único excerpt real con prosa que tenía a mano. Los directores son sintéticos y están marcados como tales, porque inventar asientos de consejo de personas reales sería peor que no tenerlos. La red donde desarrollé bloqueaba `sec.gov` y no tenía llave de Anthropic; podía escribir y probar todo el código de descarga y extracción con mocks, pero no ejecutarlo.

En consecuencia, el gold se escribió sobre el mismo grafo que se consulta y con el sub-lenguaje que el planner soporta. El 100 % demuestra que el planner y las tablas no tienen bugs en esas cien preguntas y que el traversal resuelve lo que el vector no puede. No demuestra que el sistema completo, con extracción por LLM sobre 10-K reales, alcance esa cifra; ahí aparecerán entidades duplicadas por variantes de nombre, relaciones mal tipadas y preguntas fuera del sub-lenguaje. La columna de *faithfulness* —¿el texto sintetizado dice sólo lo que el subgrafo sostiene?— está vacía y marcada como pendiente, con la rúbrica del juez escrita en código para cuando haya llave.

Prefiero un eval set pequeño que cualquiera pueda regenerar con un comando a uno grande que nadie pueda reproducir. Las métricas que están en el README son exactamente las que produce `make eval` hoy.

## Decisiones que defendería en una entrevista

*Planner de reglas en vez de text-to-Cypher.* Un LLM que escribe Cypher es más flexible y menos predecible: cada consulta cuesta tokens y segundos, hay que defenderse de inyección y el plan no es inspeccionable. El planner responde en menos de un milisegundo, devuelve el plan y falla explícitamente (plan vacío) cuando no entiende. Text-to-Cypher queda como *fallback* para preguntas fuera del sub-lenguaje.

*Reglas para los exhibits, LLM para la prosa.* Exhibit 21 y la tabla de nominados son estructurados. Para cien empresas y tres años son unos seiscientos documentos que no necesitan pasar por un modelo. Las reglas cuestan cero y dan F1 de 1,0 en el gold; el modelo se reserva para donde aporta recall.

*Sin torch en la instalación base.* `sentence-transformers` arrastra seis gigas de dependencias. Los embeddings locales viven en un extra `[ml]` con import perezoso; por defecto hay un embedder por hashing (SHA-256, determinista) suficiente como punto de entrada de respaldo, y Voyage `voyage-3` cableado por HTTP. La instalación pasó de minutos a segundos y CI no descarga modelos.

*Staleness en el propio nodo.* Dos fechas y un flag resuelven el requisito; una tabla de auditoría en Postgres duplicaba el estado.

## Qué costaría en producción

Con los precios del modelo pinneado, ingerir los 300 filings de las 100 empresas (unos 9 000 chunks) cuesta alrededor de 73 dólares y media hora; los exhibits y proxies, nada. Servir mil usuarios que hagan veinte preguntas al mes con síntesis por Claude son unos 150 dólares mensuales; sin síntesis, el coste es sólo el de una réplica de FastAPI que responde en milisegundos. El detalle, con supuestos explícitos, está en `docs/scalability.md`.

## Qué sigue

Ejecutar la descarga y la extracción reales hasta los 5 000 nodos y 20 000 aristas del objetivo, medir el F1 del extractor Claude sobre un gold ampliado, activar la síntesis y el juez de faithfulness, validar Neo4j con Docker y publicar la demo. Todo el código para hacerlo existe y está probado con mocks; lo que falta son credenciales y red, no ingeniería.

Si trabajan con documentos donde las respuestas importantes están repartidas entre varios lugares —contratos, historiales clínicos, expedientes regulatorios— la lección es la misma: extraigan los hechos una vez, guárdenlos como grafo tipado, y dejen que la recuperación por similitud haga lo único que hace bien, que es encontrar el punto de entrada.
