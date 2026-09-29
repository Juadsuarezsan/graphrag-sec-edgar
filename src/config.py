"""Environment-driven configuration.

Every knob the system exposes is declared here and read from environment
variables (or a local ``.env`` file). Nothing else in ``src/`` reads
``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

GraphBackend = Literal["memory", "networkx", "neo4j"]
EmbedderName = Literal["hashing", "tfidf", "voyage", "local"]

#: Dated model id used by the extractor and the synthesizer. Never an alias.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


class Settings(BaseSettings):
    """Runtime settings.

    Attributes:
        anthropic_api_key: Key for Claude; ``None`` keeps the system in
            deterministic mode (rule-based extraction, template synthesis).
        anthropic_model: Dated Claude model id.
        voyage_api_key: Key for Voyage AI embeddings (optional).
        voyage_model: Voyage embedding model id.
        embedder: Which embedder to build (``hashing`` needs nothing).
        graph_backend: Where the knowledge graph lives.
        graph_path: JSON file used by the ``networkx`` backend.
        neo4j_uri: Bolt URI of the Neo4j instance.
        neo4j_user: Neo4j user.
        neo4j_password: Neo4j password.
        cors_origins: Comma-separated list of allowed origins.
        rate_limit: slowapi rate string applied to ``/api/query``.
        stale_after_days: Age after which a node is flagged as stale.
        request_timeout_s: Timeout for every outbound HTTP call.
        sec_user_agent: User-Agent required by SEC EDGAR.
        langsmith_api_key: LangSmith key; tracing is wired only when set.
        langsmith_project: LangSmith project name.
        langsmith_endpoint: LangSmith ingestion endpoint.
        log_level: loguru level.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL, alias="ANTHROPIC_MODEL")
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-3", alias="VOYAGE_MODEL")
    embedder: EmbedderName = Field(default="hashing", alias="EMBEDDER")

    graph_backend: GraphBackend = Field(default="memory", alias="GRAPH_BACKEND")
    graph_path: str = Field(default="data/processed/graph.json", alias="GRAPH_PATH")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="neo4j_dev", alias="NEO4J_PASSWORD")

    cors_origins: str = Field(default="http://localhost:8000", alias="CORS_ORIGINS")
    rate_limit: str = Field(default="30/minute", alias="RATE_LIMIT")
    stale_after_days: int = Field(default=365, alias="STALE_AFTER_DAYS")
    request_timeout_s: float = Field(default=30.0, alias="REQUEST_TIMEOUT_S")
    sec_user_agent: str = Field(
        default="graphrag-sec-edgar juadsuarezsan@unal.edu.co", alias="SEC_USER_AGENT"
    )

    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="graphrag-sec-edgar", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed ``CORS_ORIGINS``; never contains ``*`` unless set explicitly."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings (tests call ``cache_clear``)."""
    return Settings()
