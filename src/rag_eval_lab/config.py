"""Configuration read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    database_url: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            embed_model=os.getenv("EMBED_MODEL", cls.embed_model),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            database_url=os.getenv("DATABASE_URL", ""),
        )
