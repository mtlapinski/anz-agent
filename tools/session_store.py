from __future__ import annotations
import db
from llm import ModelConfig


class SessionNotFound(Exception):
    pass


def create_session(thread_id: str, config: ModelConfig) -> None:
    with db.get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (thread_id, provider, model) VALUES (%s, %s, %s)",
            (thread_id, config.provider, config.model),
        )
        conn.commit()


def get_session(thread_id: str) -> ModelConfig:
    """Raises SessionNotFound if no row exists for thread_id. Any other
    exception (e.g. a connection error) propagates unchanged — a session
    lookup FAILURE must not be indistinguishable from a legitimate MISS."""
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT provider, model FROM sessions WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    if row is None:
        raise SessionNotFound(thread_id)
    return ModelConfig(provider=row[0], model=row[1])
