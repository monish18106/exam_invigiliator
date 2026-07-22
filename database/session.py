"""
database/session.py

SQLAlchemy engine and session management.
"""

from __future__ import annotations

from config import settings
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker


# PostgreSQL engine
engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

from .base import Base
from . import models

# Create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.

    Ensures every session is properly closed.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()