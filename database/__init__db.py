"""
database/init_db.py

Database initialization.
"""

from .base import Base
from .session import engine

# Import all models so SQLAlchemy registers them.
from . import models  # noqa: F401


def initialize_database() -> None:
    """
    Create all database tables.

    For the MVP we use SQLAlchemy's metadata.
    Later this can be replaced by Alembic migrations.
    """

    Base.metadata.create_all(bind=engine)