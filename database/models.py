"""
database/models.py

Database models for the AI Assisted Exam Proctoring System.
"""

from __future__ import annotations

from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import String

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from .base import Base


class Alert(Base):
    """
    Stores generated alerts.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    student_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    rule_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    alert_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    created_at: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )


class Evidence(Base):
    """
    Stores evidence clip metadata.
    """

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    student_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    clip_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    created_at: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )