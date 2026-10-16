"""Ascend NPU RAG pipeline with selectable retrieval databases."""

from .config import Settings
from .stores import create_retrieval_store

__all__ = ["Settings", "create_retrieval_store"]
