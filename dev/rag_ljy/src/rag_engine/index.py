"""Compatibility import for the original Elasticsearch-only module path."""

from .stores.elasticsearch import ElasticsearchStore, index_definition

__all__ = ["ElasticsearchStore", "index_definition"]
