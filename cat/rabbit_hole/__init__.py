"""The RabbitHole: the ingestion of the sources (files and URLs) in the memory of the agents."""
from cat.rabbit_hole.core import RabbitHole, run_in_ingestion_executor

__all__ = ["RabbitHole", "run_in_ingestion_executor"]
