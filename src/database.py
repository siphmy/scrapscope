from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Iterable, Iterator

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint
from qdrant_client.models import VectorParams

from . import index
from .models import Vector


@dataclass
class Qdrant:
    host: str
    port: int
    collection: str
    batch_size: int = 256

    @cached_property
    def client(self):
        return QdrantClient(host=self.host, port=self.port)

    def query(self, vector: Vector) -> Iterator[index.Hit]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=5,
        )

        return map(hit_from_qdrant, hits)

    def index(self, *, vectors: Iterable[Vector], documents: Iterable[index.Document]):
        payload = map(asdict, documents)

        self.client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            payload=payload,
            ids=None,
            batch_size=self.batch_size,
            # parallel=2,
        )

    def reset(self, *, dimensions: int):
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimensions, distance="Cosine"),  # type: ignore
        )


def hit_from_qdrant(hit: ScoredPoint) -> index.Hit:
    assert hit.payload is not None
    doc = index.Document(
        kind=hit.payload["kind"],
        page_title=hit.payload["page_title"],
        content=hit.payload["content"],
    )

    return index.Hit(score=hit.score, document=doc)
