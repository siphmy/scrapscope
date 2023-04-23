from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol

from . import embedding, scrapbox
from .models import Vector


@dataclass
class Document:
    page_title: str
    content: str


@dataclass
class Hit:
    score: float
    document: Document


class Database(Protocol):
    def query(self, vector: Vector) -> Iterator[Hit]:
        ...

    def index(self, *, vectors: Iterable[Vector], documents: Iterable[Document]):
        ...

    def reset(self, *, dimensions: int):
        ...


@dataclass
class Index:
    db: Database
    model: embedding.Model

    def index(self, project: scrapbox.Project, *, force=False):
        encoder = embedding.LineEncoder(model=self.model, project=project)

        self.db.reset(dimensions=self.model.dimensions)
        self.db.index(
            vectors=encoder.encode(force=force),
            documents=documents(project),
        )

    def query(self, prompt: str) -> Iterator[Hit]:
        vector = next(self.model.encode([prompt]))
        return self.db.query(vector)


def documents(project: scrapbox.Project) -> Iterator[Document]:
    for page in project.pages():
        for line in page.lines():
            yield Document(
                page_title=page.title,
                content=line,
            )
