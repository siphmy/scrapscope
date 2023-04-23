from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional, Protocol, Union

from . import embedding, scrapbox
from .models import Vector


@dataclass
class Document:
    kind: Union[Literal["page"], Literal["line"]]
    page_title: str
    content: Optional[str]


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
        self.db.reset(dimensions=self.model.dimensions)

        line_encoder = embedding.LineEncoder(model=self.model, project=project)
        self.db.index(
            vectors=line_encoder.encode(force=force),
            documents=line_documents(project),
        )

        page_encoder = embedding.PageEncoder(model=self.model, project=project)
        self.db.index(
            vectors=page_encoder.encode(force=force),
            documents=page_documents(project),
        )

    def query(self, prompt: str) -> Iterator[Hit]:
        vector = next(self.model.encode([prompt]))
        return self.db.query(vector)


def line_documents(project: scrapbox.Project) -> Iterator[Document]:
    for page in project.pages():
        for line in page.lines():
            yield Document(
                kind="line",
                page_title=page.title,
                content=line,
            )


def page_documents(project: scrapbox.Project) -> Iterator[Document]:
    for page in project.pages():
        yield Document(
            kind="page",
            page_title=page.title,
            content=None,
        )
