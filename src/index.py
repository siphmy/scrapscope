import os
from dataclasses import dataclass
from os import path
from typing import Iterable, Iterator, Protocol

import numpy as np
from tqdm import tqdm

from . import config, embedding, scrapbox
from .embedding import Vector
from .logging import logger


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
        self.db.reset(dimensions=self.model.dimensions)
        self.db.index(
            vectors=self.embeddings(project, force=force),
            documents=documents(project),
        )

    def query(self, prompt: str) -> Iterator[Hit]:
        vector = next(self.model.encode([prompt]))
        return self.db.query(vector)

    def embeddings(self, project: scrapbox.Project, *, force=False) -> Iterator[Vector]:
        model_dir = self.model.identifier.replace("/", "_")
        cache_dir = path.join(config.cache_dir, project.hash, model_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = path.join(cache_dir, "embeddings.bin")

        shape = (project.count_lines(), self.model.dimensions)

        if path.exists(cache_file) and not force:
            logger.info(f"using embeddings from cache (hash: {project.hash})")
        else:
            if force:
                logger.info("ignoring embeddings cache")
            else:
                logger.info(f"no embeddings cache found (hash: {project.hash})")

            sentences = enumerate_sentences(project)
            embeddings = self.model.encode(tqdm(sentences))

            mm = np.memmap(cache_file, mode="w+", dtype="float32", shape=shape)
            for i, embedding in enumerate(embeddings):
                mm[i] = embedding

            mm.flush()

        mm = np.memmap(cache_file, mode="r", dtype="float32", shape=shape)
        for vector in mm:
            yield vector.tolist()


def documents(project: scrapbox.Project) -> Iterator[Document]:
    for page in project.pages():
        for line in page.lines():
            yield Document(
                page_title=page.title,
                content=line,
            )


def enumerate_sentences(project: scrapbox.Project) -> Iterator[str]:
    for page in project.pages():
        for line in page.lines():
            yield f"{page.title}; {line}"
