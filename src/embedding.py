import os
from dataclasses import dataclass
from os import path
from typing import Callable, Iterable, Iterator, List, Protocol, TypeVar

import numpy as np
from tqdm import tqdm

from . import config, scrapbox
from .logging import logger
from .models import Vector


class Model(Protocol):
    @property
    def identifier(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    def preload(self):
        ...

    def encode(self, sentences: Iterable[str]) -> Iterator[Vector]:
        ...


@dataclass
class LineEncoder:
    model: Model
    project: scrapbox.Project

    def encode(self, *, force=False) -> Iterator[Vector]:
        model_dir = self.model.identifier.replace("/", "_")
        cache_dir = path.join(config.cache_dir, self.project.hash, model_dir)
        cache_file = path.join(cache_dir, "lines.bin")

        shape = (self.project.count_lines(), self.model.dimensions)

        with_memmap = cached_memmapper(file=cache_file, shape=shape, ignore=force)
        return with_memmap(self._embeddings)

    def _embeddings(self):
        sentences = self._enumerate_sentences()
        return self.model.encode(tqdm(sentences))

    def _enumerate_sentences(self) -> Iterator[str]:
        for page in self.project.pages():
            for line in page.lines():
                yield f"{page.title}; {line}"


@dataclass
class PageEncoder:
    model: Model
    project: scrapbox.Project

    def encode(self, *, force=False) -> Iterator[Vector]:
        model_dir = self.model.identifier.replace("/", "_")
        cache_dir = path.join(config.cache_dir, self.project.hash, model_dir)
        cache_file = path.join(cache_dir, "pages.bin")

        shape = (self.project.count_pages(), self.model.dimensions)

        with_memmap = cached_memmapper(file=cache_file, shape=shape, ignore=force)
        return with_memmap(self._embeddings)

    def _embeddings(self):
        sentences = self._enumerate_sentences()
        return self.model.encode(tqdm(sentences))

    def _enumerate_sentences(self) -> Iterator[str]:
        for page in self.project.pages():
            yield " ".join(list(page.lines()))


T = TypeVar("T")


def cached_memmapper(
    *,
    file: str,
    shape: tuple[int, ...],
    ignore: bool = False,
) -> Callable[[Callable[[], Iterable[List[T]]]], Iterator[List[T]]]:
    def with_memmap(fn: Callable[[], Iterable[List[T]]]) -> Iterator[List[T]]:
        dir = path.dirname(file)
        os.makedirs(dir, exist_ok=True)

        if path.exists(file) and not ignore:
            logger.info(f"using cache (path: {file})")
        else:
            if ignore:
                logger.info("ignoring cache")
            else:
                logger.info(f"no cache found (path: {file})")

            mm = np.memmap(file, mode="w+", dtype="float32", shape=shape)
            iter = fn()
            for i, item in enumerate(iter):
                mm[i] = item

            mm.flush()

        mm = np.memmap(file, mode="r", dtype="float32", shape=shape)
        for item in mm:
            yield item.tolist()

    return with_memmap
