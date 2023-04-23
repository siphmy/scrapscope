from dataclasses import dataclass
from functools import cached_property
from hashlib import file_digest
from typing import Iterator, List

import json_stream
from json_stream.base import TransientStreamingJSONObject


@dataclass
class Page:
    title: str
    raw_lines: List[str]

    def lines(self) -> Iterator[str]:
        "iterates non-empty lines"

        for raw_line in self.raw_lines:
            line = raw_line.strip()
            if line:
                yield line

    def count_lines(self) -> int:
        "counts non-empty lines"

        return sum(1 for _ in self.lines())


@dataclass
class Project:
    file: str
    "path to an exported json file. assumes no metadata is included"

    def pages(self) -> Iterator[Page]:
        with open(self.file) as f:
            data = json_stream.load(f)
            assert isinstance(data, TransientStreamingJSONObject)

            for page in data["pages"]:
                page.persistent()
                yield Page(title=page["title"], raw_lines=page["lines"])

    def count_lines(self) -> int:
        "counts non-empty lines"

        return sum(page.count_lines() for page in self.pages())

    @cached_property
    def hash(self):
        with open(self.file, "rb") as f:
            return file_digest(f, "blake2b").hexdigest()[:16]
