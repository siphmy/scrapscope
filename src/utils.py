from itertools import islice
from typing import Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    it = iter(iterable)

    while True:
        batch = list(islice(it, n))
        if not batch:
            return

        yield batch
