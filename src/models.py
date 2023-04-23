from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Protocol

from sentence_transformers import SentenceTransformer  # type: ignore

from .logging import logger
from .utils import batched

Vector = List[float]


class STModelManager:
    def __init__(self) -> None:
        self.models = dict()

    def load(self, name: str) -> Any:
        if name not in self.models:
            logger.info(f"loading sentence transformer model: {name}")
            self.models[name] = SentenceTransformer(name)

        return self.models[name]


st_models = STModelManager()


def sentence_transformer_model(name: str):
    @dataclass
    class STModel:
        batch_size: int = 64

        @property
        def model(self):
            return st_models.load(name)

        @property
        def identifier(self) -> str:
            return name

        @property
        def dimensions(self) -> int:
            return self.model.get_sentence_embedding_dimension()

        def encode(self, sentences: Iterable[str]) -> Iterator[Vector]:
            for batch in batched(sentences, self.batch_size):
                for embedding in self.model.encode(batch):
                    yield embedding

    return STModel


STMultiQaMiniLmL6CosV1 = sentence_transformer_model(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

STDistiluseBaseMultilingualCasedV2 = sentence_transformer_model(
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
)

STParaphraseMultilingualMiniLmL12V2 = sentence_transformer_model(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

STParaphraseMultilingualMpnetBaseV2 = sentence_transformer_model(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
