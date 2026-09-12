"""Local embedding model wrapped as a LangChain Embeddings implementation.

FastEmbed runs a quantised ONNX sentence-transformer on CPU, so semantic search
works without an API key or GPU. The model is downloaded once and cached by
fastembed under the user cache directory.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings backed by ``fastembed.TextEmbedding``.

    Uses the asymmetric prefixes recommended for BGE models: passages are
    embedded as-is, queries are prefixed with a short instruction, which
    measurably improves retrieval quality for question-style inputs.
    """

    query_instruction = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = _load_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self._model.embed(list(texts), batch_size=self.batch_size)
        ]

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{self.query_instruction}{text}" if "bge" in self.model_name.lower() else text
        return next(iter(self._model.embed([prefixed]))).tolist()


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from fastembed import TextEmbedding  # imported lazily: heavy ONNX runtime

    return TextEmbedding(model_name=model_name)
