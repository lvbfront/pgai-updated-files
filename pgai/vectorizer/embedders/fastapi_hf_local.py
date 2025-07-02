from collections.abc import AsyncGenerator, Sequence
from typing import Literal

from pydantic import BaseModel
from typing_extensions import override

from ..embeddings import Embedder, EmbeddingResponse, EmbeddingVector, Usage, logger

class FastAPIHFLocal(BaseModel, Embedder):
    """
    Embedder that uses a FastAPI-based Hugging Face local server to embed documents into vector representations.

    Attributes:
        implementation (Literal["fastapi_hf_embed"]): The literal identifier for this implementation.
        url (str): The URL of the FastAPI server.
        dimensions (int): The dimensionality of the embeddings.
    """

    implementation: Literal["fastapi_hf_embed"]
    url: str
    dimensions: int

    @override
    async def embed(
        self, documents: list[str]
    ) -> AsyncGenerator[list[EmbeddingVector], None]:
        """
        This method is not used as the embedding logic is handled in vectorizer.py.
        """
        raise NotImplementedError("Embedding is handled in vectorizer.py for fastapi_hf_embed")

    @override
    def _max_chunks_per_batch(self) -> int:
        # Arbitrary default, can be tuned if needed
        return 2048

    @override
    async def setup(self):
        # No setup required for FastAPIHFLocal
        pass

    @override
    async def call_embed_api(self, documents: list[str]) -> EmbeddingResponse:
        # Not used, but required by the interface
        raise NotImplementedError("Embedding is handled in vectorizer.py for fastapi_hf_embed")

    async def _context_length(self) -> int | None:
        # Arbitrary value; can be adjusted if needed
        return None