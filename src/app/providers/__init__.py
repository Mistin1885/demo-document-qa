"""Provider abstractions and concrete adapters.

Public re-exports
-----------------
- Abstract base classes: ``ChatProvider``, ``EmbeddingProvider``, ``RerankerProvider``
- Shared data models: ``ChatMessage``, ``ChatCompletion``, ``ChatChunk``,
  ``Usage``, ``ProviderTestResult``
- Registry factories: ``build_chat_provider``, ``build_embedding_provider``,
  ``build_reranker_provider``
- Registry protocol: ``ProviderProfileLike``
"""

from app.providers.base import (
    ChatChunk as ChatChunk,
)
from app.providers.base import (
    ChatCompletion as ChatCompletion,
)
from app.providers.base import (
    ChatMessage as ChatMessage,
)
from app.providers.base import (
    ChatProvider as ChatProvider,
)
from app.providers.base import (
    EmbeddingProvider as EmbeddingProvider,
)
from app.providers.base import (
    ProviderTestResult as ProviderTestResult,
)
from app.providers.base import (
    RerankerProvider as RerankerProvider,
)
from app.providers.base import (
    Usage as Usage,
)
from app.providers.registry import (
    ProviderProfileLike as ProviderProfileLike,
)
from app.providers.registry import (
    build_chat_provider as build_chat_provider,
)
from app.providers.registry import (
    build_embedding_provider as build_embedding_provider,
)
from app.providers.registry import (
    build_reranker_provider as build_reranker_provider,
)

__all__ = [
    # ABCs
    "ChatProvider",
    "EmbeddingProvider",
    "RerankerProvider",
    # Data models
    "ChatMessage",
    "ChatCompletion",
    "ChatChunk",
    "Usage",
    "ProviderTestResult",
    # Registry
    "ProviderProfileLike",
    "build_chat_provider",
    "build_embedding_provider",
    "build_reranker_provider",
]
