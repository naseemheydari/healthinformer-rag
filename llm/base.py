"""Abstract base class for LLM clients.

All LLM backends (Groq, Bedrock, etc.) inherit from BaseLLM so the
pipeline can swap models with a config change, not a code change.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Common interface for language model backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user message / main prompt text.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0 = deterministic).
            system_prompt: Optional system-level instruction.

        Returns:
            The generated text as a string.
        """
        ...
