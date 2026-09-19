"""Generator: query + retrieved chunks → cited answer from an LLM.

Builds a prompt from the system prompt template and retrieved context,
calls the selected LLM backend, and returns the response.

Usage:
    from pipeline.generator import Generator
    from llm.groq_client import GroqLLM

    gen = Generator(llm=GroqLLM())
    answer = gen.generate(query="...", retrieved_chunks=[...])
"""

from typing import Any

from config.settings import PROMPTS_DIR
from llm.base import BaseLLM


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8").strip()


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM.

    Each chunk gets a source number [N] with its metadata and text.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        authors = meta.get("authors", "Unknown")
        title = meta.get("title", "Untitled")
        journal = meta.get("journal", "")
        year = meta.get("year", "")
        url = meta.get("url", "")

        header = f'[{i}] {authors}. "{title}." {journal}, {year}. {url}'
        text = chunk["document"]
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


# Llama 3.3 over-interprets the "say so honestly" clause in the system
# prompt and refuses to answer ~65% of questions even when the retrieved
# context contains relevant (if imperfect) information.  This supplement
# nudges it to synthesise from what is available without removing the
# honest-uncertainty clause that works well for Claude Haiku.
_LLAMA_SYNTHESIS_SUPPLEMENT = (
    "\n\nIMPORTANT: The context passages above were retrieved from a "
    "medical research corpus and are likely relevant to the question. "
    "Even if the passages do not perfectly match the question, extract "
    "and synthesize whatever relevant information they contain. Only "
    "say the context is insufficient if truly none of the passages "
    "relate to the topic at all. Partial answers with citations are "
    "far more helpful than refusals."
)


class Generator:
    """Builds a prompt from context and query, calls the LLM."""

    def __init__(self, llm: BaseLLM, model_name: str = "") -> None:
        """Initialize with an LLM backend.

        Args:
            llm: Any BaseLLM implementation (GroqLLM, BedrockLLM, etc.).
            model_name: Backend name (e.g. "bedrock-llama") used to
                select model-specific prompt adjustments.
        """
        self.llm = llm
        self.model_name = model_name
        self.system_prompt = _load_prompt("system_prompt.txt")
        self.citation_prompt = _load_prompt("citation_prompt.txt")

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        demographic_context: str | None = None,
    ) -> str:
        """Generate a cited answer for a health question.

        Args:
            query: The user's question.
            retrieved_chunks: Results from the Retriever (list of dicts
                with 'document' and 'metadata' keys).
            demographic_context: Optional formatted demographic string
                (age range, sex) to inject into the prompt.

        Returns:
            The LLM's response as a string with inline citations.
        """
        context_block = _build_context_block(retrieved_chunks)

        demo_section = ""
        if demographic_context:
            demo_section = f"\n\n{demographic_context}\n"

        # Llama-specific synthesis encouragement (see _LLAMA_SYNTHESIS_SUPPLEMENT)
        llama_extra = ""
        if "llama" in self.model_name.lower():
            llama_extra = _LLAMA_SYNTHESIS_SUPPLEMENT

        user_prompt = (
            f"{self.citation_prompt}\n\n"
            f"## Retrieved Context\n\n"
            f"{context_block}\n\n"
            f"## Question\n\n"
            f"{query}{demo_section}{llama_extra}\n\n"
            f"## Your Answer\n\n"
            f"Provide a clear, well-cited answer following the instructions above. "
            f"End with a Sources list and then this disclaimer on its own line:\n\n"
            f"*⚕️ This information is for educational purposes only and is not "
            f"a substitute for professional medical advice. Please consult a "
            f"healthcare provider for personal medical decisions.*"
        )

        return self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
        )


if __name__ == "__main__":
    from llm.groq_client import GroqLLM
    from pipeline.retriever import Retriever

    retriever = Retriever()
    chunks = retriever.retrieve("What causes type 2 diabetes?", top_k=5)
    gen = Generator(llm=GroqLLM())
    answer = gen.generate("What causes type 2 diabetes?", chunks)
    print(answer)
