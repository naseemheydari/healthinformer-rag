# Portfolio Code Notes

This directory preserves the core HealthInformer application architecture from the collaborative project.

The final academic report centers its model comparison on Claude 3.5 Haiku and Llama 3.3 70B served through AWS Bedrock. The codebase also contains an optional Groq client that was used during development/evaluation experimentation. It is retained here for transparency but should not be interpreted as the primary reported deployment path.

The RAG application itself is implemented with custom retrieval, generation, and orchestration modules. LangChain is used in the evaluation layer for RAGAS integration rather than as the primary application orchestration framework.
