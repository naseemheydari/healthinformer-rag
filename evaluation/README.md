# Evaluation

HealthInformer was evaluated from several complementary angles:

- **RAGAS:** faithfulness, answer relevancy, context precision, and context recall on the curated evaluation set.
- **PubMedQA:** biomedical reasoning benchmark using each question's provided gold context.
- **Manual spot checks:** qualitative review of generated answers for quality, citation support, and hallucinations.

The final project configuration retrieves the **top 8 passages** for RAG generation.

Generated-answer artifacts and large evaluation outputs are intentionally not duplicated in this portfolio directory. The final reported results are summarized in the project README, notebooks, figures, and full project report.
