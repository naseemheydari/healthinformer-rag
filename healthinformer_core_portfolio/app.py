"""Streamlit chat interface for HealthInformer.

Usage:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="HealthInformer",
    page_icon="🏥",
    layout="centered",
)

# ── Disclaimer banner ──────────────────────────────────────────────────────
st.warning(
    "⚕️ **HealthInformer is an educational tool.** "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider for personal medical decisions."
)

st.title("HealthInformer")
st.caption("Evidence-based answers to your health questions, grounded in PubMed research.")

# ── Sidebar controls ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_choice = st.selectbox(
        "LLM Model",
        options=["bedrock", "bedrock-llama", "groq"],
        format_func=lambda x: {
            "bedrock": "AWS Bedrock — Claude 3.5 Haiku",
            "bedrock-llama": "AWS Bedrock — Llama 3.3 70B",
            "groq": "Groq — Llama 3.3 70B",
        }[x],
    )

    top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=5)

    st.divider()
    st.header("Personalization (optional)")
    st.caption("Tailors language emphasis — does not change medical facts.")

    age_range = st.selectbox(
        "Age range",
        options=[
            "Not specified",
            "18–29",
            "30–44",
            "45–59",
            "60–74",
            "75+",
        ],
    )

    sex = st.selectbox(
        "Sex",
        options=["Not specified", "Female", "Male"],
    )

# ── Session state init ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _build_demographic_context(age: str, sex_val: str) -> str | None:
    """Build a demographic context string, or None if nothing specified."""
    if age == "Not specified" and sex_val == "Not specified":
        return None
    from config.settings import PROMPTS_DIR
    template = (PROMPTS_DIR / "demographic_context.txt").read_text(encoding="utf-8")
    return template.format(age_range=age, sex=sex_val)


@st.cache_resource(show_spinner="Loading models and vectorstore...")
def get_chain(model: str, k: int):
    """Cache the RAGChain so we don't reload embedder on every interaction."""
    from pipeline.rag_chain import RAGChain
    return RAGChain(model=model, top_k=k)


def _render_sources(sources: list[dict]) -> None:
    """Render source cards in an expandable section."""
    # Deduplicate by PMID (same article may appear as multiple chunks)
    seen = set()
    unique_sources = []
    for s in sources:
        if s["pmid"] not in seen:
            seen.add(s["pmid"])
            unique_sources.append(s)

    with st.expander(f"📚 Sources ({len(unique_sources)} articles)", expanded=False):
        for i, s in enumerate(unique_sources, 1):
            st.markdown(
                f"**[{i}]** {s['authors']}. "
                f"*\"{s['title']}\"* — {s['journal']}, {s['year']}.  \n"
                f"🔗 [PubMed]({s['url']})"
            )
            if i < len(unique_sources):
                st.divider()


# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"])

# ── Chat input ─────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a health question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching medical literature and generating answer..."):
            try:
                chain = get_chain(model_choice, top_k)
                demo_ctx = _build_demographic_context(age_range, sex)
                result = chain.ask(question, demographic_context=demo_ctx)

                if not result["sources"]:
                    st.info(
                        "No relevant sources were found in our database for this question. "
                        "Try rephrasing or asking about a different health topic."
                    )
                    answer_text = "I couldn't find relevant medical literature to answer this question."
                    sources = []
                else:
                    answer_text = result["answer"]
                    sources = result["sources"]

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    answer_text = (
                        "⚠️ **API key not configured.** "
                        f"Please add your {'GROQ_API_KEY' if model_choice == 'groq' else 'AWS credentials'} "
                        "to the `.env` file and restart the app."
                    )
                elif "bedrock" in error_msg.lower() or "credential" in error_msg.lower():
                    answer_text = (
                        "⚠️ **AWS Bedrock not configured.** "
                        "Please add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and "
                        "AWS_SESSION_TOKEN to your `.env` file and restart."
                    )
                else:
                    answer_text = f"⚠️ **An error occurred:** {error_msg}"
                sources = []

        st.markdown(answer_text)
        if sources:
            _render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources,
    })
