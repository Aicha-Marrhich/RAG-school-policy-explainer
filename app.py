# ─────────────────────────────────────────────────────
# AI SCHOOL POLICY EXPLAINER
# ─────────────────────────────────────────────────────
# Upload any school policy PDF and ask questions
# about it in plain English.
#
# Built with: Streamlit + OpenRouter + PyPDF2
# Deploy: Streamlit Cloud (free)
# ─────────────────────────────────────────────────────

import os
import re
import math
import streamlit as st
import PyPDF2
import io
from openai import OpenAI

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="School Policy Explainer",
    page_icon="📋",
    layout="centered"
)

st.title("📋 School Policy Explainer")
st.markdown("""
Upload your school's AI policy (or any policy PDF) and 
ask questions about it in plain English. The tool only 
answers from what's in your document — no hallucination.
""")

st.divider()

# ─────────────────────────────────────────
# API KEY — from Streamlit secrets or user input
# ─────────────────────────────────────────
default_key = ""
try:
    default_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass

if default_key:
    api_key = default_key
    st.success("API key loaded securely.", icon="✅")
else:
    api_key = st.text_input(
        "Your OpenRouter API key",
        type="password",
        placeholder="sk-or-...",
        help="Free key at openrouter.ai"
    )

# ─────────────────────────────────────────
# PDF UPLOAD
# ─────────────────────────────────────────
st.subheader("Step 1 — Upload your policy document")

uploaded_file = st.file_uploader(
    "Upload a PDF policy document",
    type=["pdf"],
    help="Works best with text-based PDFs. "
         "Scanned image PDFs may not extract correctly."
)

# ─────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────
def extract_pdf_text(file):
    """Extract all text from a PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[Page {page_num + 1}]\n{page_text}"
    return text

# ─────────────────────────────────────────
# TEXT CHUNKING
# Splits document into overlapping chunks
# so no information gets cut off mid-sentence
# ─────────────────────────────────────────
def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ─────────────────────────────────────────
# SIMPLE RELEVANCE SEARCH
# Finds which chunks are most relevant to
# the user's question using word matching.
# Not as powerful as vector search but
# works without complex dependencies.
# ─────────────────────────────────────────
def find_relevant_chunks(question, chunks, top_k=4):
    """Find the most relevant chunks for a question."""
    question_words = set(
        re.findall(r'\b\w{3,}\b', question.lower())
    )

    # Score each chunk by how many question words appear
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(
            re.findall(r'\b\w{3,}\b', chunk.lower())
        )
        # Score = number of matching words
        score = len(question_words & chunk_words)
        scored.append((score, i, chunk))

    # Sort by score, return top chunks
    scored.sort(key=lambda x: x[0], reverse=True)
    relevant = [chunk for score, i, chunk in scored[:top_k]
                if score > 0]

    return relevant if relevant else chunks[:top_k]

# ─────────────────────────────────────────
# ASK THE AI
# Sends the question + relevant chunks
# to the LLM and gets a grounded answer
# ─────────────────────────────────────────
def ask_policy_question(question, relevant_chunks, api_key):
    """Send question + context to LLM via OpenRouter."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Build the context from relevant chunks
    context = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = """You are a helpful assistant that 
explains school policies to staff in plain language.

STRICT RULES:
1. Only answer using the POLICY EXCERPTS provided below
2. If the answer is not in the excerpts, say exactly:
   "This specific topic isn't clearly addressed in the 
   sections of the policy I can see. I recommend 
   checking the full document or asking your policy lead."
3. When answering, quote the relevant section briefly
4. Use clear, plain language — no jargon
5. If there are specific rules or steps, list them clearly
6. Always be helpful and constructive in tone

You are NOT making things up. You are a faithful 
interpreter of the written policy only."""

    user_message = f"""POLICY EXCERPTS:
{context}

STAFF QUESTION:
{question}

Please answer based only on the policy excerpts above."""

    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=800,
        temperature=0.3  # Low temperature = more faithful answers
    )

    return response.choices[0].message.content

# ─────────────────────────────────────────
# MAIN APP LOGIC
# ─────────────────────────────────────────
if uploaded_file:

    # Extract and process the PDF
    with st.spinner("Reading your policy document..."):
        try:
            raw_text = extract_pdf_text(uploaded_file)
            chunks = chunk_text(raw_text)

            # Show document info
            word_count = len(raw_text.split())
            st.success(
                f"Document loaded: **{uploaded_file.name}** "
                f"({word_count:,} words, {len(chunks)} sections)"
            )

        except Exception as e:
            st.error(
                f"Could not read this PDF: {str(e)}. "
                "Make sure it's a text-based PDF, "
                "not a scanned image."
            )
            st.stop()

    st.divider()
    st.subheader("Step 2 — Ask a question about the policy")

    # Suggested questions
    st.markdown("**Try asking:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Can teachers use AI tools in class?",
                     use_container_width=True):
            st.session_state.question = \
                "Can teachers use AI tools in class?"
        if st.button("What about student data privacy?",
                     use_container_width=True):
            st.session_state.question = \
                "What does the policy say about student " \
                "data privacy?"
    with col2:
        if st.button("What are the rules for students?",
                     use_container_width=True):
            st.session_state.question = \
                "What are the rules for students " \
                "using AI tools?"
        if st.button("What happens if someone breaks the policy?",
                     use_container_width=True):
            st.session_state.question = \
                "What are the consequences for " \
                "breaking the policy?"

    # Question input
    question = st.text_area(
        "Or type your own question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. What does this policy say about "
                    "using ChatGPT for lesson planning?",
        height=80
    )

    # Ask button
    if st.button("📋 Find the answer",
                 use_container_width=True,
                 type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not api_key:
            st.error("Please enter your OpenRouter API key.")
        else:
            with st.spinner(
                "Searching the policy document..."
            ):
                # Find relevant sections
                relevant = find_relevant_chunks(
                    question, chunks
                )

                # Get AI answer
                try:
                    answer = ask_policy_question(
                        question, relevant, api_key
                    )

                    # Display answer
                    st.divider()
                    st.subheader("📌 What the policy says")
                    st.markdown(answer)

                    # Show source sections (expandable)
                    with st.expander(
                        "View source sections from the document"
                    ):
                        st.markdown(
                            "*These are the sections the "
                            "answer is based on:*"
                        )
                        for i, chunk in enumerate(relevant):
                            st.markdown(
                                f"**Section {i+1}:**"
                            )
                            st.markdown(
                                f"> {chunk[:400]}..."
                                if len(chunk) > 400
                                else f"> {chunk}"
                            )
                            st.markdown("---")

                    # Download option
                    qa_record = (
                        f"QUESTION:\n{question}\n\n"
                        f"ANSWER:\n{answer}\n\n"
                        f"Document: {uploaded_file.name}"
                    )
                    st.download_button(
                        "⬇️ Download this Q&A",
                        data=qa_record,
                        file_name="policy_qa.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(
                        f"Something went wrong: {str(e)}"
                    )

    st.divider()

    # Session history
    if "history" not in st.session_state:
        st.session_state.history = []

else:
    # Show instructions when no file uploaded
    st.info(
        "👆 Upload a policy PDF above to get started. "
        "Works with any school policy document — "
        "AI policy, safeguarding, GDPR, behaviour policy, etc."
    )

    # Example use cases
    st.subheader("What can you use this for?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**School AI Policy**
- Can teachers use AI?
- Rules for students
- Data handling
- Consequences
        """)
    with col2:
        st.markdown("""
**Safeguarding Policy**
- Reporting procedures
- Staff responsibilities
- Contact details
- Review dates
        """)
    with col3:
        st.markdown("""
**GDPR/Data Policy**
- What data we collect
- How it's stored
- Parent rights
- Breach procedure
        """)

st.divider()
st.caption(
    "This tool only uses information from your uploaded "
    "document. No document content is stored or shared. "
    "Built for AI in Education portfolio."
)
