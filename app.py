import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Secrets
# Loads your OPENAI_API_KEY from the .env file.
load_dotenv()


# ==============================================================================
# 2. CACHED RESOURCE LOADING (Performance Optimization)
# ==============================================================================
# @st.cache_resource is CRITICAL here.
# Without it, Streamlit would reload the 1GB+ Vector Database every time
# you clicked a button, making the app incredibly slow.
# This ensures we load the DB and LLM once into memory and keep them there.
@st.cache_resource
def load_rag_pipeline():
    # A. Setup Embedding Model
    # Must match the model used to create the DB (e.g., text-embedding-3-small).
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # B. Load Vector Store
    # We point to the existing "./chroma_db_multi" folder where we saved our data previously.
    vector_store = Chroma(
        persist_directory="./chroma_db_multi",
        embedding_function=embedding_function
    )

    # C. Initialize LLM
    # We use 'gpt-4o-mini' with temperature=0 for fast, factual answers.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    return vector_store, llm


# Initialize the pipeline immediately when the app starts.
vector_store, llm = load_rag_pipeline()

# ==============================================================================
# 3. UI LAYOUT CONFIGURATION
# ==============================================================================
# 'layout="wide"' expands the page to use the full width of the browser,
# allowing us to create a split-screen view comfortably.
st.set_page_config(page_title="RAG X-Ray Debugger", layout="wide")

st.title("🔍 RAG 'Glass Box' Debugger")
st.markdown("Ask a question and see *exactly* what happens under the hood.")

# Create two side-by-side columns:
# col1: The User Interface (Chat)
# col2: The Developer Interface (Internals/Debug Info)
col1, col2 = st.columns([1, 1])

# --- LEFT COLUMN: Chat Interface ---
with col1:
    st.subheader("💬 Chat Interface")
    question = st.text_input("Enter your question:")
    run_button = st.button("Run RAG Pipeline")

# ==============================================================================
# 4. EXECUTION LOGIC (The "X-Ray" Logic)
# ==============================================================================
if run_button and question:
    with col1:
        st.info(f"Thinking... '{question}'")

    # A. Start Global Timer
    # To measure total end-to-end latency.
    start_time = time.time()

    # --- STEP 1: RETRIEVAL (The Search) ---
    # We search the vector store for the 3 most similar chunks (k=3).
    results = vector_store.similarity_search(question, k=3)

    # Capture how long just the search took (helpful for debugging slow DBs).
    retrieval_time = time.time() - start_time

    # --- STEP 2: CONTEXT PREPARATION ---
    # We need to turn the list of Document objects into a single string
    # that we can paste into the prompt.
    context_text = ""
    sources = []
    for doc in results:
        context_text += doc.page_content + "\n\n"
        # Collect metadata (e.g., filenames) for citation.
        sources.append(doc.metadata.get("source", "Unknown"))

    # --- STEP 3: PROMPT ENGINEERING ---
    # This is the "Hidden Prompt" the user usually never sees.
    # It wraps the user's question with strict instructions and the retrieved data.
    template = """
    Answer the question based ONLY on the context below.
    If the context is empty or irrelevant, say "I don't know".

    <Context>
    {context}
    </Context>

    Question: {question}
    """
    prompt_template = ChatPromptTemplate.from_template(template)

    # We fill in the placeholders {context} and {question}.
    # 'final_prompt' is the actual object sent to OpenAI.
    final_prompt = prompt_template.invoke({"context": context_text, "question": question})

    # --- STEP 4: GENERATION (The Answer) ---
    ai_response = llm.invoke(final_prompt)

    end_time = time.time()
    total_latency = end_time - start_time

    # ==========================================================================
    # 5. RESULT DISPLAY (What the User Sees)
    # ==========================================================================
    with col1:
        st.success("✅ Done!")
        st.write(f"**Answer:** {ai_response.content}")
        # Use Python sets to remove duplicate source names.
        st.caption(f"Sources: {', '.join(set(sources))}")

    # ==========================================================================
    # 6. DEBUG DISPLAY (What the Developer Sees)
    # ==========================================================================
    with col2:
        st.subheader("⚙️ Under the Hood")

        # A. Key Metrics (The Dashboard)
        # Using st.metric creates nice big number cards for KPIs.
        m1, m2, m3 = st.columns(3)
        m1.metric("Latency", f"{total_latency:.2f}s")  # Total wait time
        m2.metric("Retrieval Time", f"{retrieval_time:.2f}s")  # Search speed
        m3.metric("Chunks Retrieved", len(results))  # Data volume

        # B. Visualize Retrieved Evidence
        # We use an 'expander' to hide the messy details unless clicked.
        # This shows EXACTLY what text chunks the AI read.
        with st.expander("📂 Step 1: Retrieved Chunks (Evidence)", expanded=True):  # expander() is used here to create an section that can be expanded or collapsed
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk {i + 1}** (Source: `{doc.metadata.get('source', 'Unknown')}`)") #markdown is method of st used to display markdown text
                st.text(doc.page_content[:300] + "...")  # Preview first 300 chars, ... indicates more text, text is method of st used to display text
                st.divider() #divider() is used to display a horizontal line as separator


        # C. Visualize the Prompt
        # This is arguably the most valuable debug tool.
        # It reveals if the prompt context is empty, malformed, or overwhelmed with junk data.
        with st.expander("🧠 Step 2: The Final Prompt (What GPT saw)", expanded=False):
            st.code(final_prompt.messages[0].content)

        # D. Cost Estimation
        # A simple math formula to estimate API costs based on token length.
        # (1 token ≈ 4 characters).
        with st.expander("💰 Step 3: Token Usage & Cost", expanded=False):
            input_tokens = len(final_prompt.messages[0].content) / 4
            output_tokens = len(ai_response.content) / 4

            # Pricing logic for GPT-4o-mini (approximate rates):
            # Input: $0.15 / 1M tokens | Output: $0.60 / 1M tokens
            cost = ((input_tokens / 1_000_000) * 0.15) + ((output_tokens / 1_000_000) * 0.60)

            st.write(f"**Input Tokens:** ~{int(input_tokens)}")
            st.write(f"**Output Tokens:** ~{int(output_tokens)}")
            st.write(f"**Estimated Cost:** ${cost:.6f}")