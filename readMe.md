# 📊 Project 9: RAG Observability Dashboard

**A "Glass Box" Debugger for RAG Systems built with Streamlit.**

### 🚀 Concept
Building a RAG system is easy; debugging it is hard. When a bot gives a wrong answer, was it a **Retrieval Failure** (didn't find the doc) or a **Reasoning Failure** (LLM ignored the doc)?

This project builds a **Visual Dashboard** that exposes the internal state of the pipeline. It allows developers to "x-ray" the request and see:
* **Latency:** How long did retrieval vs. generation take?
* **Evidence:** Exactly which text chunks were retrieved from the DB.
* **Prompt:** The actual final string sent to GPT-4 (after template injection).
* **Cost:** Estimated token usage and price per query.

### ⚙️ Features
* **Split-Screen UI:** Chat interface on the left, Debug metrics on the right.
* **Latency Breakdown:** Visual metrics for Retrieval time vs. LLM generation time.
* **Source Inspection:** View the raw text and metadata (filename/URL) of retrieved chunks.
* **Cost Estimator:** Real-time calculation of input/output token costs.

### 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (for the Dashboard UI).
* **Backend:** LangChain (Orchestration).
* **Database:** ChromaDB (Vector Store from Project 6).
* **AI:** OpenAI GPT-4o-mini.

### 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rag-observability-dashboard
    ```

2.  **Prerequisite (Critical):**
    This project requires the Vector Database created in **Project 6**.
    * Copy the `chroma_db_multi` folder from Project 6 into this directory.
    * *Structure:*
        ```text
        rag-observability-dashboard/
        ├── app.py
        ├── chroma_db_multi/  <-- Paste this folder here
        └── ...
        ```

3.  **Install Dependencies:**
    ```bash
    pip install streamlit langchain langchain-chroma langchain-openai python-dotenv
    ```

4.  **Set up Environment:**
    Create a `.env` file and add your OpenAI Key:
    ```env
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
    ```

### 🏃‍♂️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
