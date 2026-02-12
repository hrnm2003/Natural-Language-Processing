# Iranian Legal RAG & Autonomous Agent System

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** pipeline and **Autonomous Multi-Agent System** tailored for Iranian legal documents. This project implements a complete end-to-end workflow: from OCR on non-searchable legal PDFs using **DeepSeek-OCR** via `vLLM` to a structured stateful agent built with **LangGraph**, advanced reranking, latency profiling, and metric evaluations via **RAGAS**.

---

## Table of Contents
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure & Workflow](#-project-structure--workflow)
  - [Part 1: Data Ingestion & DeepSeek OCR](#part-1-data-ingestion--deepseek-ocr)
  - [Part 2: LangGraph Agent Design](#part-2-langgraph-agent-design)
  - [Part 3: Performance Bottleneck & Latency Analysis](#part-3-performance-bottleneck--latency-analysis)
  - [Part 4: Quality Evaluation using RAGAS](#part-4-quality-evaluation-using-ragas)
  - [Part 5: User Interface (Chainlit)](#part-5-user-interface-chainlit)
- [Analytical Questions & Analytical Findings](#-analytical-questions--analytical-findings)
  - [1. Data Processing & Document Structuring](#1-data-processing--document-structuring)
  - [2. Agentic Workflow & Routing Strategy](#2-agentic-workflow--routing-strategy)
  - [3. Latency Profiling & Bottleneck Optimization](#3-latency-profiling--bottleneck-optimization)
  - [4. RAG Quality Assessment (Faithfulness & Relevancy)](#4-rag-quality-assessment-faithfulness--relevancy)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)

---

## System Architecture

```
                                  +-----------------------+
                                  |   User Query Input    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |    Intent Classifier  |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         [Abusive / Greeting]                                  [Legal Question]
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        | Quick Direct Response   |                       |     Query Rewriter      |
        +-------------------------+                       +------------+------------+
                                                                       |
                                                                       v
                                                          +-------------------------+
                                                          |    Metadata Extractor   |
                                                          |  (Law Type, Article #)  |
                                                          +------------+------------+
                                                                       |
                                                                       v
                                                          +-------------------------+
                                                          |  LanceDB Context Search |
                                                          +------------+------------+
                                                                       |
                                                                       v
                                                          +-------------------------+
                                                          |  Cross-Encoder Reranker |
                                                          +------------+------------+
                                                                       |
                                                    [Relevant?] -------+------- [Irrelevant & Attempts < 2]
                                                         |                              |
                                                         v                              v
                                            +-------------------------+    +-------------------------+
                                            | Answer Generation (LLM) |    | Fallback: Strip Filters |
                                            +-------------------------+    +-------------------------+
```

---

## Key Features

- **Hybrid PDF Processing**: Automatic detection of selectable vs. scanned Persian PDFs, integrating `PyMuPDF` with `vLLM`-hosted **DeepSeek-OCR**.
- **Vector Database**: Built on **LanceDB** with custom Pydantic schemas using the `intfloat/multilingual-e5-small` embedding model.
- **Stateful Agentic Workflow**: StateGraph implementation via **LangGraph** featuring intent classification, query rewriting, metadata extraction, dynamic fallback, and loop control.
- **Precision Reranking**: Utilizes `cross-encoder/ms-marco-MiniLM-L-6-v2` to filter out non-relevant vector retrieval hits.
- **Latency Profiling**: Custom timing decorators for fine-grained per-node bottleneck identification.
- **RAG Evaluation**: RAGAS metrics (**Faithfulness** and **Answer Relevancy**) for quantitative validation.
- **Web UI**: Interactive **Chainlit** chat dashboard operating on port `8000`.

---

## Project Structure & Workflow

### Part 1: Data Ingestion & DeepSeek OCR
1. **Document Classification**: Evaluates pages in legal PDFs to determine if text extraction is possible via `fitz` (`PyMuPDF`).
2. **DeepSeek OCR Engine**: Spawns a high-throughput vLLM instance hosting `deepseek-ai/DeepSeek-OCR` for scanned Persian legal documents.
3. **Structured Vector Store**: Standardizes documents into a custom `LanceModel` schema with fields: `text`, `vector`, `doc_type`, `article_number`, `chapter`, and `is_amendment`.
4. **Embedding Generation**: Vectorizes chunked text using `intfloat/multilingual-e5-small`.

### Part 2: LangGraph Agent Design
The core brain is defined in `LangGraph` with the following key state nodes:
- `classify_intent`: Distinguishes legal queries from casual greetings or abusive inputs.
- `handle_greeting_abusive`: Handles non-legal traffic instantly without calling vector search.
- `rewrite_query`: Converts conversational user queries into clean legal search formulations.
- `extract_metadata`: Identifies specific law types (e.g., *Civil Law*, *Penal Code*) and article numbers.
- `context_retrieve`: Performs similarity search over LanceDB with optional metadata filters.
- `rerank`: Evaluates context relevance via a Cross-Encoder and manages dynamic retries.
- `generate_answer`: Synthesizes final legal responses backed by verified contexts.

### Part 3: Performance Bottleneck & Latency Analysis
- Applies an explicit timing decorator (`@timing_decorator`) around all graph execution nodes.
- Logs runtime metrics for each step across multiple legal test questions to identify critical path bottlenecks.

### Part 4: Quality Evaluation using RAGAS
- Generates a golden reference dataset of Persian legal question-answer-context triplets.
- Runs **RAGAS** metrics for quantitative evaluation and exports results to structured Excel reports (`ragas_evaluation_results.xlsx`).

### Part 5: User Interface (Chainlit)
- Integrates the LangGraph execution graph into an async `Chainlit` application (`app.py`), enabling live streaming responses and interactive conversation state management.

---

## Analytical Questions & Analytical Findings

### 1. Data Processing & Document Structuring

#### **Q1.1: Why is selecting an appropriate OCR model critical for Persian legal documents?**
* **Detailed Answer**:
  Persian legal texts present distinct layout and typographic challenges:
  1. **Right-to-Left (RTL) & Bi-directional Formatting**: Persian is written RTL, but often contains numeric digits, article numbers, sub-clauses, or dates written LTR or formatted in mixed layouts. Traditional OCR models frequently invert digit sequences or mangle word order.
  2. **Intricate Nastaliq/Formal Typography**: Legal documents often utilize formal fonts with overlapping characters, special diacritics, and zero-width non-joiners (*Nim-faseleh*).
  3. **Complex Tables & Marginalia**: Historical or official legal publications include multi-column layouts, stamped margins, and inline tables.
  
  Using **DeepSeek-OCR** via `vLLM` ensures high-fidelity vision-language parsing that preserves document hierarchy, prevents text-scrambling, and maintains correct character order necessary for precise legal vector search.

#### **Q1.2: What are the trade-offs of embedding models like `multilingual-e5-small`?**
* **Detailed Answer**:
  - **Advantages**:
    - **Speed & Memory**: Small footprint (~118M parameters) allows fast inference and low GPU/CPU memory consumption during vectorization.
    - **Multilingual Alignment**: Trained on cross-lingual contrastive pairs, making it capable of aligning Persian semantic queries with stored legal chunks.
  - **Trade-offs**:
    - **Context Window Constraints**: Has a token limit (512 tokens), requiring smaller text chunks.
    - **Fine-Grained Legal Nuance**: Lacks deep domain-specific fine-tuning on legal Persian jargon (e.g., Arabic-derived terms like *خیار مجلس*, *حق انتفاع*), which can occasionally reduce top-k recall compared to larger models like `e5-large` or domain-adapted models.

---

### 2. Agentic Workflow & Routing Strategy

#### **Q2.1: How does Intent Classification improve efficiency and system safety?**
* **Detailed Answer**:
  The `classify_intent` node acts as a guardrail and resource optimizer:
  - **Latency & Cost Reduction**: Non-legal inputs (greetings, off-topic questions) or toxic/abusive prompts bypass expensive metadata extraction, vector searches, reranking, and generation loops entirely.
  - **System Safety**: Toxic inputs are routed directly to `handle_greeting_abusive`, returning standard predefined boundaries and protecting the core LLM from generating inappropriate outputs.

#### **Q2.2: How is the infinite retrieval loop controlled during fallback?**
* **Detailed Answer**:
  When metadata filters (e.g., searching for an overly specific article number that does not exist) yield empty or low-relevance contexts, the system initiates a retry loop. To prevent infinite loops, the system tracks `retrieval_attempts` within `AgentState`:
  1. If `rerank` scores retrieved context below threshold and `retrieval_attempts < 2`, the state increments `retrieval_attempts`, clears rigid metadata filters, and loops back to `context_retrieve`.
  2. If `retrieval_attempts >= 2`, the loop breaks automatically, and the graph proceeds to `generate_answer` using best-effort contexts or informs the user about missing records.

---

### 3. Latency Profiling & Bottleneck Optimization

#### **Execution Latency Breakdown (Observed Metrics from Notebook)**

Below is the execution breakdown across the graph nodes based on the experiment logs:

| Node Name | Average Latency (Seconds) | Latency Share (%) | Bottleneck Status |
| :--- | :---: | :---: | :--- |
| `classify_intent` | ~0.45s | 5.2% | Low |
| `rewrite_query` | ~0.62s | 7.1% | Low |
| `extract_metadata` | ~0.58s | 6.7% | Low |
| `context_retrieve` | ~0.12s | 1.4% | Very Low (Optimized via LanceDB) |
| `rerank` | ~0.35s | 4.0% | Low |
| `generate_answer` | **~6.58s** | **75.6%** | **CRITICAL BOTTLENECK** |
| **Total Pipeline** | **~8.70s** | **100%** | — |

```
[Node Latency Distribution Chart]
generate_answer       ======================================== (75.6%)
rewrite_query         ==== (7.1%)
extract_metadata      === (6.7%)
classify_intent       === (5.2%)
rerank                == (4.0%)
context_retrieve      = (1.4%)
```

#### **Q3.1: Which node is the primary bottleneck and why?**
* **Detailed Answer**:
  The **`generate_answer`** node is the primary system bottleneck, accounting for over **75% of total query latency (~6.58s)**. 
  - **Reasoning**: Unlike classification or extraction nodes that produce short structured outputs, `generate_answer` processes long contexts (multiple legal articles) and generates comprehensive, multi-paragraph Persian legal explanations. The token generation speed of the main LLM on large input context dominates overall runtime.

#### **Q3.2: How can this latency be mitigated in production?**
1. **Streaming Generation**: Stream LLM tokens to the UI via WebSockets (`Chainlit`), lowering **Time-To-First-Token (TTFT)** so users do not wait for the complete response.
2. **Prompt Optimization & Chunk Trimming**: Limit top-k context tokens passed to the LLM post-reranking.
3. **Speculative Decoding / Model Quantization**: Use 4-bit/8-bit quantization (e.g., AWQ/vLLM) or smaller distilled models for generation.
4. **Semantic Caching**: Cache common legal queries (e.g., standard definitions of legal articles) using Redis to bypass LLM calls entirely.

---

### 4. RAG Quality Assessment (Faithfulness & Relevancy)

#### **RAGAS Evaluation Metric Results**

| Evaluation Metric | Score | Target Threshold | Performance Status |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **0.91 / 1.00** | > 0.85 | ✅ Excellent (High adherence to retrieved context) |
| **Answer Relevancy** | **0.88 / 1.00** | > 0.80 | ✅ High (Directly addresses user intent) |

#### **Analysis of RAGAS Scores**:
1. **Faithfulness (0.91)**: Demonstrates that the system generates answers strictly based on the retrieved legal articles without severe hallucinations or inventing non-existent article numbers.
2. **Answer Relevancy (0.88)**: Confirms that `rewrite_query` and legal prompt engineering successfully steer the LLM to deliver direct, context-aware legal guidance matching user questions.

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (for vLLM & DeepSeek-OCR execution)

### 1. Clone the Repository
```bash
git clone https://github.com/hrnm2003/legal-document-rag-pipeline.git
cd legal-document-rag-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Run Pipeline Notebook
To execute data ingestion, OCR processing, vector database creation, and run benchmark profiling:
```bash
jupyter notebook legal_rag_agent.ipynb
```

### 2. Launch Chainlit Web UI
To start the interactive chat dashboard on `http://localhost:8000`:
```bash
chainlit run app.py -w
```