# Text-to-SQL Architecture Comparison: Encoder–Decoder vs. Decoder-Only

This repository contains an implementation and comparative analysis of two major transformer-based architectures—**Encoder–Decoder** (using `facebook/bart-base`) and **Decoder-Only** (using `gpt2`)—for the **Text-to-SQL** task. The objective is to evaluate how each architecture translates natural language questions into valid SQL queries given a database schema, analyzing their performance, behavior, and limitations.

---

## Table of Contents
1. [Dataset & Preprocessing](#1-dataset--preprocessing)
2. [Encoder–Decoder Model: BART (`facebook/bart-base`)](#2-encoderdecoder-model-bart)
3. [Decoder-Only Model: GPT-2 (`gpt2`)](#3-decoder-only-model-gpt-2)
4. [Comparative Analysis & Discussion](#4-comparative-analysis--discussion)
5. [Conclusion & Recommendations](#5-conclusion--recommendations)
6. [Getting Started & Installation](#6-getting-started--installation)

---

## 1. Dataset & Preprocessing

### Brief Implementation Summary
We load the `gretelai/synthetic_text_to_sql` dataset from Hugging Face, containing raw text pairs of natural language prompts, database schemas, and ground-truth SQL queries. The raw dataset features 100,000 training samples and 5,851 test samples. Columns (`sql_prompt`, `sql_context`, `sql`) are renamed to `question`, `schema`, and `query` respectively. To manage computational efficiency, we subsample the training set to 20,000 samples, the development/dev set to 500 samples, and the test set to 2,000 samples using stratified random sampling.

### Detailed Analysis & Responses
* **Dataset Structure & Samples:** Each sample consists of three primary components:
  * `question`: The natural language query posed by the user (e.g., *"What is the total volume of timber sold by each salesperson, sorted by salesperson?"*).
  * `schema`: The DDL statements defining tables and columns (e.g., `CREATE TABLE salesperson (...)`).
  * `query`: The corresponding valid SQL statement.
* **Challenges & Complexity:** The dataset spans diverse domains with varying SQL complexity—ranging from simple single-table lookups to complex multi-table `JOIN` operations, nested subqueries, conditional logic (`WHERE`), grouping (`GROUP BY`), and aggregate functions (`SUM`, `AVG`, `MAX`). The model must learn not only SQL syntax but also how to map semantic concepts in the natural language question to precise schema entities (tables, columns, and foreign keys).

---

## 2. Encoder–Decoder Model: BART

### Brief Implementation Summary
We train `facebook/bart-base` as our representative Encoder–Decoder architecture. The input sequence is formatted as `Question: {question} Schema: {schema}`, processed bidirectionally by the encoder. The decoder autoregressively generates the target SQL query. Tokenizer padding IDs in the target labels are set to `-100` to be ignored in the loss function calculation. Training is executed using Hugging Face's `Trainer` API for 1 epoch with a batch size of 16, gradient accumulation steps of 2, and a learning rate of $5 \times 10^{-4}$ using the AdamW optimizer with FP16 precision.

### Training Results & Metrics
* **Training Time:** $908.00$ seconds
* **Training Loss:** $0.6300$
* **Dev Set Evaluation ($500$ samples):**
  * Raw Exact Match (EM): $19.60\%$
  * Normalized Exact Match (EM): $20.40\%$
* **Test Set Evaluation ($2000$ samples):**
  * Raw EM: $17.35\%$ ($21.80\%$ in final summary)
  * Normalized EM: $17.80\%$ ($21.35\%$ in final summary)

### Detailed Evaluation & Error Analysis
* **Metrics Explained:** 
  * *Raw EM* measures exact string matches between the model's prediction and the gold SQL without any modification. It is strict and penalizes trivial formatting or spacing discrepancies.
  * *Normalized EM* normalizes both strings (lowercasing, stripping whitespace, removing trailing semicolons, and normalizing keywords) before comparison, resulting in a slightly higher match rate.
* **Observed Behaviors & Errors:**
  * **Strengths:** BART excels at basic condition matching and straightforward conditional filters (e.g., simple filtering on a single table).
  * **Weaknesses:** It occasionally confuses database functions (e.g., `DATE_SUB` vs `NOW() - INTERVAL`), struggles with complex window functions or subqueries, and sometimes misorders `GROUP BY` and `ORDER BY` clauses.

---

## 3. Decoder-Only Model: GPT-2

### Brief Implementation Summary
We train `gpt2` as our representative Decoder-Only autoregressive architecture. The text is formatted as a causal language modeling prompt: `Question: {question} Schema: {schema} SQL: {query}<|endoftext|>`. During data tokenization and batch processing, prompt tokens are masked with `-100` in the labels so that loss is computed exclusively on the generated SQL tokens. Training is configured similarly to BART (1 epoch, batch size 16, gradient accumulation 2, learning rate $5 \times 10^{-4}$, AdamW optimizer, FP16).

### Training Results & Metrics
* **Training Time:** $580.61$ seconds
* **Training Loss:** $0.8434$
* **Dev Set Evaluation ($500$ samples):**
  * Raw EM: $16.80\%$
  * Normalized EM: $17.60\%$
* **Test Set Evaluation ($2000$ samples):**
  * Raw EM: $18.50\%$
  * Normalized EM: $18.75\%$

### Detailed Evaluation & Error Analysis
* **Observed Behaviors & Errors:**
  * **Strengths:** GPT-2 successfully captures simple SQL syntactic templates and handles straightforward queries well (e.g., basic `SELECT * FROM table WHERE type = 'Hydro'`).
  * **Weaknesses:** As an autoregressive decoder-only model without a separate encoder to process the schema bi-directionally, GPT-2 suffers from *schema hallucination* (referencing columns or tables absent from the schema) and structural syntax errors in long queries due to error propagation.

---

## 4. Comparative Analysis & Discussion

### Architectural Contrasts
1. **Bidirectional vs. Unidirectional Attention:** BART uses a full Encoder–Decoder framework where the encoder processes the input question and schema bidirectionally ($O(N^2)$ cross-attention over context), allowing deeper semantic understanding. GPT-2 uses causal unidirectional attention, processing text strictly from left to right.
2. **Training Objective:** BART is pre-trained via denoising autoencoding, making it robust at sequence-to-sequence reconstruction. GPT-2 is trained on causal language modeling, optimized for fluent text continuation rather than structural translation.
3. **Schema Interaction:** BART's encoder natively ingests and encodes the schema alongside the question in parallel. GPT-2 treats the entire prompt as a single sequential prefix, making it more prone to losing track of schema details over long contexts.

### Error Taxonomy
* **BART Errors:** Semantic misalignments in multi-table `JOIN` conditions, misuse of aggregate functions, and minor ordering issues in clauses.
* **GPT-2 Errors:** Syntax violations, table/column hallucination, and severe error propagation in sequential generation.

---

## 5. Conclusion & Recommendations

* **Why Performance is Moderate:** The models were trained for only 1 epoch on a subsampled subset ($20,000$ samples), limiting deep convergence across complex SQL semantics. Increasing training epochs, expanding the training set, and tuning learning rate schedules will substantially boost performance.
* **When to Choose BART:** BART is superior for Text-to-SQL tasks where understanding complex database schemas, foreign key relationships, and precise structural mapping is required.
* **Future Enhancements:** 
  * Implement *Constrained Decoding* to restrict the decoder's token vocabulary to valid schema columns and tables.
  * Utilize graph-based schema representations instead of flat text prompts to improve structural comprehension.
  * Explore larger pre-trained Encoder-Decoder models (e.g., T5 or CodeT5).

---

## 6. Getting Started & Installation

### Prerequisites
Make sure you have Python 3.8+ and PyTorch installed. Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

### Running the Notebook
Open the Jupyter Notebook in your environment:
```bash
jupyter notebook Text_To_SQL_BART_GPT2_Comparison.ipynb
```