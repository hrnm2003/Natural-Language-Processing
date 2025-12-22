# LLM Instruction Following and Fine-Tuning Project

This repository contains a comprehensive pipeline for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs) using Parameter-Efficient Fine-Tuning (PEFT/LoRA) and evaluating their instruction-following capabilities via the **IFEval** benchmark.

---

## Table of Contents
1. [Section 1: Dataset, Tokenizers, and Model Loading](#section-1-dataset-tokenizers-and-model-loading)
2. [Section 2: Data Preparation and LoRA Configuration](#section-2-data-preparation-and-lora-configuration)
3. [Section 3: Model Training and Weight Merging](#section-3-model-training-and-weight-merging)
4. [Section 4: Model Evaluation (Qualitative & Quantitative)](#section-4-model-evaluation-qualitative--quantitative)
5. [Installation & Requirements](#installation--requirements)

---

## Section 1: Dataset, Tokenizers, and Model Loading

### 1. Dataset Overview
* **Dataset Used:** `allenai/tulu-3-sft-personas-instruction-following`
* **Objective:** Designed for instruction-following tasks based on various user personas. 
* **Structure:** Each entry contains an ID, a user prompt, a list of conversation messages (user/assistant roles), and specific constraints (e.g., formatting, keyword inclusion/exclusion). A sample size of 2,000 examples was selected for efficient fine-tuning.

### 2. Tokenizer Analysis
We analyzed and compared three different tokenizers:
1. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. `google/gemma-3-4b-it` (and Gemma-2-2b-it variants)
3. `meta-llama/Llama-3.1-8B`

#### Descriptive Questions & Answers:
* **Q: Based on the outputs, which model performs better in processing Persian text?**
  * **Answer:** **Llama-3.1-8B** and **Gemma-3-4b-it** demonstrate significantly better performance and tokenization efficiency for Persian text compared to TinyLlama. TinyLlama's vocabulary is heavily optimized for English and code, causing Persian words to be broken down into excessive subwords or individual characters (high token inflation). In contrast, newer models with larger, more multilingual vocabularies handle Persian scripts much more compactly.
* **Q: Why do these models tokenize the same text differently?**
  * **Answer:** Tokenizers rely on different underlying algorithms (e.g., Byte-Pair Encoding (BPE) vs. Unigram), distinct vocabulary sizes (ranging from ~32k to over 256k tokens), and diverse pre-training corpora. Consequently, token boundaries, punctuation handling, whitespace preservation, and multi-byte character encodings vary across models.

### 3. Model Loading & Precision (VRAM Analysis)
* **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (~1.1 billion parameters).
* **Q: What is the difference in VRAM usage when defining `torch_dtype` as `float16` or `float32`?**
  * **Answer:** 
    * `float32` allocates 4 bytes per parameter, resulting in a VRAM footprint of approximately **4.10 GB**.
    * `float16` (or `bfloat16`) allocates 2 bytes per parameter, cutting the memory footprint in half to approximately **2.05 GB**. This optimization is crucial for fitting model training and inference onto consumer-grade GPUs.

### 4. Chat Template Structure
* **Q: Why is adhering to the chat template necessary for instruction-tuned models?**
  * **Answer:** Chat templates format raw role-based messages into specialized marker tokens (e.g., `<|system|>`, `<|user|>`, `<|assistant|>`, `<s>`, `</s>`). These tokens signal to the model who is speaking, where turns begin and end, and how to separate system instructions from user queries. Ignoring templates leads to degraded generation quality and failure to follow conversational constraints.

---

## Section 2: Data Preparation and LoRA Configuration

### 1. SFT Data Preparation
Conversations were tokenized and formatted using the model's native chat template, truncated and padded to a maximum sequence length of `1024` tokens.

### 2. LoRA (Low-Rank Adaptation) Configuration
To avoid full-parameter fine-tuning overhead, LoRA injects trainable rank decomposition matrices into target layers.
* **Configuration Parameters:**
  * `r = 16` (Rank)
  * `lora_alpha = 32` (Scaling factor)
  * `lora_dropout = 0.05`
  * `target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]`
* **Trainable Parameters:** Out of ~1.1B parameters, only **4,505,600 parameters** (~0.41%) are trainable.

#### Descriptive Questions & Answers:
* **Q: What is your logic for choosing the values of rank ($r$), $\alpha$, and `target_modules`?**
  * **Answer:** 
    * **Rank ($r=16$):** Provides a balanced trade-off between model adaptability and memory efficiency, capturing complex instruction patterns without overfitting on a 2k subset.
    * **Alpha ($\alpha=32$):** Set to $2 \times r$ as a standard scaling heuristic to stabilize learning rates when scaling rank.
    * **Target Modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`):** Targeting all attention projection matrices maximizes the model's capacity to learn new instruction-following behaviors while keeping the parameter count minimal.

---

## Section 3: Model Training and Weight Merging

* **Training Arguments:** 5 epochs, batch size of 4, gradient accumulation steps = 4, learning rate = $2 \times 10^{-4}$, FP16 enabled.
* **Artifacts & File Sizes:**
  * **LoRA Adapter Size:** `~21.15 MB` (contains only lightweight adapter weights).
  * **Merged Model Size:** `~2.05 GB` (full base model integrated with fine-tuned adapter weights for seamless deployment).

---

## Section 4: Model Evaluation (Qualitative & Quantitative)

### 1. Qualitative Evaluation
Comparing prompt outputs between the Base Model and the Fine-Tuned Model showed that the fine-tuned version strictly adhered to complex constraints (such as writing in ALL CAPS, ending with specific words like "soul", and formatting in JSON), whereas the base model frequently missed structural guidelines.

### 2. Quantitative Evaluation via IFEval Benchmark
We evaluated instruction-following rigor using the **IFEval** benchmark via the `lm-evaluation-harness` framework.

* **Q: Briefly explain what kind of instructions IFEval evaluates and its evaluation metrics.**
  * **Answer:** IFEval (Instruction Following Evaluation) tests verifiable verification constraints (e.g., "start with X", "use keyword Y", "write in N paragraphs", "json format", "uppercase"). Its metrics include **Strict Accuracy** (exact compliance with all constraints) and **Loose Accuracy** (allowing minor formatting deviations).

#### Final Benchmark Comparison Results:
Based on the execution logs stored in `./results_base` and `./results_finetuned`:

| Metric | Base Model | Fine-Tuned Model | Enhancement |
| :--- | :---: | :---: | :---: |
| **Inst Level Strict Acc** | 12.55% | 34.65% | **+22.10%** |
| **Inst Level Loose Acc** | 13.15% | 36.45% | **+23.30%** |
| **Prompt Level Loose Acc** | 9.25% | 22.55% | **+13.30%** |
| **Prompt Level Strict Acc** | 8.51% | 20.33% | **+11.82%** |

* **Analysis:** Supervised fine-tuning with LoRA dramatically improved instruction-following capability, increasing prompt-level strict accuracy from **8.51%** to **20.33%** and instruction-level strict accuracy by nearly **22%**.

---

## Installation & Requirements

To run this project, install the required dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```