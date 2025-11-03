# NLP Utilities: Tokenization, Edit Distance, and Regex Text Processing

This repository contains Python implementations for foundational Natural Language Processing (NLP) tasks. The project covers text parsing with regular expressions, minimum edit distance calculation for spell correction, and the training of advanced subword tokenizers (BPE and WordPiece) on a Persian dataset. It covers rule-based text processing, string distance algorithms via dynamic programming, and training state-of-the-art tokenizers from scratch using Hugging Face datasets.


## Project Overview
This project demonstrates core text processing methodologies:
1. **Email Validation:** Utilizing regex to filter and extract valid emails matching strict domain and local-part constraints.
2. **Levenshtein Distance:** Implementing a dynamic programming approach to compute the minimum edit distance between strings, applied here for context-aware spell correction.
3. **Regex Tokenization:** Building and improving custom tokenizers to handle complex punctuation, abbreviations, and numerical values.
4. **Subword Tokenization (BPE & WordPiece):** Training modern subword tokenizers, commonly utilized in transformer architectures, on the `TinyStories-Farsi` dataset.


## Features
- **Strict Email Validation:** Extracts and validates emails based on strict structural rules (unique `@`, valid TLDs, non-consecutive dots) and cross-references user names with local-part components.
- **Dynamic Programming String Matching:** Custom implementation of the Levenshtein Distance algorithm to compute edit costs (insertion, deletion, substitution) between words.
- **Auto-Correction System:** Leverages the custom Levenshtein algorithm and a provided `vocab.txt` to automatically correct misspelled sentences while preserving punctuation marks.
- **Rule-Based Tokenization:** Implements regex tokenizers capable of correctly separating advanced patterns like currencies (`$12.40`), acronyms (`U.S.A.`), and hyphenated words.
- **Modern Subword Tokenization (BPE & WordPiece):** Uses the Hugging Face `tokenizers` and `datasets` libraries to train subword segmenters on the `taesiri/TinyStories-Farsi` dataset, mapping the subword chunks comparatively.


## Theoretical Analysis

### Regex Tokenizer Analysis

**Issues with the Basic Tokenizer:**
The basic tokenizer implemented (`\w+|[^.,;:?!()\s]`) splits text purely based on non-word characters and basic punctuation. Its main flaws are:
1. **Abbreviations:** It incorrectly breaks abbreviations like "U.S.A." into isolated letter tokens ("U", "S", "A").
2. **Hyphenated Words:** Compound words like "poster-print" are forcefully separated into "poster", "-", and "print", destroying their semantic unity.
3. **Currency and Decimals:** It shatters monetary values (e.g., "$12.40") into symbols and disjointed numbers ("$", "12", "40").

**Proposed Solutions for Improvement:**
1. **Capture Abbreviations:** Introduce explicit regex capture groups for common abbreviations (e.g., matching sequences of letters alternating with periods).
2. **Preserve Financial/Numeric Structures:** Include numeric patterns with optional decimal points and currency symbols to keep values intact (`\$?\d+(\.\d+)?`).
3. **Allow Internal Punctuation:** Permit hyphens and apostrophes strictly inside word boundaries to preserve compound words and contractions.
4. **Regex Verbose Mode:** Combine these rules using a unified, verbose regular expression with prioritized alternation. This allows the regex engine to match complex patterns before falling back to isolated characters. *(This solution is implemented in the accompanying Notebook).*

### WordPiece vs. BPE Tokenization

**What is WordPiece Tokenizer?**
WordPiece is a subword tokenization algorithm prominently used in language representation models like BERT. It effectively mitigates the out-of-vocabulary (OOV) problem by breaking unknown words into recognizable subword units. Internal subwords are usually marked with a special prefix (typically `##` in English, or similar variations depending on the configuration).

**How WordPiece is Trained:**
1. **Initialization:** The process begins with a base vocabulary comprising all distinct characters present in the training corpus.
2. **Language Model Likelihood (Scoring):** Unlike purely frequency-driven models, WordPiece evaluates adjacent pairs of existing tokens and calculates a score based on language model likelihood: 
   $Score = \frac{Frequency(A, B)}{Frequency(A) \times Frequency(B)}$
3. **Merging:** It merges the pair of tokens that yields the highest score. This means it favors combining subwords where the probability of them appearing together is significantly higher than them appearing independently.
4. **Iteration:** This merging process repeats iteratively until the pre-configured maximum vocabulary size is reached.

**Comparison: WordPiece vs. BPE (Byte-Pair Encoding):**
- **Merging Criterion:** BPE uses a greedy frequency approach; it simply merges the most frequently co-occurring adjacent pairs of tokens in the corpus. In contrast, WordPiece merges pairs that maximize the language model's likelihood (utilizing a form of pointwise mutual information). 
- **Efficiency in Subword Selection:** Because WordPiece normalizes by the individual frequencies of the subwords, it is less likely to blindly merge highly frequent individual characters if they don't form a highly cohesive pair compared to other options.
- **Prefix Conventions:** WordPiece heavily relies on the `##` prefix to indicate that a token is a continuation of the preceding token (e.g., `['run', '##ning']`). While BPE can implement similar prefix/suffix rules, classical BPE often uses byte-level representations to handle unknown characters without needing explicit subword markers.


## Project Structure
The repository heavily relies on the execution of the main Jupyter Notebook along with external text files for context:
- `Regex_Tokenization.ipynb`: Core notebook containing all logic, algorithms, and training pipelines.
- `emails.txt`: Required text file containing lines structured as `name=..., email=...` for regex validation tests.
- `vocab.txt`: Required vocabulary dictionary containing valid words (separated by `, `) used by the auto-correction system.
- `.env`: Environment file containing the `HUGGINGFACE_TOKEN` required to fetch datasets.


## Requirements
Install the required packages using:

```bash
pip install -r requirements.txt
```