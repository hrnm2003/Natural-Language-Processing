# Persian N-Gram Language Modeling & Smoothing on TinyStories

This repository contains a complete implementation and empirical analysis of **N-Gram Language Models** trained on the Persian split of the **TinyStories-Farsi** dataset (`taesiri/TinyStories-Farsi`).

The project explores text preprocessing, subword tokenization via Byte-Pair Encoding (BPE), N-gram model construction, perplexity evaluation, advanced smoothing techniques (Laplace, Interpolation, Backoff), candidate sentence ranking, and temperature-controlled text generation.

---

## Table of Contents
1. [Part 1: Data Cleaning & Tokenization](#part-1-data-cleaning--tokenization)
2. [Part 2: N-Gram Implementation & Text Generation Analysis](#part-2-n-gram-implementation--text-generation-analysis)
3. [Part 3: Perplexity Evaluation & Candidate Sentence Ranking](#part-3-perplexity-evaluation--candidate-sentence-ranking)
4. [Part 4: Smoothing Techniques Analysis](#part-4-smoothing-techniques-analysis)
5. [Part 5: Temperature Sampling Analysis](#part-5-temperature-sampling-analysis)

---

## Part 1: Data Cleaning & Tokenization

### 1. Preprocessing Necessity Analysis
Raw Persian text datasets, especially translated or web-scraped corpora like TinyStories-Farsi, contain character inconsistencies that introduce artificial noise and inflate vocabulary size. The following preprocessing steps were applied to normalize the training and validation data:

* **Arabic-to-Persian Letter Standardization (`ي` $\rightarrow$ `ی` and `ك` $\rightarrow$ `ک`):**
  * **Reasoning:** In Persian, Arabic characters `ي` (Unicode U+064A) and `ك` (Unicode U+0643) cause duplicate representations for identical words (e.g., treating `کودک` with Arabic `ک` and Persian `ک` as two distinct tokens). Standardizing them to `ی` (U+06CC) and `ک` (U+06A9) ensures consistency across the dataset.
* **Digit Normalization (`0-9` $\rightarrow$ `۰-۹`):**
  * **Reasoning:** Converts all ASCII digits to standard Persian digits to unify numerical representations.
* **Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله) Handling (`\u200c` $\rightarrow$ space):**
  * **Reasoning:** ZWNJ character handling can fragment subword tokens across word boundaries in count-based N-gram models. Replacing ZWNJs with regular spaces standardizes word boundaries for BPE subword segmentation.
* **Whitespace Collapsing and Stripping:**
  * **Reasoning:** Replaces multiple consecutive whitespace characters with a single space and strips trailing/leading whitespaces to maintain uniform N-gram context windows.

#### Empirical Cleaning Statistics (Training Split):
* **Initial Corpus Length:** 95,545,837 characters
* **Final Corpus Length:** 94,859,981 characters
* **Total Characters Removed:** 685,856 characters (0.72%)
* **Detailed Transformations:**
  * Arabic `ي` $\rightarrow$ Persian `ی`: 51,097 replacements
  * Arabic `ك` $\rightarrow$ Persian `ک`: 15,147 replacements
  * English Digits $\rightarrow$ Persian Digits: 424 replacements
  * Half-space (ZWNJ) $\rightarrow$ Space: 1,116,558 replacements

### 2. BPE Tokenizer Setup
A **Byte-Pair Encoding (BPE)** tokenizer was trained on the cleaned Persian training set with the following specifications:
* **Vocabulary Size ($V$):** 5,000 subword tokens
* **Pre-tokenizer:** Whitespace
* **Special Tokens:** `[UNK]` (Index 0), `[PAD]` (Index 1), `[SOS]` (Index 2), `[EOS]` (Index 3)

---

## Part 2: N-Gram Implementation & Text Generation Analysis

Three count-based N-gram language models (**2-gram**, **4-gram**, and **8-gram**) were implemented with $(n-1)$ `[SOS]` padding tokens at sentence starts and an `[EOS]` token at sentence endings.

### Comparative Analysis of Generated Texts (100 tokens):

1. **2-Gram (Bigram):**
   * **Characteristics:** Extremely low coherence, frequent topic shifts, and weak grammatical structure.
   * **Analysis:** Bigram models only consider the single preceding word context ($P(w_i \vert{} w_{i-1})$). As a result, local transitions between two adjacent words are occasionally valid, but overall sentences lack syntactic agreement, semantic logic, or structural persistence across a 100-token span.

2. **4-Gram (Quadgram):**
   * **Characteristics:** Moderate fluency, locally sound grammar, and recognizable children's story phrasing.
   * **Analysis:** By condition on a 3-word history ($P(w_i \vert{} w_{i-3}, w_{i-2}, w_{i-1})$), the 4-gram captures multi-word phrases, verb agreements, and short narrative arcs typical of TinyStories (e.g., character dialogues and simple actions). However, long-range global story coherence remains limited.

3. **8-Gram:**
   * **Characteristics:** High local fluency and flawless short sentence construction, but prone to structural rigidity.
   * **Analysis:** An 8-gram conditions on a 7-word history. Because an exact match for 7 consecutive words is rare in natural text, the 8-gram model largely memorizes and reproduces verbatim sequence passages directly from the training set. When it encounters a context with no matching 8-gram, it must restart or fail, showing high memorization with limited generalization.

---

## Part 3: Perplexity Evaluation & Candidate Sentence Ranking

### 1. Perplexity Results on Validation Set

Perplexity ($PPL$) measures how well a language model predicts a sample text. Lower perplexity indicates superior predictive performance.

$$\text{Perplexity}(W) = \exp \left( -\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \vert{} \text{context}_i) \right)$$

| Model Order ($n$) | Unsmoothed PPL | Laplace (Add-1) PPL |
| :---: | :---: | :---: |
| **2-Gram** | $\infty$ (`inf`) | **72.0676** |
| **4-Gram** | $\infty$ (`inf`) | **473.9390** |
| **8-Gram** | $\infty$ (`inf`) | **3279.6492** |

#### Theoretical & Empirical Analysis:
1. **Unsmoothed Models ($\text{PPL} = \infty$):**
   * Any unseen N-gram in the validation dataset has a count of zero ($C(\text{context}, w) = 0$), yielding $P(w \vert{} \text{context}) = 0$. Since $\ln(0) = -\infty$, the log-likelihood drops to negative infinity, causing the perplexity to evaluate to $\infty$.
2. **Laplace (Add-1) Smoothing Trends:**
   * Under Laplace smoothing ($P(w\vert{}\text{context}) = \frac{C(\text{context}, w) + 1}{C(\text{context}) + V}$), **higher-order N-grams yield higher perplexity scores** ($2\text{-gram: } 72.07 < 4\text{-gram: } 473.94 < 8\text{-gram: } 3279.65$).
   * **Explanation:** Higher-order contexts ($n=4, 8$) suffer from severe sparsity. When a high-order context has never been seen in training ($C(\text{context}) = 0$), Laplace smoothing assigns $P(w\vert{}\text{context}) = \frac{1}{V} = \frac{1}{5000} = 0.0002$. Adding $V=5000$ to the denominator severely dilutes the probability space and heavily penalizes unseen contexts, making naive Laplace smoothing ineffective for high-order N-grams.

---

### 2. Candidate Sentence Ranking Analysis

Using an **Interpolated 4-Gram Model**, the perplexity scores of four candidate sentences were evaluated to determine which sentences are most probable under the TinyStories dataset distribution:

| Rank | Sentence | Perplexity ($PPL$) |
| :---: | :--- | :---: |
| **1** | **جیل و تام به همراه مامان و بابا به ساحل رفتند** *(Sentence 2)* | **20.703** |
| **2** | **آنها دوست داشتند در ماسه بازی کنند و جزر و مد آب را تماشا کنند** *(Sentence 1)* | **44.044** |
| **3** | **تام بطری نوشابه را تا حد ممکن بالا انداخت و به سمت او فریاد زد** *(Sentence 3)* | **95.413** |
| **4** | **باری خیلی دوست داشت بیرون از منزل نقاشی کند و با پدربزرگ منظره تماشا کند** *(Sentence 4)* | **289.776** |

#### Linguistic & Probabilistic Justification:
* **Rank 1 (Sentence 2 - $PPL = 20.703$):** Contains ultra-high-frequency N-gram collocations and archetypal children's story structures (`به همراه مامان و بابا`, `به ساحل رفتند`). These contexts appear repeatedly in the training corpus, resulting in maximum conditional probabilities and the lowest perplexity.
* **Rank 2 (Sentence 1 - $PPL = 44.044$):** Follows standard narrative structures (`دوست داشتند در ... بازی کنند`, `تماشا کنند`), maintaining high probability despite slightly longer vocabulary combinations.
* **Rank 3 (Sentence 3 - $PPL = 95.413$):** Introduces less frequent n-gram combinations (`تا حد ممکن بالا انداخت`, `نوشابه`) that appear less frequently in TinyStories stories, increasing evaluation perplexity.
* **Rank 4 (Sentence 4 - $PPL = 289.776$):** Contains formal/uncommon phrasing (`بیرون از منزل`, `منظره تماشا کند`) that deviates from the simplified conversational subword transitions of the TinyStories-Farsi domain, yielding the highest perplexity.

---

## Part 4: Smoothing Techniques Analysis

Three smoothing algorithms were implemented for the 4-gram model:
1. **Laplace (Add-One) Smoothing:**
   $$P_{\text{Laplace}}(w\vert{}h) = \frac{C(h, w) + 1}{C(h) + V}$$
2. **Jelinek-Mercer Linear Interpolation:**
   $$P(w\vert{}h_3, h_2, h_1) = 0.4 \cdot P(w\vert{}h_3, h_2, h_1) + 0.3 \cdot P(w\vert{}h_2, h_1) + 0.2 \cdot P(w\vert{}h_1) + 0.1 \cdot P(w)$$
3. **Backoff ($\lambda = 0.4$):**
   $$P_{\text{Backoff}}(w\vert{}h) = \begin{cases}     \frac{C(h, w)}{C(h)} & \text{if } C(h, w) > 0 \\     0.4 \cdot P_{\text{Backoff}}(w\vert{}\text{lower\_context}) & \text{otherwise}     \end{cases}$$

### 1. Validation Set Perplexity Comparison

| Smoothing Method | Validation Perplexity ($PPL$) |
| :--- | :---: |
| **Unsmoothed (Basic)** | $\infty$ (`inf`) |
| **Laplace (Add-One)** | `473.9390` |
| **Linear Interpolation** | `30.1312` |
| **Backoff ($\lambda = 0.4$)** | **27.0948** |

---

### 2. Analysis of Generated Text Quality & Word Diversity

* **Unsmoothed (Basic):**
  * **Behavior:** Fails immediately upon reaching any unobserved context window during sampling ($P=0$).
* **Laplace (Add-One):**
  * **Behavior:** Generates incoherent, noisy word sequences (e.g., `جیل امواج بون انگشتان اینکه نیرو بذ سال حصار...`).
  * **Reason:** Because vocabulary $V=5000$ is large relative to context counts, Laplace over-allocates probability mass to rare/unseen tokens across all histories, destroying linguistic fluency.
* **Linear Interpolation:**
  * **Behavior:** Produces highly fluent, syntactically coherent text with smooth narrative transitions (e.g., `یک روز ، پسربچه ای بغل کرد و گفت : " بیایید اینجا بایست ی ! ...`).
  * **Reason:** Combining probabilities from 4-gram, 3-gram, 2-gram, and unigram models guarantees non-zero probabilities while prioritizing longer context matches when available.
* **Backoff:**
  * **Behavior:** Achieves the lowest overall perplexity (**27.0948**) and yields dynamic, fluent story generation.
  * **Reason:** Unlike interpolation, Backoff does not penalize strong 4-gram contexts by mixing them with lower-order probabilities; it utilizes the highest-order non-zero match directly and backs off only when context is missing.

---

## Part 5: Temperature Sampling Analysis

### 1. Theoretical Concept of Temperature ($T$)
Temperature ($T$) rescales the log-probabilities (logits) of candidate tokens before applying the Softmax function:

$$P(w_i) = \frac{\exp(\ln P(w_i) / T)}{\sum_j \exp(\ln P(w_j) / T)}$$

* **Low Temperature ($T \rightarrow 0$):** Magnifies the logit differences. The distribution becomes sharp and concentrated at the most probable token ($\text{argmax}$). Text generation becomes greedy, deterministic, and repetitive.
* **Standard Temperature ($T = 1.0$):** Preserves the exact learned probability distribution.
* **High Temperature ($T \rightarrow \infty$):** Flattens logit differences toward a uniform distribution. Every token in the vocabulary becomes equally likely, leading to high entropy, random sampling, and incoherent output.

---

### 2. Empirical Results (20-token Generations, 3 Runs Each)

#### **A. Low Temperature ($T = 0.03$)**
* **Sample 1:** `یک روز ، یک گربه بزرگ و یک سگ بزرگ و دوستانه ای به نام تیم بود . تیم دوست`
* **Sample 2:** `یک روز ، یک گربه بزرگ و یک سگ بزرگ و دوستانه ای به نام تیم بود . تیم دوست`
* **Sample 3:** `یک روز ، یک گربه بزرگ و یک سگ بزرگ و دوستانه ای به نام تیم بود . تیم دوست`
* **Analysis:** All 3 generations are **100% identical**. At $T=0.03$, sampling collapses into deterministic argmax selection, generating the single most frequent sequence continuously without variation.

#### **B. High Temperature ($T = 3.0$)**
* **Sample 1:** `یه زنجیر ed رنگی شادتر کمک بوسه پیدایش نمودند خواب کوچکش حم بسازد دور میومد دیوارها ؟ پید ترو ببینه`
* **Sample 2:** `ُ هیجان گرمی کوچولو برداشتن دوستشان بزرگش نوشیدن un عادی چند سوئی اونا مشکلات ا گزار رعایت ن قفسه بهترین`
* **Sample 3:** `جغد مورد ز سنگ طنا بدهد لوسی پوست خوش دند هیچکس m جویدن زه к شیطان ماندند قص حرکت چار`
* **Analysis:** All 3 outputs are **completely different and chaotic**. At $T=3.0$, probability distribution approaches uniformity, forcing the model to pick random subwords and characters, resulting in zero semantic structure and pure noise.

---

## Conclusion
* **N-gram Order Trade-off:** Higher $n$ increases local fluency but suffers from extreme sparsity.
* **Smoothing Impact:** Linear Interpolation and Backoff resolve zero-frequency issues, reducing perplexity from $\infty$ down to $\sim 27-30$.
* **Temperature Dynamics:** $T \approx 0.7 - 1.0$ balances creativity and semantic coherence, whereas extreme values lead to deterministic repetition ($T \to 0$) or random noise ($T \to \infty$).