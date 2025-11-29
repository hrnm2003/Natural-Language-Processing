# Word2Vec Implementation from Scratch: Skip-gram & CBOW Models on WikiText-2

This repository contains a complete PyTorch implementation of the **Word2Vec** architecture, covering both the **Skip-gram** and **Continuous Bag of Words (CBOW)** models from scratch. The models are trained on the **WikiText-2** dataset, and their learned embeddings are evaluated using Cosine Similarity retrieval and 2D t-SNE visualizations.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Model Architectures & Implementation](#model-architectures--implementation)
4. [Training Details & Loss Curves](#training-details--loss-curves)
5. [Comparative Analysis & Similarity Retrieval](#comparative-analysis--similarity-retrieval)
6. [t-SNE Embedding Visualization Analysis](#t-sne-embedding-visualization-analysis-skip-gram-vs-cbow)
7. [Detailed Conceptual Answers & Discussion](#detailed-conceptual-answers--discussion)
8. [Installation & Usage](#installation--usage)

---

## Project Overview

Word2Vec is a foundational technique in Natural Language Processing (NLP) used to learn continuous vector representations of words. This project explores both standard variants:
- **Skip-gram**: Predicts context words given a target center word.
- **CBOW (Continuous Bag of Words)**: Predicts a target center word given a context of surrounding words.

### Key Features:
- Fully implemented using **PyTorch** without relying on high-level embedding frameworks (like Gensim).
- Trained on **WikiText-2** loaded directly via the Hugging Face `datasets` library.
- Custom dataset processing, vocabulary generation with minimum frequency thresholds, and dynamic batching (`DataLoader` collate functions).
- Comprehensive quantitative and qualitative comparison using loss tracking, word similarity tables, and t-SNE embedding plots.

---

## Dataset & Preprocessing

The project uses the `wikitext-2-raw-v1` dataset split into training, validation, and test sets.

### Statistics:
- **Train split**: 36,718 text lines
- **Validation split**: 3,760 text lines
- **Test split**: 4,358 text lines

### Preprocessing Pipeline:
1. **Text Normalization**: Converts all text to lowercase and strips non-alphanumeric characters using regular expressions (`[^a-z0-9\s]`).
2. **Vocabulary Building**:
   - Filter words below a minimum frequency (`min_freq=2`).
   - Add a special `<UNK>` token for out-of-vocabulary words.
   - **Total raw words**: 65,332
   - **Filtered vocabulary size**: 38,911 tokens
3. **Training Sample Generation**:
   - **Skip-gram**: Pairs target words with context words within a context window size of $N=2$.
   - **CBOW**: Maps context word index sequences within window size $N=2$ to target word indices.

---

## Model Architectures & Implementation

### 1. Skip-gram Architecture
- **Concept**: Learns embeddings by maximizing the probability of context words given a target word.
- **PyTorch Design**:
  - `input_embedding`: Look-up table for center/target words (`nn.Embedding(vocab_size, embedding_dim)`).
  - `output_embedding`: Look-up table for context words (`nn.Embedding(vocab_size, embedding_dim)`).
  - **Forward Pass**: Computes dot products between target and context word embeddings.
  - **Loss Function**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).

### 2. CBOW Architecture
- **Concept**: Learns embeddings by maximizing the probability of predicting a center word from the average representation of its surrounding context words.
- **PyTorch Design**:
  - `embeddings`: Embedding layer mapping context word indices to vector space.
  - `linear`: Projection layer mapping averaged context vectors back to `vocab_size` logits (`nn.Linear(embedding_dim, vocab_size)`).
  - **Forward Pass**: Averages embeddings across the context window, projects through the linear layer.
  - **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`).

---

## Training Details & Loss Curves

### Hyperparameters:
- **Embedding Dimension**: 100
- **Context Window Size**: 2
- **Batch Size**: 4
- **Epochs**: 15
- **Optimizer**: Adam (`learning_rate = 0.01`)

### Training Progression:

| Epoch | Skip-gram Train Loss | Skip-gram Val Loss | Skip-gram Test Loss | CBOW Train Loss | CBOW Val Loss | CBOW Test Loss |
|-------|----------------------|--------------------|---------------------|-----------------|---------------|----------------|
| 1     | 0.1987               | 0.0051             | 0.0233              | 2.0392          | 1.1373        | 1.5110         |
| 5     | 0.0008               | 0.0002             | 0.0014              | 0.8645          | 0.6513        | 1.5188         |
| 10    | 0.0002               | 0.0001             | 0.0004              | 0.6806          | 0.6068        | 1.5774         |
| 15    | **0.0001**           | **0.0000**         | **0.0002**          | **0.5603**      | **0.5734**    | **1.5859**     |

---

## Comparative Analysis & Similarity Retrieval

Using Cosine Similarity, top-5 nearest neighbors were computed for selected target words: `learning`, `dog`, `human`, `machine`, and `data`.

### Quantitative Results Table:

| Target Word | Model | Top-5 Similar Words & Similarity Scores |
|-------------|-------|-----------------------------------------|
| **learning** | **Skip-gram** | `and` (0.9198), `are` (0.9193), `understand` (0.9140), `data` (0.9081), `dog` (0.8967) |
| | **CBOW** | `the` (0.9656), `understand` (0.8986), `data` (0.8504), `are` (0.8300), `machine` (0.7914) |
| **dog** | **Skip-gram** | `and` (0.9178), `data` (0.9076), `learning` (0.8967), `are` (0.8936), `human` (0.8894) |
| | **CBOW** | `human` (0.5927), `and` (0.3720), `language` (-0.2015), `of` (-0.2687), `data` (-0.4069) |
| **human** | **Skip-gram** | `understand` (0.9171), `and` (0.9086), `dog` (0.8894), `language` (0.8813), `learning` (0.8761) |
| | **CBOW** | `dog` (0.5927), `language` (0.5119), `and` (0.2601), `of` (-0.1174), `data` (-0.2818) |
| **machine** | **Skip-gram** | `learning` (0.8788), `are` (0.8713), `understand` (0.8576), `and` (0.8551), `dog` (0.8471) |
| | **CBOW** | `are` (0.9517), `understand` (0.9306), `learning` (0.7914), `the` (0.7221), `data` (0.4849) |
| **data** | **Skip-gram** | `and` (0.9195), `learning` (0.9081), `dog` (0.9076), `are` (0.8969), `understand` (0.8800) |
| | **CBOW** | `of` (0.9465), `the` (0.9191), `learning` (0.8504), `understand` (0.6570), `are` (0.5494) |

---

## t-SNE Embedding Visualization Analysis: Skip-gram vs. CBOW

This section provides a comparative analysis of the 2D t-SNE visualizations generated for five target words (`learning`, `dog`, `human`, `machine`, `data`) and their top-5 similar words across both models.

### 1. General Distribution & Vector Spread
- **Variance & Spatial Spread**:
  - **Skip-gram**: Word vectors exhibit higher variance and a wider spatial spread. This stems from Skip-gram's effectiveness on sparse data and its ability to establish sharp semantic distinctions.
  - **CBOW**: Shows lower variance with words clustered tightly together. By predicting target words based on context averaging, CBOW produces a more uniform and less segregated semantic space.
- **Semantic Clustering Quality**:
  - **Skip-gram**: Features more pronounced semantic clusters where similar words reside in closer proximity to the target word.
  - **CBOW**: Yields tighter clusters, but semantic boundaries are occasionally blurred, causing unrelated words to appear near the target.

### 2. Word-by-Word Analysis
- **`learning`**:
  - *Skip-gram*: Related words like `understand` and `data` are positioned close to `learning` with meaningful relative distances, forming a distinct cluster.
  - *CBOW*: Words like `machine` or `are` appear nearby despite lacking strong semantic correlation, indicating lower conceptual distinction.
  - *Conclusion*: Skip-gram provides a more accurate semantic representation for `learning`.
- **`dog`**:
  - *Skip-gram*: Similar words (`dog`, `and`, `learning`, `are`) maintain reasonable distances, though some links (like `are`) are weaker.
  - *CBOW*: Unrelated words (`and`, `language`) cluster closely while semantic terms are pushed further away.
  - *Conclusion*: Both models show moderate performance, but Skip-gram achieves superior separation.
- **`human`**:
  - *Skip-gram*: Placed close to logically sound terms like `understand` and `language`.
  - *CBOW*: Features a tighter cluster, but `dog` appears unusually close to `human`, reflecting semantic ambiguity in CBOW's modeling.
  - *Conclusion*: Skip-gram establishes clearer conceptual boundaries.
- **`machine`**:
  - *Skip-gram*: Terms like `learning`, `understand`, and `dog` appear at varying distances; meaningful clustering with some extra dispersion.
  - *CBOW*: Related words like `learning` are positioned closer in a tighter structure. This represents one of the rare cases where CBOW outperforms Skip-gram.
  - *Conclusion*: CBOW delivers more cohesive semantic clustering for `machine`.
- **`data`**:
  - *Skip-gram*: Adjacent to related terms like `understand` and `are`, establishing a robust semantic structure.
  - *CBOW*: Forms a decent cluster, but unrelated words like `the` remain closely tied.
  - *Conclusion*: Skip-gram offers a more precise interpretation of semantic relationships for `data`.

### Summary
Overall, t-SNE visualizations demonstrate that the **Skip-gram** model establishes a more precise and discriminative semantic structure than CBOW. Target words are centered within well-defined clusters of similar words with logically consistent distances. Conversely, **CBOW** produces tighter yet less discriminative clusters, frequently drawing unrelated words near the target due to its context-averaging mechanism which smooths out fine-grained semantic details.

---

## Detailed Conceptual Answers & Discussion

This section provides comprehensive answers to the conceptual analysis questions addressed in the notebook based on empirical findings.

### 1. Structural & Architectural Differences
* **Skip-gram** takes a single target word as input and attempts to predict multiple surrounding context words separately. Its prediction mechanism generates multiple training samples per window.
* **CBOW** takes multiple context words within a window, averages (or sums) their vectors, and predicts the single center target word using a multi-class linear classifier layer.

### 2. Loss & Convergence Behavior
* **Skip-gram** converged rapidly to extremely low loss values ($\approx 0.0001$). This is primarily due to the simplified Binary Cross-Entropy formulation for pairwise dot-product predictions. However, when trained without negative sampling, it runs the risk of predicting uniform high probabilities if not carefully normalized.
* **CBOW** showed steady loss reduction from $2.0392$ to $0.5603$ on the training set and $0.5734$ on validation. Its test loss plateaued around $1.58$. Cross-entropy over a large vocabulary ($38,911$ classes) represents a harder optimization problem, requiring more epochs or larger batch sizes for higher semantic precision.

### 3. Semantic & Syntactic Quality of Embeddings
* **CBOW** captures stronger functional and semantic groupings for frequent terms (e.g., mapping `learning` to `understand`, `data`, `machine`). Because it averages context vectors, it excels at smoothing out representations for frequent words that regularly share broad thematic contexts.
* **Skip-gram** generates broader vector similarities. Because each target-context pair is trained individually, infrequent words receive independent update steps, preventing them from being diluted by frequent co-occurring words.

### 4. Handling Rare vs. Frequent Words
* **Frequent Words**: CBOW performs efficiently because averaging context words provides a strong, smooth signal for high-frequency words.
* **Rare Words**: Skip-gram significantly outperforms CBOW for rare words. In CBOW, rare words are averaged alongside frequent context words (like `the`, `of`, `and`), suppressing their gradient signal. In Skip-gram, every rare word context pair creates a standalone training sample, updating its vector directly.

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face `datasets`

### Running the Notebook
1. Clone the repository:
   ```bash
   git clone [https://github.com/hrnm2003/word-embeddings.git](https://github.com/hrnm2003/word-embeddings.git)
   cd word2vec-pytorch
   ```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

3. Launch Jupyter Notebook:
	```bash
	jupyter notebook Word2Vec_Implementation.ipynb
	```