# AG News Text Classification using FastText & PyTorch FNN

This project implements an end-to-end text classification pipeline for the **AG News** dataset using **FastText pre-trained word subword embeddings** and a **PyTorch Feedforward Neural Network (FNN)**. The workflow covers balanced data sampling, text preprocessing, subword-aware embedding extraction, model training, and an in-depth performance evaluation.


## Table of Contents
1. [Overview](#overview)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Text Embedding](#text-embedding)
4. [Network Implementation and Training](#network-implementation-and-training)
5. [Results Analysis](#results-analysis)
6. [Getting Started & Usage](#requirements-&-installation)


## Data Loading and Preprocessing

### Implementation Details
* **Dataset Source:** The AG News dataset is loaded via the Hugging Face `datasets` library (`SetFit/ag_news`), categorizing news articles into four distinct classes: *World* (0), *Sports* (1), *Business* (2), and *Sci/Tech* (3).
* **Balanced Sampling:** To ensure fair representation and prevent class imbalance issues, a balanced subset is sampled:
  * **Training Set:** 5,000 samples total (1,250 samples per class).
  * **Test & Validation Splits:** 1,000 test samples and 1,000 evaluation samples (500 samples per class, stratified split).
* **Text Preprocessing:** A custom preprocessing pipeline standardizes the text corpus by converting all characters to lowercase and removing extra whitespace.

### Why Minimal Preprocessing is Required with FastText
A key characteristic of this pipeline is that **heavy text preprocessing (such as heavy tokenization, stemming, lemmatization, or aggressive stopword removal) is unnecessary**. This is due to the fundamental design of the **FastText** embedding model:
1. **Subword (Character N-Gram) Representation:** Unlike traditional word embeddings (like Word2Vec or GloVe) that treat words as atomic units, FastText represents each word as a bag of character n-grams. 
2. **Handling Out-of-Vocabulary (OOV) and Morphological Variations:** Because words are broken down into subwords, the model can construct vector representations for misspelled words, rare words, suffixes, and prefixes it has never seen before during training. 
3. **Robustness to Noise:** Minor typographical errors, punctuation variations, or unstemmed words do not result in missing vectors (OOV), allowing the downstream neural network to capture semantic meaning directly from raw text with minimal cleaning.


## Text Embedding

* **FastText Model:** Utilizes the pre-trained FastText English subword embedding model (`wiki-news-300d-1M-subword.bin`), mapping words to 300-dimensional vectors.
* **Sentence-Level Representation:** Sentence embeddings are computed by taking the element-wise arithmetic mean of all constituent word vectors in each text instance, yielding uniform feature matrices of shape `(N, 300)`.


## Network Implementation and Training

* **DataLoaders:** Embedded features and target labels are wrapped in PyTorch `TensorDataset` and batched using `DataLoader` with a batch size of `64`.
* **Architecture (`NewsClassifier`):** A custom PyTorch Feedforward Neural Network comprising:
  * **Input Layer:** 300 units (matching FastText embedding dimension).
  * **Hidden Layer:** Linear layer mapping from `300` to `128` units, followed by a non-linear **ReLU** activation function.
  * **Output Layer:** Linear layer mapping from `128` units to `4` output classes.
* **Training Configuration:**
  * **Loss Function:** `CrossEntropyLoss`
  * **Optimizer:** `Adam` optimizer with a learning rate of `0.001` over 30 epochs.


## Results Analysis

### Overall Performance Summary
* **Test Accuracy:** `87.40%`
* **Test Macro-F1 Score:** `87.37%`

### Class-Wise Performance Metrics
| Class ID | Class Name | Accuracy | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | World | 0.8640 | 0.8816 | 0.8640 | 0.8727 | 250 |
| **1** | Sports | **0.9480** | 0.9258 | 0.9480 | **0.9368** | 250 |
| **2** | Business | 0.8880 | 0.8073 | 0.8880 | 0.8457 | 250 |
| **3** | Sci/Tech | **0.7960** | 0.8884 | 0.7960 | 0.8397 | 250 |

### Detailed Answers to Evaluation Questions

#### 1. Best and Worst Performing Classes
* **Best Performing Class:** **Class 1 (Sports)** achieved the highest accuracy (**94.80%**) and F1-score (**93.68%**). Sports articles feature highly distinctive, domain-specific vocabulary (e.g., player names, match outcomes, league terminology) that rarely overlaps with other domains, making them exceptionally easy for the classifier to separate.
* **Worst Performing Class:** **Class 3 (Sci/Tech)** exhibited the lowest accuracy (**79.60%**). Technological news frequently intersects with corporate business announcements, product launches, and economic ventures, leading to inherent semantic ambiguity.

#### 2. Most Confused Classes
* The confusion matrix indicates that the model most frequently confuses **Class 3 (Sci/Tech)** with **Class 2 (Business)**. For instance, **33 true Sci/Tech samples** were misclassified as Business. 
* *Reason:* Technological advancements, startup funding, and corporate tech strategies share a heavy vocabulary overlap with commercial and financial news, blurring the boundary between business reporting and technological reporting.

#### 3. Misclassification Inspection & Analysis
Inspecting specific misclassified instances reveals the underlying semantic overlap:
* **Example 1 (Sports misclassified as Business):**
  * *True Label:* Sports (1) | *Predicted Label:* Business (2)
  * *Content Context:* News articles discussing legal petitions, corporate governance, or financial disputes involving sports networks or franchises (e.g., broadcast rights litigation). The presence of legal and financial terminology misleads the model into predicting Business.
* **Example 2 (Business misclassified as Sci/Tech):**
  * *True Label:* Business (2) | *Predicted Label:* Sci/Tech (3)
  * *Content Context:* Corporate announcements regarding hardware releases, retail tech strategies (e.g., HP/Dell holiday product lines), or tech-sector corporate earnings. The heavy focus on technical specifications causes the model to lean toward Sci/Tech.
* **Example 3 (Sci/Tech misclassified as Business):**
  * *True Label:* Sci/Tech (3) | *Predicted Label:* Business (2)
  * *Content Context:* Articles covering telecommunications market mergers, internet service provider business expansions, or tech company stock performances. Although categorized under technology, the focal point of the text is market competition and corporate revenue.

## Requirements & Installation

Make sure you have the required libraries installed:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Clone this repository or open the `.ipynb` file in Google Colab / Jupyter Notebook.
2. Ensure your Hugging Face token is set up in your environment (`HUGGINGFACE_TOKEN`) if required for dataset downloads.
3. Run all cells sequentially to reproduce the data loading, FastText embedding extraction, model training curves, and evaluation metrics.