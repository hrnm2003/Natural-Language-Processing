# Text Classification from Scratch: Logistic Regression & Naive Bayes

A complete, library-free implementation of **Logistic Regression** and **Multinomial Naive Bayes** for binary text classification (Spam vs. Ham). This project demonstrates text preprocessing, Bag-of-Words feature extraction, custom metric definitions, and model implementation using pure Python and NumPy.

---

## Table of Contents
1. [Project Overview](#-project-overview)
2. [Datasets](#-datasets)
3. [Workflow & Implementation](#-workflow--implementation)
   - [Part 1: Text Preprocessing & Bag of Words](#part-1-text-preprocessing--bag-of-words)
   - [Part 2: Train/Test Data Split](#part-2-traintest-data-split)
   - [Part 3: Evaluation Metrics from Scratch](#part-3-evaluation-metrics-from-scratch)
   - [Part 4: Logistic Regression Model](#part-4-logistic-regression-model)
   - [Part 5: Multinomial Naive Bayes Model](#part-5-multinomial-naive-bayes-model)
   - [Part 6: Comparative Results & In-Depth Analysis](#part-6-comparative-results--in-depth-analysis)
4. [Experimental Results Summary](#-experimental-results-summary)

---

## Project Overview

The objective of this project is to build and evaluate two foundational text classification algorithms **from scratch** without relying on high-level machine learning libraries like `scikit-learn` for model training or metric computations.

- **Primary Task**: Binary email classification ($\text{Spam} = 1$, $\text{Ham} = 0$).
- **Core Models**:
  - Logistic Regression (trained via Gradient Descent)
  - Multinomial Naive Bayes (with Laplace Smoothing)
- **Key Concepts Covered**: Bag-of-Words (BoW) text representation, Feature Scaling, Evaluation Metrics ($Accuracy$, $Precision$, $Recall$, $F_1\text{-Score}$), Model Diagnostics.

---

## Datasets

1. **`emails_1.csv`**: Used for initial preprocessing demonstration. Contains raw text and `status` labels (`spam`/`ham`).
2. **`emails_2.csv`**: The primary dataset containing **5,172 emails** represented in a Bag-of-Words matrix across **3,000 top frequent words**, along with an `Email No.` identifier and a target label (`Prediction`: `0` for Ham, `1` for Spam).

---

## Workflow & Implementation

### Part 1: Text Preprocessing & Bag of Words
- **Label Encoding**: Converted string targets (`spam` $\rightarrow 1$, `ham` $\rightarrow 0$).
- **Text Cleaning**: Tokenized text while removing punctuation, non-alphabetic characters, and converting all tokens to lowercase.
- **Feature Extraction**: Created a Bag-of-Words feature matrix selecting the top 15 most frequent tokens for demonstration using `CountVectorizer`.

### Part 2: Train/Test Data Split
- Implemented a custom `train_test_split` function using random index permutations (`random_state=42`).
- **Dataset Partition**:
  - **Total Samples**: 5,172
  - **Training Set (80%)**: 4,138 samples
  - **Testing Set (20%)**: 1,034 samples

### Part 3: Evaluation Metrics from Scratch
All metrics were implemented using pure `NumPy` operations:

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

$$ \text{Precision} = \frac{TP}{TP + FP} $$

$$ \text{Recall} = \frac{TP}{TP + FN} $$

$$ F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

---

### Part 4: Logistic Regression Model
- Implemented a custom `StandardScaler` to scale features ($\mu = 0, \sigma = 1$).
- Implemented `LogisticRegression` using binary cross-entropy loss and standard gradient descent updates:
  - **Activation**: Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - **Hyperparameters**: $\text{Learning Rate} = 0.01$, $\text{Iterations} = 1000$

---

### Part 5: Multinomial Naive Bayes Model

#### Theoretical Background & Q&A

> **Q1: Why did we use Multinomial Naive Bayes instead of Gaussian Naive Bayes?**  
> **Answer:**  
> Text features represented as Bag-of-Words vectors consist of **discrete term frequencies (counts)**. Multinomial Naive Bayes is specifically formulated to model discrete event counts over a multinomial distribution:
> 
> $$ P(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot n_{\text{features}}} $$
> 
> In contrast, **Gaussian Naive Bayes** assumes features follow a continuous Normal (Gaussian) distribution. Text data matrices are sparse and highly skewed, making Gaussian assumptions inappropriate and detrimental to model performance.

> **Q2: Can other Naive Bayes variants be used for text classification?**  
> **Answer:**  
> Yes:
> 1. **Bernoulli Naive Bayes**: Suitable when features are binary indicators (word presence/absence rather than counts).
> 2. **Complement Naive Bayes**: Optimized for imbalanced datasets, often outperforming standard Multinomial NB in text classification tasks where class sizes differ significantly.

---

### Part 6: Comparative Results & In-Depth Analysis

#### Performance Comparison Table

| Metric | Logistic Regression | Multinomial Naive Bayes | Best Model |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **96.81%** | 95.36% | **Logistic Regression** |
| **Precision** | **94.28%** | 88.75% | **Logistic Regression** |
| **Recall** | 94.59% | **95.95%** | **Multinomial Naive Bayes** |
| **F1-Score** | **94.44%** | 92.21% | **Logistic Regression** |

---

#### Comprehensive Technical Analysis

1. **Precision vs. Recall Trade-off**:
   - **Multinomial Naive Bayes** achieves a higher **Recall (95.95%)** than Logistic Regression (94.59%). This indicates that Naive Bayes is extremely sensitive to spam indicators and catches almost all positive spam emails. However, this comes at the cost of **Precision (88.75%)**, producing a higher rate of **False Positives** (legitimate emails incorrectly flagged as spam).
   - **Logistic Regression** achieves significantly higher **Precision (94.28%)** while maintaining high Recall (94.59%). This balance produces a superior overall **F1-Score (94.44%)**.

2. **Feature Independence Assumption vs. Linear Optimization**:
   - **Naive Bayes** makes the strong assumption that word occurrences are conditionally independent given the class label ($P(w_1, w_2 \mid y) = P(w_1 \mid y) P(w_2 \mid y)$). In natural language, words frequently co-occur (e.g., "claim" and "free"), violating this assumption and slightly overestimating class likelihoods.
   - **Logistic Regression** optimizes a linear decision boundary directly by learning feature weights. It naturally adjusts for correlations among correlated features during gradient optimization.

3. **Domain Recommendation (Spam Filtering)**:
   - In real-world email systems, a **False Positive** (a legitimate email delivered to Spam) is generally far more damaging to a user than a **False Negative** (a spam email delivered to the Inbox).
   - Therefore, **Logistic Regression** is the preferred model for this application due to its substantially lower false-positive rate and higher precision.

---

## Experimental Results Summary

Execution output logs from the experimental notebook:

```text
Logistic Regression Metric Results:
-----------------------------------
Accuracy:  96.81%
Precision: 94.28%
Recall:    94.59%
F1 Score:  94.44%

Multinomial Naive Bayes Results:
-----------------------------------
Accuracy:  95.36%
Precision: 88.75%
Recall:    95.95%
F1 Score:  92.21%
```