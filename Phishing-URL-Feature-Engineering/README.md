# Phishing URL Detection using Machine Learning

## Project Overview
This project focuses on identifying phishing URLs by extracting various numerical features directly from raw URLs and evaluating them using Machine Learning models. The primary goal is to compare the performance of **Logistic Regression** and **Naive Bayes (Gaussian and Multinomial)** algorithms across different feature sets to understand the impact of feature engineering, data normalization, and model assumptions.

The project is divided into four main feature extraction phases and a final comprehensive analysis.

## Dataset
The project relies on a dataset (`urls.csv`) containing raw URLs and their respective status:
*   `legitimate` (mapped to `0`)
*   `phishing` (mapped to `1`)

---

## 🔬 Methodology & Step-by-Step Analysis

### Part 1: Structural Features
In this phase, simple occurrence counts were extracted from the URLs:
- `nb_dots`: Number of dots (.)
- `nb_slashes`: Number of slashes (/)
- `nb_hyphens`: Number of hyphens (-)

**Q&A Analysis:**
* **Is normalization necessary for these features?**
  No, normalization is not recommended when using the **Multinomial Naive Bayes (MNB)** model. MNB is specifically designed for discrete occurrence counts. Normalizing these values disrupts the underlying probability calculations that rely on actual feature frequencies. However, for Logistic Regression and Gaussian NB, standard scaling was applied to maintain numerical stability.
* **Which Naive Bayes variant (Gaussian vs. Multinomial) is more appropriate here?**
  Theoretically, **Multinomial Naive Bayes** is the most suitable variant for this specific dataset subset because the extracted features represent the exact counts of characters (discrete data). 

### Part 2: Statistical and Content Features
The following continuous features were extracted:
- `length_url`: Total length of the URL.
- `ratio_digits_url`: The ratio of digits (0-9) to the total URL length.
- `longest_words_raw`: The length of the longest alphabetical word separated by non-alphanumeric delimiters.

**Q&A Analysis:**
* **Is it logical to use Naive Bayes for these features? Which version is better?**
  While possible, standard Naive Bayes is not the ideal theoretical fit because these features are continuous and proportional (not discrete counts). However, among the variants, **Gaussian Naive Bayes** is the most logical choice. Gaussian NB assumes that continuous features follow a normal distribution, making it mathematically better suited for lengths and ratios compared to the Multinomial variant.

### Part 3: Creative Features
To capture more complex phishing behaviors, three custom features were engineered:
- `domain_length`: The character length of the domain.
- `is_ip_address`: A binary feature indicating if the domain is a raw IP address.
- `path_length`: The length of the URL path after the domain.

**Q&A Analysis:**
* **Why are these features effective in detecting phishing?**
  * **`is_ip_address`:** Legitimate websites almost always use registered domain names. Attackers often use raw IP addresses to bypass domain-based blacklists or to set up temporary malicious hosts.
  * **`domain_length`:** Phishing URLs frequently utilize unusually long domains. Attackers do this to spoof legitimate brand names (e.g., `paypal-secure-login-update.com`) or to hide the actual root domain on smaller screens.
  * **`path_length`:** Malicious URLs often have extremely long and complex paths containing random hashes, deep directories, or obfuscated payloads designed to mimic complex legitimate site structures or bypass simple filters.

### Part 4: Combining All Features
All 9 features (3 Structural + 3 Statistical + 3 Creative) were combined into a single dataset to train and evaluate the models.

**Q&A Analysis:**
* **Did combining all features improve model performance?**
  Combining the features **significantly improved the Logistic Regression model** (achieving the highest Accuracy of ~72.6% and F1-Score of ~69.9%). However, it **degraded or stagnated the Naive Bayes models' performance** (Gaussian NB F1-Score dropped heavily to ~43.8%).
* **Why did this happen?**
  The degradation in Naive Bayes is due to its core assumption of **feature independence**. Many of our combined features are highly correlated (e.g., `length_url` is naturally correlated with `path_length` and `domain_length`). When correlated features are fed to Naive Bayes, it essentially "double-counts" their importance, leading to poor predictions. Logistic Regression, on the other hand, does not assume independence and can learn optimal weights for correlated features, leading to improved performance.

---

## Part 5: Final Conclusion and Findings

Based on the empirical results of this project:

1. **Which model performed best and why?**
   **Logistic Regression** was the clear winner. Unlike Naive Bayes, it does not strictly assume that features are statistically independent. Since URL features are often highly intertwined (e.g., total length and path length), Logistic Regression successfully balanced these relationships and handled the mix of binary, continuous, and discrete data seamlessly.

2. **Which features had the most and least impact?**
   Based on the final Logistic Regression coefficients:
   * **Most Impactful:** Structural length features like `path_length` (coefficient: ~1.57) and `domain_length` (coefficient: ~1.15) had the highest positive weights. This proves that phishers heavily rely on long, obfuscated structures.
   * **Least Impactful:** Simple character counts like `nb_dots` (coefficient: ~-0.06) had weights closest to zero. These basic structural elements are too common in both legitimate and phishing URLs to be strong standalone indicators.

3. **Was normalization necessary?**
   **Yes, specifically for Logistic Regression.** The dataset contained features with vastly different scales (e.g., `is_ip_address` is binary [0,1], while `path_length` can be >100). Without standard scaling (Z-score normalization), features with larger numeric ranges would disproportionately dominate the Logistic Regression gradient. However, normalization was deliberately avoided for the Multinomial Naive Bayes model to preserve the integrity of the count-based data.

4. **Alternative approaches for phishing detection:**
   Beyond URL feature extraction, a robust anti-phishing system could utilize:
   * **Content Analysis:** Scraping the HTML/JavaScript of the page to detect spoofed login forms or hidden iframes.
   * **Third-Party Integrations:** Utilizing WHOIS data (domain age, registrar anomalies) and SSL certificate validation.
   * **Deep Learning:** Applying sequence models like LSTMs (Long Short-Term Memory) or Transformers directly on the raw URL character sequences, allowing the model to automatically learn latent patterns without manual feature engineering.

---

## How to Run

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `urls.csv` is in the same directory as the Jupyter Notebook.
4. Run `Phishing_Feature_Extraction_and_Modeling.ipynb` to see the step-by-step feature extraction and model evaluation.