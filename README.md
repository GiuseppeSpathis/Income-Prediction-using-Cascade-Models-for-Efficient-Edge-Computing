# Income Prediction with Cascade Inference Models

## 📌 Project Overview
This project explores the use of **Cascade Models** to improve efficiency in Edge Computing environments. The goal is to perform a regression task—predicting an individual's income based on census data—while balancing **accuracy** and **resource consumption** (inference time, bandwidth, energy).

Instead of relying solely on a heavy deep learning model or a lightweight linear model, this system uses a **Confidence-Based Cascade**:
1.  **Stage 1 (Edge):** A lightweight probabilistic model makes a quick prediction and estimates its own uncertainty ($\sigma$).
2.  **Decision Gate:** If the uncertainty is below a threshold, the prediction is accepted.
3.  **Stage 2 (Cloud):** If uncertainty is high, the data is offloaded to a larger, more accurate Neural Network.

## 📊 Dataset
* [cite_start]**Name:** ACSIncome (Folktables) [cite: 109]
* [cite_start]**Source:** US Census Bureau (American Community Survey) [cite: 110]
* [cite_start]**Scope:** Data from 50 U.S. states and Puerto Rico (2014–2018) [cite: 110]
* [cite_start]**Features:** 15 demographic features including Age, Occupation, Education Level, Marital Status, etc. [cite: 110]
* [cite_start]**Target:** Annual Income (Regression task) [cite: 109]

## 🧠 Methodology

### 1. Probabilistic Regression
Standard regression models predict a single mean value ($\mu$). To enable cascade logic, we implement **Probabilistic Regression** where the models output both a mean ($\mu$) and a standard deviation ($\sigma$) representing uncertainty.
* [cite_start]**Loss Function:** Gaussian Negative Log-Likelihood (NLL) [cite: 118]
    $$\text{Loss} = \frac{\log(\sigma^2)}{2} + \frac{(y - \mu)^2}{2\sigma^2}$$

### [cite_start]2. Model Architectures [cite: 122, 124]
* **Small Model (The "Edge" Model):** A simple Linear Regression model modified to output uncertainty. Optimized for speed.
* **Large Model (The "Cloud" Model):** A Multi-Layer Perceptron (MLP) with hidden layers (e.g., [512, 512, 256]). Optimized for accuracy.

### [cite_start]3. The Cascade System [cite: 101, 102]
The system processes inputs as follows:
1.  Input $x$ $\rightarrow$ Small Model $\rightarrow$ ($\mu_{small}, \sigma_{small}$)
2.  **Check Confidence:** Is $\sigma_{small} < \text{Threshold}$?
    * **Yes:** Return $\mu_{small}$ (Fast, Cheap)
    * **No:** Send $x$ $\rightarrow$ Large Model $\rightarrow$ Return $\mu_{large}$ (Slow, Expensive)

## 🚀 Results
* **Small Model Error (MAE):** ~$56,755 (High error, very fast)
* **Large Model Error (MAE):** ~$42,178 (Low error, slow)
* **Cascade Performance:**
    * By offloading only the **~20%** most uncertain cases to the Large Model:
    * **Speedup:** ~4.7x faster than using the Large Model alone.
    * **Bandwidth Savings:** 80% of data remains local.
    * **Error:** ~$54,852 (Significant improvement over the standalone linear model).

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.x
* PyTorch
* Scikit-Learn
* Pandas / NumPy
* Folktables

### Running the Project
1.  **Install Dependencies:**
    ```bash
    pip install torch pandas numpy scikit-learn folktables matplotlib seaborn
    ```
2.  **Run the Notebook:**
    Open `IncomePrediction_Cascade.ipynb` in Google Colab or Jupyter Notebook.

## 📂 Project Structure
1.  **Data Prep:** Downloads ACSIncome data using `folktables`, filters for valid entries, and performs One-Hot Encoding.
2.  **Model Definition:** Defines `ProbabilisticModel` class (dynamic architecture).
3.  **Training:** Trains the Linear (Small) and MLP (Large) models using Gaussian NLL Loss.
4.  [cite_start]**Cascade Simulation:** Tests various confidence thresholds to find the optimal trade-off between error and computational cost[cite: 126].
5.  **Visualization:** Plots the "Error vs. Cost" trade-off curve to justify the cascade strategy.

## 📚 References
* **Dataset:** Ding et al. (2021/2024) [cite_start]- *Retiring Adult: New Datasets for Fair Machine Learning* [cite: 109]
* [cite_start]**Cascade Context:** Viola, P., & Jones, M. (2001) - *Rapid object detection using a boosted cascade of simple features* [cite: 113]
