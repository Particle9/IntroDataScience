# Intro to Data Science Cheat Sheet - Organized

This repository contains a comprehensive Data Science cheat sheet, now **reorganized into 6 focused notebooks** for easier learning and reference.

## 📚 Notebook Structure

The original massive cheatsheet (3200+ lines) has been split into logical topic-focused notebooks:

### [1_Risk_Supervised_Learning.ipynb](1_Risk_Supervised_Learning.ipynb)
**Fundamentals of supervised learning and model evaluation**
- Loss functions (MSE, MAE, Log Loss, Zero-One)
- Average risk calculation
- Train/Test/Validation split strategies
- Grid search for hyperparameter tuning
- K-fold cross-validation
- Stratified cross-validation

**Best for:** Understanding how to properly evaluate and tune machine learning models

---

### [2_Estimation_Bias_Variance.ipynb](2_Estimation_Bias_Variance.ipynb)
**Statistical estimation theory and the bias-variance tradeoff**
- Monte Carlo estimation of bias and variance
- Bootstrap resampling methods
- Bootstrap confidence intervals
- Mean and standard error calculations
- Confidence intervals using t-distribution

**Best for:** Understanding statistical uncertainty and the fundamental bias-variance tradeoff

---

### [3_Sampling_RNG.ipynb](3_Sampling_RNG.ipynb)
**Random number generation and sampling techniques**
- Basic random number generators (RNG)
- Linear Congruential Generator (LCG)
- Inversion sampling method
- Rejection sampling algorithm
- Importance sampling for variance reduction
- Monte Carlo integration
- Hoeffding's inequality and confidence bounds

**Best for:** Learning how to generate random samples from complex distributions

---

### [4_Markov_Chains.ipynb](4_Markov_Chains.ipynb) ⭐ *Largest section*
**Complete theory and practice of Markov chains**
- Transition matrices (validation, estimation, simulation)
- N-step transition probabilities
- First passage time and k-th passage time
- Irreducibility and communicating classes
- Transient and recurrent states
- Absorbing states and absorbing chains
- Periodicity and aperiodicity
- **Stationary distributions** (computation and properties)
- Reversibility and detailed balance
- Mean return times
- Canonical form and time-reversed chains
- Expected hitting times
- **Convergence theory** (mixing time, spectral gap)
- **Theoretical foundations:**
  - Chapman-Kolmogorov equation
  - Limiting behavior theorem
  - Ergodic theorem

**Best for:** Deep dive into stochastic processes and Markov chain theory

---

### [5_Pattern_Recognition.ipynb](5_Pattern_Recognition.ipynb)
**Classification metrics and binary classification**
- Confusion matrix (TP, TN, FP, FN)
- Precision, recall, and accuracy
- F1 score and F-beta score
- ROC curves and AUC
- Threshold selection for classification
- Cost-sensitive decision making
- Logistic regression from scratch

**Best for:** Understanding and implementing classification evaluation metrics

---

### [6_Dimensionality_Reduction.ipynb](6_Dimensionality_Reduction.ipynb)
**High-dimensional data analysis and PCA**
- Distance concentration phenomenon (curse of dimensionality)
- Principal Component Analysis (PCA) using SVD
- Choosing number of components (variance explained)
- Scree plots and the elbow method
- Reconstruction error for anomaly detection
- Feature standardization for PCA
- Whitening transformation

**Best for:** Reducing dimensionality and visualizing high-dimensional data

---

## 🎯 Suggested Learning Paths

### For Beginners (Machine Learning Fundamentals)
1. Start with **Notebook 1** (Risk & Supervised Learning)
2. Then **Notebook 5** (Pattern Recognition)
3. Then **Notebook 6** (Dimensionality Reduction)

### For Statistics Focus
1. **Notebook 2** (Estimation & Bias-Variance)
2. **Notebook 3** (Sampling & RNG)
3. **Notebook 4** (Markov Chains)

### For Applied Data Scientists
- **Notebooks 1, 5, 6** cover the most immediately practical topics
- **Notebook 4** (Markov Chains) is valuable for understanding sequential/temporal data

### For Theory-Minded Students
- Work through **all notebooks in order** (1→6)
- **Notebook 4** (Markov Chains) contains the deepest theoretical content

---

## 💡 How to Use These Notebooks

Each notebook is **self-contained** with:
- ✅ Clear explanations of concepts
- ✅ Mathematical formulas (rendered with LaTeX)
- ✅ Complete, runnable Python code
- ✅ Function implementations you can use directly
- ✅ "What is this?" and "When to use" sections

**To get started:**
```bash
# Open in Jupyter
jupyter notebook

# Or open in VS Code with Jupyter extension
code .
```

Then simply open any notebook and run the cells!

---

## 📦 Dependencies

All notebooks use standard scientific Python libraries:
- `numpy` - Numerical computing
- `scipy` - Scientific computing (for stats, optimization)
- `sklearn` (scikit-learn) - Machine learning utilities

Install with:
```bash
pip install numpy scipy scikit-learn
```

---

## 📖 Original Cheatsheet

The complete original cheatsheet is still available as [cheatsheet.ipynb](cheatsheet.ipynb) if you prefer a single comprehensive reference.

---

## 🤝 Contributing

This is a learning resource! If you find errors or have improvements:
1. The content is organized for clarity and ease of learning
2. Each function includes docstrings and explanations
3. All mathematical notation is clearly defined

---

## 📝 Notes

- All code is **from-scratch implementations** for educational purposes
- Functions are designed to be **readable and understandable**, not necessarily optimized
- Mathematical formulas use LaTeX notation for clarity
- Examples and use cases are provided throughout

**Happy Learning! 🚀**
