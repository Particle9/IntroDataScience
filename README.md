# Introduction to Data Science - Complete Learning Repository

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](LICENSE)

A comprehensive collection of Data Science materials including cheat sheets, lecture notes, assignments, and exam preparations. This repository covers fundamental topics from probability theory and statistical learning to machine learning algorithms and high-dimensional data analysis.

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Organized Cheat Sheets](#-organized-cheat-sheets)
- [Getting Started](#-getting-started)
- [Learning Paths](#-learning-paths)
- [Topics Covered](#-topics-covered)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)

## 🎯 Overview

This repository serves as a complete learning resource for Introduction to Data Science, containing:

- **6 Organized Topic-Focused Cheat Sheets** - Modular notebooks covering core data science concepts
- **Lecture Notes** - Comprehensive chapter-by-chapter toolkit notebooks
- **Assignments** - 4 practical assignments with solutions
- **Past Exams** - Previous exam papers (2022-2025) for practice
- **Utility Functions** - Reusable Python functions for common data science tasks
- **All from-scratch implementations** - Educational code for deep understanding

## 📁 Repository Structure

```
├── cheatsheet/                    # 📚 6 Organized Topic-Focused Notebooks
│   ├── 1_Risk_Supervised_Learning.ipynb
│   ├── 2_Estimation_Bias_Variance.ipynb
│   ├── 3_Sampling_RNG.ipynb
│   ├── 4_Markov_Chains.ipynb
│   ├── 5_Pattern_Recognition.ipynb
│   └── 6_Dimensionality_Reduction.ipynb
│
├── Lecture Notes/                 # 🎓 Chapter-by-Chapter Toolkits
│   ├── Chapter4_Risk_Toolkit.ipynb
│   ├── Chapter5_6_Estimation_RV_Generation_Toolkit.ipynb
│   ├── Chapter7_Finite_Markov_Chains_Toolkit.ipynb
│   ├── Chapter8_Pattern_Recognition_VC_Toolkit.ipynb
│   ├── Chapter10_High_Dimension_Toolkit.ipynb
│   ├── Chapter11_Dimensionality_Reduction_Toolkit.ipynb
│   └── original/                  # Original lecture notebooks
│
├── Assignments/                   # 📝 Course Assignments
│   ├── Assignment_1B.ipynb
│   ├── Assignment_2.ipynb
│   ├── Assignment_3.ipynb
│   └── Assignment_4.ipynb
│
├── Prev Exam/                     # 🎯 Past Examination Papers
│   ├── ExamJanuary_2022_problem.ipynb
│   ├── ExamJanuary_2023_problem.ipynb
│   ├── ExamJanuary_2024_problem.ipynb
│   ├── Exam_ang_tent25.ipynb
│   ├── data/                      # Exam datasets
│   └── Utils.py                   # Helper functions for exams
│
├── Utils.py                       # 🛠️ General utility functions
├── cheatsheet.ipynb              # 📖 Original comprehensive cheatsheet
├── sklearn_cheatsheet.ipynb      # 🔧 Scikit-learn quick reference
└── README.md                      # 📘 This file
```

## 📚 Organized Cheat Sheets

The original massive cheatsheet (3200+ lines) has been reorganized into **6 focused, modular notebooks** for easier learning and reference:

### 🎲 [1. Risk & Supervised Learning](cheatsheet/1_Risk_Supervised_Learning.ipynb)

**Fundamentals of supervised learning and model evaluation**

**Topics Covered:**
- Loss functions (MSE, MAE, Log Loss, Zero-One)
- Average risk calculation
- Train/Test/Validation split strategies
- Grid search for hyperparameter tuning
- K-fold cross-validation
- Stratified cross-validation

**Best for:** Understanding how to properly evaluate and tune machine learning models

---

### 📊 [2. Estimation & Bias-Variance](cheatsheet/2_Estimation_Bias_Variance.ipynb)

**Statistical estimation theory and the bias-variance tradeoff**

**Topics Covered:**
- Monte Carlo estimation of bias and variance
- Bootstrap resampling methods
- Bootstrap confidence intervals
- Mean and standard error calculations
- Confidence intervals using t-distribution

**Best for:** Understanding statistical uncertainty and the fundamental bias-variance tradeoff

---

### 🎰 [3. Sampling & Random Number Generation](cheatsheet/3_Sampling_RNG.ipynb)

**Random number generation and sampling techniques**

**Topics Covered:**
- Basic random number generators (RNG)
- Linear Congruential Generator (LCG)
- Inversion sampling method
- Rejection sampling algorithm
- Importance sampling for variance reduction
- Monte Carlo integration
- Hoeffding's inequality and confidence bounds

**Best for:** Learning how to generate random samples from complex distributions

---

### ⛓️ [4. Markov Chains](cheatsheet/4_Markov_Chains.ipynb) ⭐ *Most Comprehensive*

**Complete theory and practice of Markov chains**

**Topics Covered:**
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

### 🎯 [5. Pattern Recognition](cheatsheet/5_Pattern_Recognition.ipynb)

**Classification metrics and binary classification**

**Topics Covered:**
- Confusion matrix (TP, TN, FP, FN)
- Precision, recall, and accuracy
- F1 score and F-beta score
- ROC curves and AUC
- Threshold selection for classification
- Cost-sensitive decision making
- Logistic regression from scratch

**Best for:** Understanding and implementing classification evaluation metrics

---

### 🔍 [6. Dimensionality Reduction](cheatsheet/6_Dimensionality_Reduction.ipynb)

**High-dimensional data analysis and PCA**

**Topics Covered:**
- Distance concentration phenomenon (curse of dimensionality)
- Principal Component Analysis (PCA) using SVD
- Choosing number of components (variance explained)
- Scree plots and the elbow method
- Reconstruction error for anomaly detection
- Feature standardization for PCA
- Whitening transformation

**Best for:** Reducing dimensionality and visualizing high-dimensional data

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (or VS Code with Jupyter extension)
- Basic knowledge of Python and linear algebra

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Intro To Data Science Cheat Sheet"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or install manually:
   pip install numpy scipy scikit-learn matplotlib pandas jupyter
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   # Or use JupyterLab:
   jupyter lab
   ```

4. **Start learning:**
   - Navigate to `cheatsheet/` folder
   - Open any notebook (start with `1_Risk_Supervised_Learning.ipynb`)
   - Run cells and explore!

---

## 🎯 Learning Paths

Choose your learning path based on your goals and background:

### 🌟 Path 1: Machine Learning Fundamentals (Recommended for Beginners)
**Duration: ~2-3 weeks**

1. [Risk & Supervised Learning](cheatsheet/1_Risk_Supervised_Learning.ipynb) - Learn model evaluation
2. [Pattern Recognition](cheatsheet/5_Pattern_Recognition.ipynb) - Master classification metrics
3. [Dimensionality Reduction](cheatsheet/6_Dimensionality_Reduction.ipynb) - Handle high-dimensional data

**Why this path?** Covers the most practical, immediately applicable machine learning concepts.

### 📈 Path 2: Statistical Foundations
**Duration: ~3-4 weeks**

1. [Estimation & Bias-Variance](cheatsheet/2_Estimation_Bias_Variance.ipynb) - Statistical inference
2. [Sampling & RNG](cheatsheet/3_Sampling_RNG.ipynb) - Random sampling techniques
3. [Markov Chains](cheatsheet/4_Markov_Chains.ipynb) - Stochastic processes

**Why this path?** Builds deep understanding of probability and statistics underlying ML.

### 🚀 Path 3: Complete Data Scientist Track
**Duration: ~6-8 weeks**

Work through **all notebooks in order** (1→6), supplemented with:
- Lecture notes for theoretical depth
- Assignments for hands-on practice
- Past exams for assessment preparation

**Why this path?** Comprehensive coverage from theory to practice.

### 🎓 Path 4: Exam Preparation
**Duration: ~2-3 weeks (intensive)**

1. Review all **6 cheat sheet notebooks** quickly
2. Work through **past exams** in `Prev Exam/` folder
3. Complete **assignments** for practice
4. Use `sklearn_cheatsheet.ipynb` for quick reference

**Why this path?** Optimized for exam success with focused practice.

---

## 📖 Topics Covered

### Core Machine Learning
- Supervised learning fundamentals
- Loss functions and risk minimization
- Model evaluation and validation
- Hyperparameter tuning
- Cross-validation techniques
- Classification metrics (precision, recall, F1, ROC-AUC)
- Logistic regression

### Statistical Methods
- Monte Carlo estimation
- Bootstrap methods
- Bias-variance tradeoff
- Confidence intervals
- Statistical inference
- Sampling techniques
- Random number generation

### Advanced Topics
- Markov chains and stochastic processes
- Stationary distributions
- Convergence analysis
- Principal Component Analysis (PCA)
- Dimensionality reduction
- Curse of dimensionality
- Feature engineering

### Computational Techniques
- Linear Congruential Generators
- Inversion sampling
- Rejection sampling
- Importance sampling
- Monte Carlo integration
- Numerical optimization

---

## 📦 Installation

### Option 1: Using pip (Recommended)

```bash
# Install core dependencies
pip install numpy scipy scikit-learn matplotlib pandas jupyter

# Optional: Install additional visualization libraries
pip install seaborn plotly
```

### Option 2: Using conda

```bash
# Create a new environment
conda create -n datascience python=3.9

# Activate the environment
conda activate datascience

# Install packages
conda install numpy scipy scikit-learn matplotlib pandas jupyter notebook
```

### Option 3: Using requirements.txt

Create a `requirements.txt` file:
```txt
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
pandas>=1.3.0
jupyter>=1.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Using Jupyter Notebooks

```bash
# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab (modern interface)
jupyter lab
```

Navigate to any notebook and start exploring!

### Using VS Code

1. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
2. Open any `.ipynb` file
3. Select Python kernel
4. Run cells with Shift+Enter

### Running Code Snippets

All functions are self-contained and can be imported:

```python
# Import from Utils.py
from Utils import *

# Or copy functions directly from notebooks
# All implementations are educational and well-documented
```

---

## 📚 Additional Resources

### Included in This Repository

- **Original Cheatsheet**: [cheatsheet.ipynb](cheatsheet.ipynb) - Single comprehensive reference (3200+ lines)
- **Scikit-learn Cheatsheet**: [sklearn_cheatsheet.ipynb](sklearn_cheatsheet.ipynb) - Quick sklearn reference
- **Utils Module**: [Utils.py](Utils.py) - Reusable utility functions
- **Lecture Notes**: Detailed chapter-by-chapter breakdowns with theory
- **All of Statistics**: Additional statistical theory materials

### Learning Features

Each notebook includes:
- ✅ **Clear explanations** - "What is this?" and "When to use" sections
- ✅ **Mathematical formulas** - LaTeX-rendered equations with variable definitions
- ✅ **Runnable code** - Complete, executable Python implementations
- ✅ **Docstrings** - Detailed function documentation
- ✅ **Examples** - Practical use cases and demonstrations
- ✅ **From-scratch implementations** - No black boxes, understand everything

---