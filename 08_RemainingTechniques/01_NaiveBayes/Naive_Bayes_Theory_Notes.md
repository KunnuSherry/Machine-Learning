
# 📘 Naive Bayes Classifier – Theory Notes

## 🔍 What is Naive Bayes?
Naive Bayes is a **supervised learning algorithm** based on **Bayes' Theorem** with a strong (naive) assumption of **independence among features**.

It is mostly used for:
- **Text classification**
- **Spam detection**
- **Sentiment analysis**

## 📐 Bayes' Theorem

P(A|B) = P(B|A) * P(A) / P(B)

Where:
- P(A|B): Posterior probability (what we want)
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Evidence

## 🧠 Intuition Behind Naive Bayes

For a class C_k and feature vector x = (x1, x2, ..., xn):

P(C_k|x) ∝ P(C_k) * P(x1|C_k) * P(x2|C_k) * ... * P(xn|C_k)

This assumes:
- All features xi are **independent** given the class label C_k
- Classification is done by **maximum posterior probability**

## 🧮 Steps in Naive Bayes Classification

1. **Calculate Prior Probabilities** P(C_k)
2. **Calculate Likelihoods** P(xi | C_k)
3. **Apply Bayes' Theorem** to compute posterior P(C_k | x)
4. **Predict the Class** with the highest posterior probability

## 🔠 Types of Naive Bayes

| Type | Feature Assumption | Example Use |
|------|--------------------|-------------|
| **Gaussian NB** | Continuous values (assumed normal distribution) | Iris dataset |
| **Multinomial NB** | Discrete counts (like word frequencies) | Text classification |
| **Bernoulli NB** | Binary features (0/1) | Spam detection |

## ✅ Advantages

- Very fast and efficient
- Works well with high-dimensional data
- Requires less training data
- Performs well in NLP tasks

## ❌ Disadvantages

- Strong independence assumption rarely holds
- Poor performance if features are correlated
- Struggles with zero-frequency problems (can be handled with **Laplace Smoothing**)

## 🧂 Laplace Smoothing

Used to avoid zero probability issue:

P(xi|C_k) = (count(xi in C_k) + 1) / (total count of all features in C_k + number of features)

## 🧪 Example

Let’s classify an email as **Spam** or **Ham** based on words:
- Compute prior: P(Spam), P(Ham)
- Compute likelihood for each word
- Multiply them together with the prior
- Class with higher value is the prediction

## 🛠️ Real-life Applications

- Gmail spam filtering
- News classification
- Sentiment analysis on tweets/reviews
- Face recognition (basic cases)

## 📚 Summary

- Based on Bayes’ theorem
- Assumes feature independence
- Simple, fast, effective
- Use **Multinomial NB** for text, **Gaussian NB** for continuous features
