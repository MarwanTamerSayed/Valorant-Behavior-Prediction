# 🎯 Valorant Players Behavior Prediction

This project aims to analyze and predict the behavior and performance level (Tier) of Valorant players using advanced data preprocessing, feature engineering, clustering, and classification models.

## 📊 Project Overview

Valorant is a competitive FPS game where individual performance metrics can help infer play styles and skill tiers. This project uses real or simulated player performance data to:

- Cluster players based on behavior & stats
- Predict their skill tier using classification models

## 🧠 Techniques & Pipeline

- **Data Cleaning & Preprocessing**
  - Outlier treatment
- **Feature Engineering**
  - One-hot encoding of roles (e.g., Duelist, Initiator)
- **Scaling & Transformation**
  - `RobustScaler` for robust normalization
  - `PowerTransformer` to address skewed distributions
- **Dimensionality Reduction**
  - `PCA` to visualize and compress high-dimensional data
- **Unsupervised Learning**
  - KMeans Clustering to segment player behaviors
- **Supervised Learning**
  - `RandomForestClassifier` with `RandomizedSearchCV` for tier prediction
- **Model Evaluation**
  - Accuracy, Confusion Matrix, and more

## 📈 Sample Visualizations

- Cluster-wise Player Stats Comparison
- PCA Projection of Clusters

## 🚀 Streamlit App

The final project is deployed as an interactive web app using [Streamlit](https://streamlit.io/).

### 🔗 [Live Demo](https://valorant-behavior-prediction-rcgztpggzt5m8ssat2kwyr.streamlit.app/)

You can:
- Input player stats
- Get a predicted tier

🙋‍♂️ Author
Marwan Tamer
