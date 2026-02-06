# 🎵 Amazon Music Clustering & Advanced Analytics

A professional machine learning ecosystem designed to analyze, categorize, and visualize the acoustic fingerprints of **96,000+ tracks**. This project moves beyond metadata to understand music through its mathematical properties.

---

## 🚀 Executive Summary
In the modern streaming era, genres are often too broad. This project leverages **Unsupervised Learning** to group tracks based on **9 distinct audio features**, creating 4 distinct "Musical Personalities." By utilizing **K-Means Clustering** validated by a **Silhouette Analysis Loop**, we achieve high-precision track segmentation suitable for recommendation engines.

---

## 🧪 The "Acoustic DNA" (Feature Dictionary)
We characterize every track using these nine core dimensions:

*   **⚡ Energy**: A perceptual measure of intensity and activity (0.0 to 1.0).
*   **🕺 Danceability**: How suitable a track is for dancing based on tempo, rhythm stability, and beat strength.
*   **🔊 Loudness**: The overall loudness of a track in decibels (dB), typical range -60 to 0 dB.
*   **🗣️ Speechiness**: Detects the presence of spoken words. Values above 0.66 describe talk-show like tracks.
*   **🌿 Acousticness**: A confidence measure (0.0 to 1.0) of whether the track is acoustic.
*   **🎹 Instrumentalness**: Predicts whether a track contains no vocals.
*   **🏟️ Liveness**: Detects the presence of an audience in the recording.
*   **🌈 Valence**: A measure from 0.0 to 1.0 describing the musical positiveness (Happiness/Sadness).
*   **🥁 Tempo**: The overall estimated beats per minute (BPM).

---

## 📊 Technical Deep Dive: The Data Science Workflow

### 1. Feature Engineering & Normalization
Since features like **Tempo** (BPM ~120) and **Acousticness** (0 to 1) have vastly different scales, we apply **Standardization**:
$$z = \frac{x - \mu}{\sigma}$$
Using `StandardScaler`, we ensure that the **Euclidean Distance** used in clustering gives equal weight to all features.

### 2. Algorithmic Comparison (The Sandbox)
In `clustering_techniques.ipynb`, we explored multiple strategies:
*   **K-Means++**: Selected as the final model for its stability and clear boundary definition.
*   **DBSCAN**: Tested for noise detection (identifying outliers that don't fit any vibe).
*   **Hierarchical (Ward)**: Used to understand the "taxonomy" of music through Dendrograms.

### 3. Evaluating Model Quality
We don't just "guess" the clusters; we use three primary mathematical validations:
*   **📉 Inertia (WCSS)**: Measures how compact each cluster is. We look for the "Elbow" where adding another cluster provides diminishing returns.
*   **✨ Silhouette Score**: Measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation). **K=4** yielded the highest scores.
*   **📐 Davies-Bouldin Index**: Analyzes the ratio of within-cluster distances to between-cluster distances. Lower values confirm better separation.

### 4. Dimensionality Reduction (PCA Visualization)
To visualize 9D space, we use **Principal Component Analysis**:
*   **PC1**: Often correlates with Energy and Loudness (Intensity).
*   **PC2**: Often correlates with Acousticness vs. Electronic elements.

---

## 🎤 The 4 Musical Personalities (Cluster Profiles)

| Cluster | Personality | Feature Signature | Top Use Case |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | 🌿 **Chill Acoustic** | Low Energy + High Acousticness | Study, Morning coffee, Zen |
| **Cluster 1** | 💃 **Party Tracks** | High Energy + High Valence + High Dance | Clubs, Gym, Driving |
| **Cluster 2** | 🎤 **Vocal / Live** | High Speechiness + High Liveness | Rap, Podcasts, Live Concerts |
| **Cluster 3** | 🎼 **Instrumental** | High Instrumentalness + Low Speech | Lo-Fi, Soundtracks, Focus |

---

## 💻 Interactive Streamlit Dashboard
The project includes a premium `app.py` dashboard for stakeholder presentation:

*   **Dynamic Explorer**: Filter 96k tracks by their categorized vibe.
*   **Validation Tab**: Run the **K-Value Comparison Loop** ($K=2 \rightarrow 10$) in real-time to see mathematical proof of the model.
*   **Acoustic Profiling**: Visual heatmaps comparing audio DNA across all categories.
*   **Performance Optimization**: Uses downsampling (10k-20k points) and `@st.cache` to ensure instant responsiveness despite the massive dataset size.

---

## 📂 Repository Structure
```text
├── Data.ipynb                  # Data Ingestion & Cleaning
├── clustering_techniques.ipynb  # Algorithm Comparison Sandbox
├── silhouette_analysis_k4.ipynb # Numerical Proof for K=4
├── clustering_visualization.ipynb # Advanced Graphical Analysis
├── final_analysis_and_export.ipynb # Final Labeling & CSV Export
├── app.py                      # Main Streamlit Dashboard
└── single_genre_artists.csv     # Raw Dataset (Audio Features)
```

---

## 🔮 Future Scope
*   **Recommendation Engine**: Using Cosine Similarity on these clusters to find the "Next Best Track."
*   **Genre Correlation**: Analyzing how traditional genres (Rock, Pop) map onto these acoustic clusters.
*   **Sentiment Analysis**: Integrating lyrics analysis with audio valence for deeper emotional tagging.

---
*Developed by the Intern Team | Project: Amazon Music Clustering 2026*
