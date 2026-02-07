# 🌌 Amazon Music Clustering: 3D DBSCAN Discovery

A high-performance machine learning ecosystem designed to analyze, categorize, and visualize the acoustic fingerprints of **95,000+ tracks** using density-based spatial clustering.

---

## 🚀 Project Overview
This project leverages **Advanced Unsupervised Learning** to move beyond metadata (like genres) and understand music through its mathematical properties. By utilizing the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm, we identify natural musical "planets" while detecting unique, outlier tracks that don't fit any standard trend.

---

## 🧬 The "Acoustic DNA" (11 Feature Matrix)
We characterize every track using an expanded **11-dimensional feature set**:

*   **⚡ Energy & 🔊 Loudness**: The intensity, power, and activity of the track.
*   **🕺 Danceability**: Suitability for dancing based on rhythm and beat stability.
*   **🗣️ Speechiness**: Presence of spoken words vs. vocals.
*   **🌿 Acousticness**: Confidence measure of organic/non-electronic sounds.
*   **🎹 Instrumentalness**: High values predict a lack of vocal content.
*   **🏟️ Liveness**: Detection of audience or live recording environment.
*   **🌈 Valence**: Musical positivity (Happiness/Sadness/Mood).
*   **🥁 Tempo**: Overall estimated Beats Per Minute (BPM).
*   **⏱️ Duration**: Length of the track in milliseconds.
*   **🎵 Key/Popularity**: Harmonic foundation and social reach of the tracks.

---

## 📊 Technical Architecture

### 1. Preprocessing & Dimensionality Reduction
*   **Standardization**: All 11 features are scaled using `StandardScaler` to ensure zero mean and unit variance for accurate spatial clustering.
*   **PCA 3D Projection**: We use **Principal Component Analysis** to reduce our 11 dimensions into 3 principal components, enabling a immersive **3D Spatial Universe** visualization.

### 2. Algorithmic Choice: Why DBSCAN?
Unlike traditional K-Means where you must "guess" the number of groups, our implementation uses **DBSCAN** because:
*   **Noise Detection**: It identifies "Outliers" (Cluster -1)—tracks that are musically unique and don't belong to any dense group.
*   **Natural Shapes**: It can discover clusters of arbitrary shapes, better capturing the fluid nature of music.
*   **Tuned Precision**: We have tuned the algorithm (`eps` and `min_samples`) to discover **Targeted 3 Core Clusters**.

---

## 💻 Premium Streamlit Dashboard
The project includes a state-of-the-art `app.py` dashboard with **Glassmorphism Design**:

*   **🌌 3D Spatial Universe**: An interactive 3D map where you can rotate and zoom through the 95k entries.
*   **📊 Dynamic KPI Cards**: Real-time metrics on track count, dimensions, and cluster density with sleek hover animations.
*   **🔥 Density Heatmaps**: Color-coded intensity matrices showing the "Feature Signature" of each cluster.
*   **📈 Distribution Deep-Dives**: Interactive violin plots to compare variance across musical characteristics.

### 🏠 Branded Aesthetics
The app features:
*   **Amazon Branding**: Official logo integration.
*   **Modern Theme**: Light-mode radial gradients with translucent blurred cards.

---

## 📂 Repository Structure
```text
├── Data.ipynb                     # Initial Data Exploration & Cleaning
├── prepare_data.py                # 11-Feature DBSCAN Clustering & PCA Logic
├── app.py                         # Premium Streamlit Visualization App
├── amazon_music_final_clusters.csv # Final Clustered Dataset (The Source of Truth)
├── single_genre_artists.csv        # Raw Musical Audio Features
└── README.md                      # Project Documentation
```

---

## 🛠️ How to Run
1. **Initialize Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install pandas numpy scikit-learn streamlit plotly
   ```
2. **Process Data**:
   ```powershell
   python prepare_data.py
   ```
3. **Launch Dashboard**:
   ```powershell
   streamlit run app.py
   ```

---
*Developed by: Amazon Music Project Team 2026*
