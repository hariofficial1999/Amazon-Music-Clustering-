
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(page_title="Amazon Music DBSCAN Universe", page_icon="🌌", layout="wide")

# Custom Premium Glossmorphism Theme
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle, #ffffff 0%, #f0f2f6 100%);
    }
    .main-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 30px;
    }
    /* different & Premium KPI Style */
    .kpi-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    .kpi-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        background: white;
    }
    .kpi-val { 
        font-size: 32px; 
        font-weight: 900; 
        background: linear-gradient(45deg, #1e1b4b, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .kpi-lab { 
        font-size: 11px; 
        color: #64748b; 
        text-transform: uppercase; 
        letter-spacing: 2px; 
        font-weight: 800;
        margin-bottom: 8px;
    }
    /* Accent Line for each card */
    .kpi-line {
        height: 4px;
        width: 40px;
        background: #1DB954;
        margin: 10px auto 0;
        border-radius: 10px;
    }
    h1, h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 800; color: #1e1b4b; }
</style>
""", unsafe_allow_html=True)

# Function to load data without caching for real-time updates
def load_data():
    df = pd.read_csv('amazon_music_final_clusters.csv')
    return df

df = load_data()

# --- Fix for PCA Missing Error ---
if 'PCA1' not in df.columns:
    features_for_pca = ['danceability', 'energy', 'speechiness', 'acousticness', 
                        'instrumentalness', 'liveness', 'valence', 'tempo']
    existing_features = [f for f in features_for_pca if f in df.columns]
    if existing_features:
        scaler = StandardScaler()
        X_pca_scaled = scaler.fit_transform(df[existing_features].fillna(0))
        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(X_pca_scaled)
        df['PCA1'] = pca_results[:, 0]
        df['PCA2'] = pca_results[:, 1]
        df['PCA3'] = pca_results[:, 2]

# Calculate numbers
# Prioritize db_cluster for DBSCAN Dashboard
if 'db_cluster' in df.columns:
    df['Cluster'] = df['db_cluster']
    n_clusters = len([c for c in df['Cluster'].unique() if c != -1])
    n_noise = (df['Cluster'] == -1).sum()
elif 'Cluster' in df.columns:
    n_clusters = len([c for c in df['Cluster'].unique() if c != -1])
    n_noise = (df['Cluster'] == -1).sum()
elif 'kmeans_cluster' in df.columns:
    df['Cluster'] = df['kmeans_cluster']
    n_clusters = len(df['Cluster'].unique())
    n_noise = 0
else:
    n_clusters = 0
    n_noise = 0

# Header with Amazon Logo
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)
with col_title:
    st.title("🌌 Amazon Music: DBSCAN Clustering Discovery")
st.markdown("Advanced Density-Based Spatial Clustering for Noise Detection & Pattern Discovery")

# KPI Row
k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi-box"><div class="kpi-lab">Total Tracks</div><div class="kpi-val">{len(df):,}</div><div class="kpi-line"></div></div>', unsafe_allow_html=True)
with k2: st.markdown('<div class="kpi-box"><div class="kpi-lab">Dimensions</div><div class="kpi-val">11 Features</div><div class="kpi-line"></div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-box"><div class="kpi-lab">Algorithm</div><div class="kpi-val">DBSCAN</div><div class="kpi-line"></div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi-box"><div class="kpi-lab">Core Clusters</div><div class="kpi-val">{n_clusters}</div><div class="kpi-line"></div></div>', unsafe_allow_html=True)

st.write("")

# 1. 3D Spatial Universe
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🌠 1. 3D Dynamic Density Map (PCA Space)")
st.caption("A premium visualization of high-dimensional musical data compressed into 3D space.")

# Get raw counts for display
counts = df['Cluster'].value_counts()
c0_count = counts.get(0, 0)
c1_count = counts.get(1, 0)
noise_count = counts.get(-1, 0)

# Display raw distribution as a sleek badge row
st.markdown(f"""
<div style="display: flex; gap: 15px; margin-bottom: 20px;">
    <div style="background: #fee2e2; padding: 5px 15px; border-radius: 10px; border: 1px solid #fecaca; font-size: 13px; color: #ff0000;">
        🌪️ <b>Noise:</b> {noise_count:,}
    </div>
    <div style="background: #dbeafe; padding: 5px 15px; border-radius: 10px; border: 1px solid #bfdbfe; font-size: 13px; color: #1e40af;">
        ✨ <b>Cluster 0:</b> {c0_count:,}
    </div>
    <div style="background: #fdf2f8; padding: 5px 15px; border-radius: 10px; border: 1px solid #fce7f3; font-size: 13px; color: #db2777;">
        ✨ <b>Cluster 1:</b> {c1_count:,}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Fix for DBSCAN Labeling and Visibility ---

# Mapping clusters to specific names with high-contrast labels
def map_cluster_type(x):
    if x == -1: return "🌪️ NOISE (Outliers)"
    return f"🎵 Cluster {int(x)}"

if 'Cluster' in df.columns:
    df['Cluster_Type'] = df['Cluster'].apply(map_cluster_type)
    # Sort for consistent legend
    df = df.sort_values(by='Cluster')
else:
    # Fallback if DBSCAN hasn't been saved to CSV yet
    df['Cluster'] = df['kmeans_cluster'] if 'kmeans_cluster' in df.columns else 0
    df['Cluster_Type'] = df['Cluster'].apply(map_cluster_type)

# High-Contrast Color Palette
color_map = {
    "🎵 Cluster 0": "#1e40af",        # Deep Blue (Massive Cluster)
    "🎵 Cluster 1": "#db2777",        # Neon Pink (Small Cluster)
    "🌪️ NOISE (Outliers)": "#ff0000" # BRIGHT RED (Noise/Outliers)
}

# Intelligent Sampling for Visualization
# We show ALL Noise and ALL Cluster 1 because they are small.
# We sample Cluster 0 because it has 95k+ points.
df_noise = df[df['Cluster'] == -1]
df_cluster1 = df[df['Cluster'] == 1]
df_cluster0 = df[df['Cluster'] == 0].sample(min(5000, len(df[df['Cluster'] == 0]))) if len(df[df['Cluster'] == 0]) > 0 else pd.DataFrame()

# Combine for visualization
df_viz = pd.concat([df_noise, df_cluster1, df_cluster0])

fig_3d = px.scatter_3d(df_viz, 
                      x='PCA1', y='PCA2', z='PCA3', 
                      color='Cluster_Type', 
                      color_discrete_map=color_map,
                      # Making Noise and C1 larger so they are visible
                      size=df_viz['Cluster'].apply(lambda x: 4 if x != 0 else 2),
                      template="plotly_white", 
                      height=800,
                      opacity=0.8,
                      hover_data={'PCA1':False, 'PCA2':False, 'PCA3':False, 'name_song':True} if 'name_song' in df.columns else None)

fig_3d.update_layout(
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(
        title="Legend (Click to filter)",
        orientation="v",
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.7)",
        font=dict(size=12, color="black")
    ),
    scene=dict(
        xaxis=dict(showbackground=False, showticklabels=False, title="PCA Dimension 1"),
        yaxis=dict(showbackground=False, showticklabels=False, title="PCA Dimension 2"),
        zaxis=dict(showbackground=False, showticklabels=False, title="PCA Dimension 3"),
    )
)

# Removing white outlines for solid, vibrant colors
fig_3d.update_traces(marker=dict(line=dict(width=0)))

st.plotly_chart(fig_3d, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 2. FEATURE ANALYSIS
st.markdown('<div class="main-card">', unsafe_allow_html=True)
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo']
cluster_means = df.groupby('Cluster_Type')[features].mean().reset_index()

col_mid1, col_mid2 = st.columns(2)

# 2. Bar Profiles
with col_mid1:
    st.subheader("📊 2. Feature Distribution by Cluster")
    df_melt = cluster_means.melt(id_vars='Cluster_Type')
    fig_bar = px.bar(df_melt, x='variable', y='value', color='Cluster_Type', barmode='group',
                    template="plotly_white", color_discrete_map=color_map)
    st.plotly_chart(fig_bar, use_container_width=True)

# 3. Heatmap
with col_mid2:
    st.subheader("🔥 3. Density Intensity Heatmap")
    hm_data = df.groupby('Cluster_Type')[features].mean()
    fig_hm = px.imshow(hm_data, text_auto=True, color_continuous_scale='Turbo', template="plotly_white")
    st.plotly_chart(fig_hm, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 4. Interactive Feature Deep-Dive
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("📈 4. Musical Feature Variance (Distribution)")
feat_sel = st.selectbox("Compare one specific audio tag:", features, index=1)
fig_vio = px.violin(df.sample(min(5000, len(df))), x="Cluster_Type", y=feat_sel, color="Cluster_Type",
                   box=True, points="all", template="plotly_white",
                   color_discrete_map=color_map)
st.plotly_chart(fig_vio, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 5. Audio Signature Distribution (Modern Histograms)
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("🎵 5. Audio Signature Density")
st.caption("How specific musical traits are distributed across the universe.")

dist_cols = [f for f in features if f in df.columns]
col_d1, col_d2 = st.columns(2)

with col_d1:
    feat_d1 = st.selectbox("Primary Distribution:", dist_cols, index=0, key='d1')
    fig_hist1 = px.histogram(df_viz, x=feat_d1, color="Cluster_Type", marginal="box", 
                             barmode="overlay", template="plotly_white", 
                             color_discrete_map=color_map, opacity=0.7)
    st.plotly_chart(fig_hist1, use_container_width=True)

with col_d2:
    feat_d2 = st.selectbox("Secondary Distribution:", dist_cols, index=min(6, len(dist_cols)-1), key='d2')
    fig_hist2 = px.histogram(df_viz, x=feat_d2, color="Cluster_Type", marginal="box", 
                             barmode="overlay", template="plotly_white", 
                             color_discrete_map=color_map, opacity=0.7)
    st.plotly_chart(fig_hist2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Outlier Detail
st.subheader(f"🌪️ Detected Noise Points (Total: {n_noise:,})")
st.write("These songs are musically 'unique' and did not fit into dense clusters.")
cols_to_show = [c for c in ['name_song', 'name_artists', 'energy', 'tempo'] if c in df.columns]
st.dataframe(df[df['Cluster'] == -1][cols_to_show].head(50), use_container_width=True)

# Data Explorer
st.subheader("📋 Core Track Registry")
cols_to_show_core = [c for c in ['name_song', 'name_artists', 'Cluster', 'energy', 'tempo'] if c in df.columns]
st.dataframe(df[df['Cluster'] != -1][cols_to_show_core].head(100), use_container_width=True)
