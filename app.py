
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# Calculate numbers
n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0)
n_noise = list(df['Cluster']).count(-1)

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
st.subheader("🌠 1. 3D Density Map (DBSCAN + PCA)")
st.caption("Rotate to see how DBSCAN identifies dense musical regions vs sparse noise (Label -1).")
# Create labels for clusters including noise
df['Cluster_Type'] = df['Cluster'].apply(lambda x: f"Cluster {x}" if x != -1 else "🌪️ Outliers/Noise")

fig_3d = px.scatter_3d(df.sample(min(5000, len(df))), x='PCA1', y='PCA2', z='PCA3', 
                      color='Cluster_Type', template="plotly_white",
                      color_discrete_sequence=px.colors.qualitative.Prism, height=750)
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig_3d, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 2. FEATURE ANALYSIS
st.markdown('<div class="main-card">', unsafe_allow_html=True)
features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'valence']
cluster_means = df.groupby('Cluster')[features].mean().reset_index()

col_mid1, col_mid2 = st.columns(2)

# 2. Bar Profiles
with col_mid1:
    st.subheader("📊 2. Feature Distribution by Cluster")
    df_melt = cluster_means.melt(id_vars='Cluster')
    fig_bar = px.bar(df_melt, x='variable', y='value', color='Cluster', barmode='group',
                    template="plotly_white", color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_bar, use_container_width=True)

# 3. Heatmap
with col_mid2:
    st.subheader("🔥 3. Density Intensity Heatmap")
    hm_data = df.groupby('Cluster')[features].mean()
    fig_hm = px.imshow(hm_data, text_auto=True, color_continuous_scale='Turbo', template="plotly_white")
    st.plotly_chart(fig_hm, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# 4. Interactive Feature Deep-Dive
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("📈 4. Musical Feature Variance (Distribution)")
feat_sel = st.selectbox("Compare one specific audio tag:", features, index=1)
fig_vio = px.violin(df.sample(min(5000, len(df))), x="Cluster_Type", y=feat_sel, color="Cluster_Type",
                   box=True, points="all", template="plotly_white",
                   color_discrete_sequence=px.colors.qualitative.Dark24)
st.plotly_chart(fig_vio, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Outlier Detail
st.subheader(f"🌪️ Detected Noise Points (Total: {n_noise:,})")
st.write("These songs are musically 'unique' and did not fit into dense clusters.")
st.dataframe(df[df['Cluster'] == -1][['name_song', 'name_artists', 'energy', 'tempo']].head(50), use_container_width=True)

# Data Explorer
st.subheader("📋 Core Track Registry")
st.dataframe(df[df['Cluster'] != -1][['name_song', 'name_artists', 'Cluster', 'energy', 'tempo']].head(100), use_container_width=True)
