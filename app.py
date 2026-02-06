import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- PREMIUM DASHBOARD CONFIG ---
st.set_page_config(
    page_title="Amazon Music clustering | Premium Analysis",
    page_icon="🎵",
    layout="wide"
)

# Custom Styling for Premium LOOK and FEEL
st.markdown("""
<style>
    .main { background-color: #f7f9fc; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1e3799; }
    .stMetric { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-left: 6px solid #4a69bd; }
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 15px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 10px; padding: 12px 30px; font-weight: 600; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
    .stTabs [aria-selected="true"] { color: #4a69bd !important; border-bottom: 2px solid #4a69bd !important; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND CLUSTERING ENGINE ---
@st.cache_data(show_spinner=False)
def load_source_data():
    return pd.read_csv('single_genre_artists.csv')

@st.cache_resource(show_spinner=False)
def run_premium_engine(df):
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                'instrumentalness', 'liveness', 'valence', 'tempo']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run optimal K=5 based on latest analysis
    k_opt = 5
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Updated mapping for 5 clusters
    cluster_mapping = {
        0: 'Chill Acoustic', 
        1: 'Party Tracks', 
        2: 'Vocal / Live', 
        3: 'Instrumental',
        4: 'Energetic Mix'
    }
    df['cluster_name'] = df['cluster'].map(cluster_mapping)
    
    # Spatial Calculation (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]
    
    # Calculate Core Silhouette score for Dashboard (5k sample)
    sample_idx = np.random.choice(X_scaled.shape[0], 5000, replace=False)
    sil_final = silhouette_score(X_scaled[sample_idx], df['cluster'].iloc[sample_idx])
    
    return df, features, X_scaled, sil_final

# --- APP EXECUTION ---
try:
    with st.spinner("🚀 Booting Premium Discovery Interface..."):
        df_source = load_source_data()
        df, features, X_scaled, silhouette_score_k4 = run_premium_engine(df_source)

    # PAGE HEADER
    st.title("🎵 Amazon Music Clustered Analysis")
    st.markdown("Elite clustering analytics identifying distinct musical patterns across 96,000 tracks.")
    st.write("---")

    # PREMIUM METRICS CARDS
    met_1, met_2, met_3, met_4 = st.columns(4)
    met_1.metric("Total Sample Size", f"{len(df):,}")
    met_2.metric("Silhouette Score (k=5)", f"{silhouette_score_k4:.3f}")
    met_3.metric("Algorithm", "K-Means++")
    met_4.metric("Musical Categories", "5")

    # SIDEBAR CONTROLS
    st.sidebar.header("🔬 Deep Study Controls")
    
    # Loop Analysis requested by user: K=2 to 10 comparison
    if st.sidebar.button("📊 Run Silhouette Study (K=2-10)"):
        with st.sidebar:
            st.info("🔄 Evaluating scores for K=2 to 10...")
            k_range = range(2, 11)
            study_scores = []
            for k in k_range:
                # Optimized loop
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
                study_scores.append(silhouette_score(X_scaled, labels, sample_size=1000))
            
            fig_study, ax_study = plt.subplots(figsize=(5, 4))
            ax_study.plot(k_range, study_scores, 'bo-', color="#e67e22", linewidth=2.5)
            ax_study.set_title("Optimal K Validation Chart")
            ax_study.set_xlabel("Clusters (K)")
            ax_study.set_ylabel("Consistency Score")
            st.pyplot(fig_study)
            st.success("Analysis Complete!")

    filter_cat = st.sidebar.selectbox("Filter Dashboard", 
                                     options=['All Categories', 'Chill Acoustic', 'Party Tracks', 'Vocal / Live', 'Instrumental', 'Energetic Mix'])
    
    display_df = df if filter_cat == 'All Categories' else df[df['cluster_name'] == filter_cat]

    # MAIN CONTENT SECTION (Tabs)
    st.write("---")
    tab_discovery, tab_profile, tab_songs = st.tabs(["🚀 Discovery Map", "📈 Cluster Profiling", "🎼 Song Inventory"])

    with tab_discovery:
        st.subheader("Global Music Fingerprint (2D PCA Projection)")
        # Performance sampling for high-density visual responsiveness (25k points)
        render_df = df.sample(min(25000, len(df)), random_state=42)
        fig_map, ax_map = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=render_df, x='pca_1', y='pca_2', hue='cluster_name', alpha=0.5, palette='Set1', s=16, ax=ax_map, edgecolor=None)
        plt.title(f"Clustering Distribution ({len(render_df):,} Points Rendered)")
        st.pyplot(fig_map)

    with tab_profile:
        st.subheader("Cluster Profiling (Mean Feature Values)")
        means_df = df.groupby('cluster_name')[features].mean().T
        fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
        sns.heatmap(means_df, annot=True, cmap='RdYlBu_r', ax=ax_heat, fmt=".2f", center=0, linewidths=.5)
        st.pyplot(fig_heat)
        
        # Feature Distribution Selector (Sidebar)
        f_box = st.sidebar.selectbox("Boxplot Comparison Feature", features, index=0)
        st.subheader(f"Detailed Variance: {f_box.capitalize()}")
        fig_box, ax_box = plt.subplots(figsize=(10, 4))
        sns.boxplot(x='cluster_name', y=f_box, data=df, palette='Pastel1', ax=ax_box)
        st.pyplot(fig_box)

    with tab_songs:
        st.subheader(f"Top Representative Tracks: {filter_cat}")
        # Show top tracks ranked by popularity
        st.dataframe(display_df.sort_values(by='popularity_songs', ascending=False)[['name_song', 'name_artists', 'popularity_songs', 'danceability', 'energy', 'tempo']].head(100), 
                     use_container_width=True)

except Exception as e:
    st.error(f"Execution Error: {e}")
