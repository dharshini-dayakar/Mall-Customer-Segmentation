# =========================================
# IMPORT LIBRARIES
# =========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="📊",
    layout="centered"
)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("About Project")

st.sidebar.info(
    """
    This project uses K-Means Clustering
    to group mall customers based on:

    • Annual Income
    • Spending Score

    Built using Streamlit and Machine Learning.
    """
)

# =========================================
# MAIN TITLE
# =========================================

st.title("📊 Mall Customer Segmentation")

st.write(
    "Customer Segmentation using "
    "K-Means Clustering"
)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("Mall_Customers.csv")

# =========================================
# SHOW DATASET
# =========================================

st.subheader("📁 Dataset")

st.dataframe(df)

# Dataset Shape

st.write("Dataset Shape:", df.shape)

# =========================================
# SELECT FEATURES
# =========================================

X = df.iloc[:, [3,4]].values

# =========================================
# ELBOW METHOD
# =========================================

st.subheader("📉 Elbow Method")

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)

# =========================================
# ELBOW GRAPH
# =========================================

fig1, ax1 = plt.subplots()

ax1.plot(range(1,11), wcss, marker='o')

ax1.set_title("Elbow Method")

ax1.set_xlabel("Number of Clusters")

ax1.set_ylabel("WCSS")

st.pyplot(fig1)

# =========================================
# SELECT NUMBER OF CLUSTERS
# =========================================

st.subheader("⚙ Select Number of Clusters")

clusters = st.slider(
    "Choose Clusters",
    2,
    10,
    5
)

# =========================================
# TRAIN MODEL
# =========================================

kmeans = KMeans(
    n_clusters=clusters,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# =========================================
# CUSTOMER SEGMENTATION GRAPH
# =========================================

st.subheader("📌 Customer Segmentation")

fig2, ax2 = plt.subplots(figsize=(8,6))

scatter = ax2.scatter(
    X[:,0],
    X[:,1],
    c=y_kmeans,
    s=100
)

# CENTROIDS

ax2.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='yellow',
    label='Centroids'
)

ax2.set_title("Customer Groups")

ax2.set_xlabel("Annual Income")

ax2.set_ylabel("Spending Score")

ax2.legend()

st.pyplot(fig2)

# =========================================
# SUCCESS MESSAGE
# =========================================

st.success("✅ Project Completed Successfully")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.write(
    "Developed using Python, Streamlit, "
    "Pandas, Matplotlib and Scikit-learn"
)