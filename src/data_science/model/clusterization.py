# cluster_students.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---- import your preprocessing module ----
from preprocess import (
    preprocess_fit,
    preprocess_apply,
    gpa_to_num,
    attendance_to_num,
    study_to_num,
    sleep_to_num,
    prep_to_num,
)

sns.set(style="whitegrid", context="talk")


# ============================================================
# 1. LOAD RAW DATA & BUILD FEATURE MATRIX USING preprocess.py
# ============================================================
raw_df = pd.read_csv("processed_students_raw.csv")

# Fit preprocessing pipeline (on full data, dedup, etc.)
# We mainly need the fitted pipeline; X_fit is not used for clustering.
X_fit, y, prep_pipeline = preprocess_fit(raw_df)

# Apply the same pipeline to the full dataset (no dropping duplicates here)
X = preprocess_apply(raw_df, prep_pipeline)

print("✔ Feature matrix for clustering built.")
print("Shape of X:", X.shape)


# ============================================================
# 2. CHOOSE OPTIMAL NUMBER OF CLUSTERS WITH SILHOUETTE SCORE
# ============================================================
def choose_best_k(X, k_min=3, k_max=8, random_state=42):
    best_k = None
    best_score = -1
    scores = {}

    print("\nSearching for optimal k using silhouette score...")
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"k = {k}: silhouette = {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest k: {best_k} (silhouette = {best_score:.3f})")
    return best_k, scores


best_k, scores = choose_best_k(X, k_min=3, random_state=42)


# ============================================================
# 3. FIT FINAL KMEANS MODEL WITH BEST k
# ============================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X)

# Attach clusters to a clean copy of raw data for interpretation
df_clusters = raw_df.copy()
df_clusters["cluster"] = cluster_labels

print("\nCluster sizes:")
print(df_clusters["cluster"].value_counts().sort_index())


# ============================================================
# 4. BUILD INTERPRETABLE NUMERIC PROFILE PER STUDENT
#    (GPA, attendance, self-study, sleep, preparation, etc.)
# ============================================================
df_prof = df_clusters.copy()

# standardize col names and lowercase values for mapping functions
df_prof.columns = df_prof.columns.str.lower()
df_prof = df_prof.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# numeric conversions using your helper functions
df_prof["gpa_num"] = df_prof["average_gpa"].apply(gpa_to_num)
df_prof["attend_num"] = df_prof["average_attendance"].apply(attendance_to_num)
df_prof["study_num"] = df_prof["self_study_hours"].apply(study_to_num)
df_prof["sleep_num"] = df_prof["average_sleep_hours"].apply(sleep_to_num)
df_prof["prep_num"] = df_prof["exam_preparation_hours"].apply(prep_to_num)

# ordinal mappings for interpretation (same logic as preprocess.py)
motivation_map = {"very low": 1, "low": 2, "average": 3, "high": 4, "very high": 5}
stress_map = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "very often": 5}
interest_map = {
    "not important": 1,
    "slightly important": 2,
    "moderately important": 3,
    "important": 4,
    "very important": 5,
}
comm_map = {"very low": 1, "low": 2, "average": 3, "high": 4, "very high": 5}
teacher_sat_map = {
    "not satisfied at all": 1,
    "rather dissatisfied": 2,
    "hard to say": 3,
    "rather satisfied": 4,
    "fully satisfied": 5,
}
resources_map = {"never": 1, "rarely": 2, "sometimes": 3, "often": 4, "very often": 5}

def map_safe(series, mapping):
    return series.map(mapping)

df_prof["motivation_num"] = map_safe(df_prof["motivation_level"], motivation_map)
df_prof["stress_num"] = map_safe(df_prof["stress_frequency"], stress_map)
df_prof["interest_num"] = map_safe(df_prof["importance_of_interest_in_subject"], interest_map)
df_prof["comm_num"] = map_safe(df_prof["communication_skills"], comm_map)
df_prof["teacher_sat_num"] = map_safe(df_prof["teacher_competence_satisfaction"], teacher_sat_map)
df_prof["resources_num"] = map_safe(df_prof["use_of_university_resources"], resources_map)

# keep numeric interpretation features + cluster
profile_features = [
    "cluster",
    "gpa_num",
    "attend_num",
    "study_num",
    "sleep_num",
    "prep_num",
    "motivation_num",
    "stress_num",
    "interest_num",
    "comm_num",
    "teacher_sat_num",
    "resources_num",
]

profile_df = df_prof[profile_features].copy()

# compute mean profile per cluster
cluster_profile = profile_df.groupby("cluster").mean().round(2)
print("\n===== CLUSTER PROFILE (MEAN VALUES) =====")
print(cluster_profile)


# ============================================================
# 5. VISUALIZATIONS
# ============================================================

# ---------- 5.1 Cluster sizes (bar plot) ----------
plt.figure(figsize=(8, 5))
sns.countplot(x="cluster", data=df_clusters, palette="viridis")
plt.title("Cluster Sizes (Number of Students per Cluster)")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ---------- 5.2 PCA 2D scatter (beautiful) ----------
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X)

pca_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
pca_df["cluster"] = cluster_labels

explained_2d = pca_2d.explained_variance_ratio_.sum() * 100

plt.figure(figsize=(10, 7))
palette = sns.color_palette("viridis", n_colors=best_k)

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette=palette,
    s=90,
    alpha=0.9,
    edgecolor="black",
)
plt.title(f"Student Clusters (PCA 2D Projection, {explained_2d:.1f}% variance explained)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster", loc="best", frameon=True)
sns.despine()
plt.tight_layout()
plt.show()


# ---------- 5.3 OPTIONAL: 3D PCA scatter ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pca_3d = PCA(n_components=3, random_state=42)
X_3d = pca_3d.fit_transform(X)

pca3_df = pd.DataFrame(X_3d, columns=["PC1", "PC2", "PC3"])
pca3_df["cluster"] = cluster_labels

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for cl in sorted(pca3_df["cluster"].unique()):
    subset = pca3_df[pca3_df["cluster"] == cl]
    ax.scatter(
        subset["PC1"],
        subset["PC2"],
        subset["PC3"],
        s=70,
        label=f"Cluster {cl}",
    )

ax.set_title("Student Clusters (3D PCA Projection)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.show()


# ---------- 5.4 Heatmap of cluster profiles ----------
plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_profile,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Mean value"},
)
plt.title("Cluster Profiles – Mean Values of Key Features")
plt.xlabel("Feature")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

print("\nINTERPRETATION HINTS:")
print("- Higher gpa_num & attend_num: academically stronger clusters.")
print("- Higher motivation_num with higher resources_num: engaged students using university resources.")
print("- Higher stress_num with low sleep_num and high prep_num: potentially overloaded/high-pressure cluster.")
print("- Differences in cluster sizes show how common each 'student type' is in your dataset.")
