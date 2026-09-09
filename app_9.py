"""
EduPro — Student Segmentation & Personalised Course Recommendation System
Streamlit Web Application

Run with:
    streamlit run app.py

Expects EduPro_Online_Platform.xlsx (sheets: Users, Courses, Transactions)
in the same folder, or upload it from the sidebar.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

st.set_page_config(page_title="EduPro Learner Segmentation", layout="wide")

# ============================================================
# 1. DATA LOADING & PIPELINE  (cached so it only runs once)
# ============================================================

@st.cache_data(show_spinner="Loading data...")
def load_raw_data(file):
    users_df = pd.read_excel(file, sheet_name="Users")
    courses_df = pd.read_excel(file, sheet_name="Courses")
    transactions_df = pd.read_excel(file, sheet_name="Transactions")
    return users_df, courses_df, transactions_df


def most_common(series):
    return series.mode()[0]


@st.cache_data(show_spinner="Building learner profiles...")
def build_learner_profiles(users_df, courses_df, transactions_df):
    merged_df = transactions_df.merge(courses_df, on="CourseID", how="left")
    merged_df = merged_df.merge(users_df, on="UserID", how="left")

    reference_date = merged_df["TransactionDate"].max() + pd.Timedelta(days=1)

    learner_df = merged_df.groupby("UserID").agg(
        TotalCourses=("CourseID", "count"),
        TotalSpend=("Amount", "sum"),
        AvgSpend=("Amount", "mean"),
        AvgCourseRating=("CourseRating", "mean"),
        FavouriteCategory=("CourseCategory", most_common),
        FavouriteLevel=("CourseLevel", most_common),
        FreeCourseRatio=("CourseType", lambda x: (x == "Free").mean()),
        LastPurchaseDate=("TransactionDate", "max"),
    ).reset_index()

    learner_df["Recency"] = (reference_date - learner_df["LastPurchaseDate"]).dt.days
    learner_df = learner_df.drop(columns="LastPurchaseDate")
    learner_df = learner_df.merge(users_df[["UserID", "Age", "Gender"]], on="UserID", how="left")
    learner_df = learner_df.round(2)

    return learner_df, merged_df


@st.cache_data(show_spinner="Preparing features for clustering...")
def prepare_cluster_input(learner_df):
    numeric_features = ["Age", "TotalCourses", "TotalSpend", "AvgSpend",
                         "AvgCourseRating", "FreeCourseRatio", "Recency"]
    categorical_features = ["Gender", "FavouriteCategory", "FavouriteLevel"]

    scaled_numeric = StandardScaler().fit_transform(learner_df[numeric_features])
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_features)
    encoded_categoricals = pd.get_dummies(learner_df[categorical_features], drop_first=True)

    cluster_input_df = pd.concat(
        [scaled_numeric_df.reset_index(drop=True), encoded_categoricals.reset_index(drop=True)], axis=1
    )
    return cluster_input_df


@st.cache_data(show_spinner="Running K-Means clustering...")
def run_kmeans(cluster_input_df, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(cluster_input_df)
    sil_score = silhouette_score(cluster_input_df, labels)
    return labels, sil_score


@st.cache_data(show_spinner="Building course similarity matrix...")
def build_course_similarity(courses_df):
    course_features = pd.get_dummies(courses_df[["CourseCategory", "CourseType", "CourseLevel"]])
    course_features.index = courses_df["CourseID"]
    sim_df = pd.DataFrame(
        cosine_similarity(course_features), index=course_features.index, columns=course_features.index
    )
    return sim_df


@st.cache_data(show_spinner="Scoring course popularity per segment...")
def build_cluster_course_popularity(merged_df, learner_df, courses_df):
    merged_with_cluster = merged_df.merge(learner_df[["UserID", "Cluster"]], on="UserID", how="left")
    counts = (
        merged_with_cluster.groupby(["Cluster", "CourseID"]).size().reset_index(name="TimesBought")
        .merge(courses_df[["CourseID", "CourseRating"]], on="CourseID", how="left")
    )
    counts["PopularityScore"] = counts["TimesBought"] * (counts["CourseRating"] / 5)
    return counts


def recommend_courses(user_id, learner_df, merged_df, course_similarity_df,
                       cluster_course_counts, courses_df, top_n=5,
                       level_filter=None, category_filter=None):
    cluster = learner_df.loc[learner_df.UserID == user_id, "Cluster"].values[0]
    already_bought = merged_df.loc[merged_df.UserID == user_id, "CourseID"].unique()

    popularity = cluster_course_counts.loc[cluster_course_counts.Cluster == cluster].set_index("CourseID")["PopularityScore"]
    similarity = course_similarity_df[already_bought].mean(axis=1) if len(already_bought) else pd.Series(0, index=course_similarity_df.index)

    scores = pd.DataFrame({"Popularity": popularity, "Similarity": similarity}).fillna(0)
    scores = scores / scores.max().replace(0, 1)
    scores["FinalScore"] = 0.5 * scores["Popularity"] + 0.5 * scores["Similarity"]
    scores = scores.drop(index=[c for c in already_bought if c in scores.index], errors="ignore")

    result = scores.merge(courses_df, left_index=True, right_on="CourseID")

    if level_filter and level_filter != "All":
        result = result[result["CourseLevel"] == level_filter]
    if category_filter and category_filter != "All":
        result = result[result["CourseCategory"] == category_filter]

    cols = ["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseType", "CourseRating", "FinalScore"]
    return result.sort_values("FinalScore", ascending=False).head(top_n)[cols]


# ============================================================
# 2. KPI CALCULATIONS  (Evaluation & Validation)
# ============================================================

@st.cache_data(show_spinner="Calculating intra-cluster similarity...")
def compute_intra_cluster_similarity(cluster_input_df, labels):
    """Behavioural consistency: how similar are learners *within* the same
    cluster to each other, on average? 1.0 = identical behaviour, 0 = no
    similarity. Computed with cosine similarity on the same scaled feature
    vectors used for clustering."""
    sim_matrix = cosine_similarity(cluster_input_df)
    per_cluster = {}
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            per_cluster[c] = np.nan
            continue
        sub = sim_matrix[np.ix_(idx, idx)]
        n = len(idx)
        per_cluster[c] = (sub.sum() - n) / (n * (n - 1))  # average off-diagonal similarity
    overall = float(np.nanmean(list(per_cluster.values())))
    return per_cluster, overall


@st.cache_data(show_spinner="Backtesting recommendation precision...")
def compute_recommendation_precision(_learner_df, _merged_df, _course_similarity_df,
                                      _cluster_course_counts, _courses_df, top_n=5, sample_size=150):
    """Proxy for relevance: for each sampled learner, we treat their most
    recently purchased course's CATEGORY as their 'true interest'. We check
    what fraction of their top-N recommended courses share that category.
    This avoids needing separate ground-truth labels (which EduPro doesn't
    have yet) while still testing whether recommendations are on-topic."""
    eligible = _learner_df.loc[_learner_df["TotalCourses"] >= 2, "UserID"]
    sample = eligible.sample(min(sample_size, len(eligible)), random_state=42)

    precisions = []
    for uid in sample:
        user_tx = _merged_df.loc[_merged_df.UserID == uid].sort_values("TransactionDate")
        target_category = user_tx.iloc[-1]["CourseCategory"]

        recs = recommend_courses(uid, _learner_df, _merged_df, _course_similarity_df,
                                  _cluster_course_counts, _courses_df, top_n=top_n)
        if len(recs) == 0:
            continue
        match_rate = (recs["CourseCategory"] == target_category).mean()
        precisions.append(match_rate)

    return float(np.mean(precisions)) if precisions else 0.0


@st.cache_data(show_spinner="Estimating engagement lift...")
def compute_engagement_lift(_learner_df, _merged_df, _course_similarity_df,
                             _cluster_course_counts, _courses_df, top_n=5, sample_size=150):
    """Proxy for business impact: compares the average CourseRating of
    PERSONALISED recommendations against the average CourseRating of the
    whole catalogue (a stand-in for the old 'generic/random' experience).
    A positive % means personalised recommendations surface better-rated
    content than a generic baseline -> a reasonable proxy for expected
    engagement improvement."""
    baseline_rating = _courses_df["CourseRating"].mean()

    eligible = _learner_df["UserID"]
    sample = eligible.sample(min(sample_size, len(eligible)), random_state=42)

    rec_ratings = []
    for uid in sample:
        recs = recommend_courses(uid, _learner_df, _merged_df, _course_similarity_df,
                                  _cluster_course_counts, _courses_df, top_n=top_n)
        if len(recs) > 0:
            rec_ratings.append(recs["CourseRating"].mean())

    personalised_rating = float(np.mean(rec_ratings)) if rec_ratings else baseline_rating
    lift_pct = ((personalised_rating - baseline_rating) / baseline_rating) * 100
    return personalised_rating, baseline_rating, lift_pct
# ============================================================
# 3. SIDEBAR — SETTINGS
# ============================================================

st.sidebar.title("Settings")

DEFAULT_FILE_PATH = "EduProOnlinePlatform.xlsx"

k = st.sidebar.slider("Number of learner segments (k)", min_value=2, max_value=8, value=4)
top_n = st.sidebar.slider("Recommendations to show", min_value=3, max_value=10, value=5)

try:
    users_df, courses_df, transactions_df = load_raw_data(DEFAULT_FILE_PATH)
except FileNotFoundError:
    st.error(f"Could not find the file at:\n{DEFAULT_FILE_PATH}\n\nUpdate DEFAULT_FILE_PATH near the top of the Settings section in app.py to point to your file.")
    st.stop()

learner_df, merged_df = build_learner_profiles(users_df, courses_df, transactions_df)
cluster_input_df = prepare_cluster_input(learner_df)
labels, sil_score = run_kmeans(cluster_input_df, k)
learner_df["Cluster"] = labels

course_similarity_df = build_course_similarity(courses_df)
cluster_course_counts = build_cluster_course_popularity(merged_df, learner_df, courses_df)

per_cluster_sim, overall_sim = compute_intra_cluster_similarity(cluster_input_df, labels)
precision_score = compute_recommendation_precision(learner_df, merged_df, course_similarity_df,
                                                     cluster_course_counts, courses_df, top_n=top_n)
pers_rating, base_rating, lift_pct = compute_engagement_lift(learner_df, merged_df, course_similarity_df,
                                                               cluster_course_counts, courses_df, top_n=top_n)

@st.cache_data(show_spinner="Projecting segments to 2D...")
def compute_pca_coords(cluster_input_df, labels):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(cluster_input_df)
    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Cluster"] = labels.astype(str)
    return plot_df


# ============================================================
# 4. MAIN APP — TABS
# ============================================================

st.title("🎓 EduPro — Learner Segmentation & Recommendation Dashboard")

tab_kpi, tab_explorer, tab_dashboard, tab_recs, tab_compare = st.tabs(
    ["📈 KPIs & Validation", "👤 Learner Profile Explorer", "📊 Cluster Dashboard",
     "🎯 Recommendations", "⚖️ Segment Comparison"]
)

# ---------------- TAB 1: KPIs ----------------
with tab_kpi:
    st.subheader("Evaluation & Validation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette Score", f"{sil_score:.3f}", help="Cluster quality: how well-separated the segments are (-1 to 1, higher is better).")
    c2.metric("Intra-Cluster Similarity", f"{overall_sim:.3f}", help="Behavioural consistency: average similarity between learners inside the same segment (0 to 1, higher is better).")
    c3.metric("Recommendation Precision", f"{precision_score * 100:.1f}%", help="Relevance: of the top recommendations shown, what share match the learner's most recent interest category (proxy metric).")
    c4.metric("Engagement Lift (Proxy)", f"{lift_pct:+.1f}%", help="Impact estimate: how much higher-rated personalised recommendations are vs. the catalogue average.")

    st.divider()
    st.markdown(f"""
**How to read these:**
- **Silhouette Score ({sil_score:.3f})** — measures cluster quality using the same scaled features used for K-Means. Values above ~0.25 indicate reasonably distinct segments for behavioural data like this.
- **Intra-Cluster Similarity ({overall_sim:.3f})** — the average cosine similarity between every pair of learners *within* the same segment. Higher means learners inside a segment genuinely behave alike.
- **Recommendation Precision ({precision_score*100:.1f}%)** — a backtest proxy: for a sample of learners, we check whether their top-{top_n} recommendations share the category of the course they most recently bought.
- **Engagement Lift ({lift_pct:+.1f}%)** — compares the average rating of personalised recommendations (**{pers_rating:.2f}**) against the catalogue-wide average rating (**{base_rating:.2f}**), as a proxy for the quality uplift learners experience vs. a generic/random suggestion.
""")

    st.subheader("Intra-Cluster Similarity by Segment")
    sim_df = pd.DataFrame({"Cluster": list(per_cluster_sim.keys()), "Similarity": list(per_cluster_sim.values())})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(sim_df["Cluster"].astype(str), sim_df["Similarity"], color="#4C72B0")
    ax.set_xlabel("Cluster"); ax.set_ylabel("Avg. Similarity")
    st.pyplot(fig)

# ---------------- TAB 2: Learner Profile Explorer ----------------
with tab_explorer:
    st.subheader("Learner Profile Explorer")
    selected_user = st.selectbox("Select a learner (UserID)", learner_df["UserID"].sort_values())

    profile = learner_df.loc[learner_df.UserID == selected_user].iloc[0]
    seg = int(profile["Cluster"])
    seg_size = int((learner_df["Cluster"] == seg).sum())

    # ---- Prominent "assigned segment" banner ----
    st.success(f"### 🧩 Assigned Segment: **Cluster {seg}**  \n"
               f"This learner belongs to a segment of **{seg_size} learners** "
               f"(out of {len(learner_df)} total).")

    col1, col2, col3 = st.columns(3)
    col1.metric("Age", int(profile["Age"]))
    col1.metric("Gender", profile["Gender"])
    col2.metric("Total Courses Bought", int(profile["TotalCourses"]))
    col2.metric("Total Spend", f"${profile['TotalSpend']:.2f}")
    col2.metric("Avg. Spend", f"${profile['AvgSpend']:.2f}")
    col3.metric("Avg. Course Rating Bought", f"{profile['AvgCourseRating']:.2f}")
    col3.metric("Favourite Category", profile["FavouriteCategory"])
    col3.metric("Favourite Level", profile["FavouriteLevel"])

    st.markdown(f"**Days since last purchase (Recency):** {int(profile['Recency'])}  |  "
                f"**Free course ratio:** {profile['FreeCourseRatio']*100:.0f}%")

    st.divider()

    # ---- Visual 1: where this learner sits on the cluster map ----
    st.markdown("#### 📍 Where this learner sits among all segments")
    plot_df = compute_pca_coords(cluster_input_df, labels)
    plot_df["UserID"] = learner_df["UserID"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == c]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {c}", alpha=0.35, s=15)
    me = plot_df[plot_df["UserID"] == selected_user]
    ax.scatter(me["PC1"], me["PC2"], color="black", s=180, marker="*", label="Selected learner", zorder=5)
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("Selected learner's position within the segment map")
    ax.legend()
    st.pyplot(fig)

    # ---- Visual 2: learner vs. their segment vs. overall average ----
    st.markdown("#### 📊 This learner vs. their segment average vs. all learners")
    compare_metrics = ["TotalCourses", "TotalSpend", "AvgSpend", "AvgCourseRating", "Recency"]
    seg_avg = learner_df.loc[learner_df["Cluster"] == seg, compare_metrics].mean()
    overall_avg = learner_df[compare_metrics].mean()
    learner_vals = profile[compare_metrics]

    comparison = pd.DataFrame({
        "This Learner": learner_vals,
        f"Cluster {seg} Average": seg_avg,
        "All Learners Average": overall_avg,
    }).round(2)
    st.dataframe(comparison, width='stretch')

    fig, axes = plt.subplots(1, len(compare_metrics), figsize=(4 * len(compare_metrics), 3.2))
    for ax, m in zip(axes, compare_metrics):
        ax.bar(comparison.columns, comparison.loc[m], color=["#C44E52", "#4C72B0", "#55A868"])
        ax.set_title(m, fontsize=9)
        ax.tick_params(axis='x', labelrotation=20, labelsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Purchase History")
    history = merged_df.loc[merged_df.UserID == selected_user, ["CourseID", "CourseName", "CourseCategory", "CourseLevel", "TransactionDate", "Amount"]]
    st.dataframe(history.sort_values("TransactionDate", ascending=False), width='stretch')

# ---------------- TAB 3: Cluster Dashboard ----------------
with tab_dashboard:
    st.subheader("Cluster Visualisation Dashboard")

    plot_df = compute_pca_coords(cluster_input_df, labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(plot_df["Cluster"].unique()):
        subset = plot_df[plot_df["Cluster"] == c]
        ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {c}", alpha=0.6, s=15)
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    ax.set_title("Learner Segments (2D projection via PCA)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Segment Sizes")
    st.bar_chart(learner_df["Cluster"].value_counts().sort_index())

    st.subheader("Segment Profile Summary")
    cluster_profile = learner_df.groupby("Cluster").agg(
        Learners=("UserID", "count"),
        AvgAge=("Age", "mean"),
        AvgTotalCourses=("TotalCourses", "mean"),
        AvgTotalSpend=("TotalSpend", "mean"),
        AvgRating=("AvgCourseRating", "mean"),
        AvgRecency=("Recency", "mean"),
        TopCategory=("FavouriteCategory", most_common),
        TopLevel=("FavouriteLevel", most_common),
    ).round(1)
    st.dataframe(cluster_profile, width='stretch')

# ---------------- TAB 4: Recommendations ----------------
with tab_recs:
    st.subheader("Personalised Course Recommendations")

    rec_user = st.selectbox("Select a learner (UserID)", learner_df["UserID"].sort_values(), key="rec_user")

    rec_cluster = int(learner_df.loc[learner_df.UserID == rec_user, "Cluster"].values[0])
    st.success(f"🧩 **Assigned Segment:** Cluster {rec_cluster}")

    st.markdown("#### 🔍 Filter Recommendations")
    level_options = ["All"] + sorted(courses_df["CourseLevel"].unique().tolist())
    category_options = ["All"] + sorted(courses_df["CourseCategory"].unique().tolist())

    fcol1, fcol2 = st.columns(2)
    level_filter = fcol1.selectbox("Course Level", level_options)
    category_filter = fcol2.selectbox("Course Category", category_options)
    st.caption(f"Showing recommendations filtered to: Level = **{level_filter}**, Category = **{category_filter}**")

    recs = recommend_courses(rec_user, learner_df, merged_df, course_similarity_df,
                              cluster_course_counts, courses_df, top_n=top_n,
                              level_filter=level_filter, category_filter=category_filter)

    st.markdown("#### 🎯 Recommended Learning Path")
    if len(recs) == 0:
        st.warning("No courses match the selected filters for this learner.")
    else:
        st.dataframe(recs.reset_index(drop=True), width='stretch')

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(recs["CourseName"], recs["FinalScore"], color="#4C72B0")
        ax.invert_yaxis()
        ax.set_xlabel("Recommendation Score")
        ax.set_title("Why these courses were recommended (relative score)")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- TAB 5: Segment Comparison ----------------
with tab_compare:
    st.subheader("Segment Comparison Panel")

    cluster_options = sorted(learner_df["Cluster"].unique())
    chosen_clusters = st.multiselect("Select segments to compare", cluster_options, default=cluster_options[:2])

    if len(chosen_clusters) >= 2:
        compare_df = learner_df[learner_df["Cluster"].isin(chosen_clusters)]
        metrics = ["TotalCourses", "TotalSpend", "AvgSpend", "AvgCourseRating", "FreeCourseRatio", "Recency"]

        summary = compare_df.groupby("Cluster")[metrics].mean().round(2)
        st.dataframe(summary, width='stretch')

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
        for ax, m in zip(axes, metrics):
            ax.bar(summary.index.astype(str), summary[m], color="#55A868")
            ax.set_title(m, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Select at least 2 segments to compare.")
