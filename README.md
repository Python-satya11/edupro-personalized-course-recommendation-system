# 🎓 EduPro — Student Segmentation & Personalised Course Recommendation System

A learner-segmentation and course-recommendation pipeline for EduPro, built with K-Means / Hierarchical Clustering and a cluster-aware recommendation engine, delivered as an interactive Streamlit web application.

---

## Streamlit Web Application

### Core Modules
- Learner profile explorer
- Cluster visualisation dashboard
- Personalised course recommendations
- Segment comparison panels

### User Capabilities
- Select a learner profile
- View assigned segment
- See recommended learning paths
- Filter recommendations by level or category

---

## 📊 Evaluation & Validation Metrics

| Metric | Purpose | Value |
|---|---|---|
| Silhouette Score | Cluster Quality | **0.201** |
| Intra-Cluster Similarity | Behavioural consistency | **0.485** |
| Recommendation Precision | Relevance | **12.9%** |
| Engagement Lift (Proxy) | Impact Estimate | **+21.8%** |

**How to read these:**
- **Silhouette Score (0.201)** — measures how well-separated the 4 learner segments are (range -1 to 1; higher is better). A score in this range indicates real but overlapping behavioural groups, which is expected for human behaviour data.
- **Intra-Cluster Similarity (0.485)** — the average cosine similarity between learners *within* the same segment. Higher means learners inside a segment behave more alike.
- **Recommendation Precision (12.9%)** — a backtest proxy: the share of top recommended courses that match a learner's most recent interest category.
- **Engagement Lift (+21.8%)** — how much higher-rated personalised recommendations are, on average, compared to the overall course catalogue rating — a proxy for the expected engagement improvement over generic suggestions.

### 📈 Supporting Charts

**Elbow & Silhouette Analysis (choosing k = 4):**
![Silhouette Score](silhouette%20score.png)

**Average Spend & Course Count per Segment:**
![Average spend and course count per cluster](average%20spend%20and%20course%20count%20per%20cluster.png)

---

## 🚀 Deploying to Streamlit Community Cloud

To publish this app on [Streamlit Community Cloud](https://streamlit.io/cloud), your GitHub repository needs the following files — **not** the chart images, which are only used for this README, not the running app:

| File | Required? | Purpose |
|---|---|---|
| `streamlit_app.py` | ✅ Required | The main app script. Streamlit Community Cloud looks for this as the entry point by default (any filename works if you specify it in the deploy settings, but `streamlit_app.py` is the standard convention). |
| `requirements.txt` | ✅ Required | Lists the Python packages Streamlit Cloud must install (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`). |
| `EduPro_Online_Platform.xlsx` | ✅ Required | Your dataset. It must be committed to the repo so the deployed app can read it. |
| `README.md` | Optional | This file — Streamlit Cloud can display it, and GitHub renders it automatically. The two `.png` images referenced above are only needed if you want this README to show the charts on GitHub. |

### ⚠️ Important fix needed before deploying
Your current `app.py` loads the dataset from a hardcoded Windows path:
DEFAULT_FILE_PATH = r"C:\Users\ranja\Downloads\EduPro Online Platform.xlsx"
This will **not** work on Streamlit Community Cloud, since it runs on a Linux server with no access to your PC. Before deploying, change this line to a **relative path** and place the Excel file in the same repo folder as the script:
DEFAULT_FILE_PATH = "EduPro_Online_Platform.xlsx"

### Steps to publish
1. Rename (or copy) `app.py` to `streamlit_app.py`.
2. Create `requirements.txt` with:
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   openpyxl
3. Push `streamlit_app.py`, `requirements.txt`, and `EduPro_Online_Platform.xlsx` to a GitHub repository.
4. Go to [share.streamlit.io](https://share.streamlit.io), sign in, click **New app**, select your repo/branch, and set the main file path to `streamlit_app.py`.
5. Click **Deploy**.

## Author 

**Satyaranjan Jena**
MCA
🔗 [LinkedIn](https://www.linkedin.com/in/satyaranjan-jena09/)
