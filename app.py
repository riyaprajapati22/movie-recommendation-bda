"""Step 3: simple web UI. Run: streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 WatchNext")
st.caption("Movie Recommendation System")


@st.cache_data
def load_all():
    recs = pd.read_csv("output/recommendations.csv")
    movies = pd.read_csv("output/movies.csv")
    factors = pd.read_pickle("output/item_factors.pkl")
    return recs, movies, factors


try:
    recs, movies, factors = load_all()
except FileNotFoundError:
    st.error("Run `python train.py` first to generate the model outputs.")
    st.stop()

try:
    st.sidebar.metric("Model test RMSE", open("output/metrics.txt").read().split("=")[1].strip())
except Exception:
    pass

tab1, tab2 = st.tabs(["Recommend for a user", "Similar movies"])

with tab1:
    uid = st.selectbox("User ID", sorted(recs.userId.unique()))
    st.subheader(f"Top 10 picks for user {uid}")

    user_recs = recs[recs.userId == uid][["title", "genres", "score"]].copy()

    # Keep predicted ratings between 0.5 and 5.0
    user_recs["score"] = user_recs["score"].clip(lower=0.5, upper=5.0)
    user_recs["score"] = user_recs["score"].round(1)

    # Display rating as "4.7 / 5.0"
    user_recs["score"] = user_recs["score"].apply(
        lambda x: f"⭐ {x:.1f} / 5.0"
    )

    # Rename column
    user_recs = user_recs.rename(columns={"score": "Predicted Rating"})

    st.dataframe(
        user_recs.reset_index(drop=True),
        use_container_width=True
    )

with tab2:
    titles = movies.merge(factors, on="movieId").sort_values("title")
    choice = st.selectbox("Pick a movie", titles.title.tolist())
    mat = np.vstack(titles.features.values)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    i = titles.index.get_loc(titles[titles.title == choice].index[0])
    sims = mat @ mat[i]
    top = np.argsort(-sims)[1:11]
    out = titles.iloc[top][["title", "genres"]].copy()

# Convert similarity score to percentage
out["similarity"] = np.round(sims[top] * 100, 1)

# Add % symbol
out["similarity"] = out["similarity"].apply(lambda x: f"{x:.1f}%")

# Rename column
out = out.rename(columns={"similarity": "Similarity"})

st.subheader("Movies you might also like")
st.dataframe(
    out.reset_index(drop=True),
    use_container_width=True
)
