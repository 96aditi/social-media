import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. DATA
# =========================

data = {
    "post_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    "user": [
        "Aditi", "Rahul", "Priya", "Aman", "Neha",
        "Riya", "Karan", "Aditi", "Rahul", "Priya"
    ],

    "content": [
        "Learn ethical hacking and cybersecurity",
        "Protect computer from cyber attacks",
        "Python programming for beginners",
        "Machine learning with Python",
        "Network security and penetration testing",
        "Artificial intelligence and deep learning",
        "Latest cybersecurity threats and tools",
        "Linux commands for ethical hacking",
        "Python automation and scripting",
        "AI applications in cybersecurity"
    ],

    "category": [
        "Cybersecurity",
        "Cybersecurity",
        "Programming",
        "Machine Learning",
        "Cybersecurity",
        "Artificial Intelligence",
        "Cybersecurity",
        "Cybersecurity",
        "Programming",
        "Artificial Intelligence"
    ],

    "likes": [120, 90, 150, 180, 110, 220, 160, 130, 140, 200],

    "views": [1000, 800, 1500, 2000, 900, 2500, 1800, 1200, 1600, 2300],

    "comments": [30, 20, 40, 35, 25, 50, 45, 28, 30, 48],

    "shares": [25, 15, 35, 40, 20, 60, 45, 30, 32, 55]
}

df = pd.DataFrame(data)


# =========================
# 2. ENGAGEMENT SCORE
# =========================

df["engagement"] = (
    df["likes"]
    + df["comments"] * 2
    + df["shares"] * 3
)


# =========================
# 3. TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf = vectorizer.fit_transform(
    df["content"]
)


# =========================
# 4. USER INTEREST
# =========================

user_interests = {
    "Aditi": [
        "Cybersecurity",
        "Programming"
    ],

    "Rahul": [
        "Programming",
        "Machine Learning"
    ],

    "Priya": [
        "Artificial Intelligence",
        "Machine Learning"
    ]
}


# =========================
# 5. RECOMMENDATION
# =========================

def recommend(user, number=5):

    interests = user_interests.get(
        user,
        []
    )

    scores = []

    for i in range(len(df)):

        # User interest
        interest_score = 1 if df.iloc[i]["category"] in interests else 0

        # Content similarity
        similarity = cosine_similarity(
            tfidf[i],
            tfidf
        ).mean()

        # Engagement
        engagement = (
            df.iloc[i]["engagement"]
            / df["engagement"].max()
        )

        # Final score
        score = (
            interest_score * 0.5
            + similarity * 0.3
            + engagement * 0.2
        )

        scores.append(score)


    df["recommendation_score"] = scores


    result = df[
        df["user"] != user
    ].sort_values(
        "recommendation_score",
        ascending=False
    ).head(number)


    return result


# =========================
# 6. STREAMLIT UI
# =========================

st.title("📱 Social Media Analytics")

st.write(
    "Content Recommendation System"
)


# User selection

user = st.selectbox(
    "Select User",
    ["Aditi", "Rahul", "Priya"]
)


# Number of recommendations

number = st.slider(
    "Number of Recommendations",
    1,
    10,
    5
)


# Button

if st.button("🎯 Recommend Content"):

    result = recommend(
        user,
        number
    )

    st.subheader(
        f"Recommendations for {user}"
    )


    for _, row in result.iterrows():

        st.write(
            "### 📌",
            row["content"]
        )

        st.write(
            "Category:",
            row["category"]
        )

        st.write(
            "❤️ Likes:",
            row["likes"]
        )

        st.write(
            "👁 Views:",
            row["views"]
        )

        st.write(
            "💬 Comments:",
            row["comments"]
        )

        st.write(
            "🔄 Shares:",
            row["shares"]
        )

        st.write(
            "⭐ Recommendation Score:",
            round(
                row["recommendation_score"],
                3
            )
        )

        st.divider()


# =========================
# 7. ANALYTICS
# =========================

st.subheader(
    "📊 Social Media Analytics"
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Posts",
    len(df)
)

col2.metric(
    "Total Likes",
    df["likes"].sum()
)

col3.metric(
    "Total Views",
    df["views"].sum()
)


# =========================
# 8. CHART
# =========================

st.subheader(
    "📈 Engagement by Category"
)


category_data = (
    df.groupby("category")["engagement"]
    .sum()
)


fig, ax = plt.subplots()

category_data.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel(
    "Category"
)

ax.set_ylabel(
    "Engagement"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

st.pyplot(fig)