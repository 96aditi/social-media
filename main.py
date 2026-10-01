import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. SOCIAL MEDIA DATA
# =========================================================

data = {

    "post_id": list(range(1, 16)),

    "user": [
        "Rahul", "Priya", "Aman", "Neha", "Riya",
        "Karan", "Rahul", "Aditi", "Priya", "Neha",
        "Aman", "Riya", "Karan", "Rahul", "Priya"
    ],

    "content": [

        "Learn ethical hacking and cybersecurity basics",

        "How to protect your computer from cyber attacks",

        "Python programming for beginners",

        "Machine learning with Python",

        "Network security and penetration testing",

        "Artificial intelligence and deep learning",

        "Latest cybersecurity threats and security tools",

        "Linux commands for ethical hacking",

        "Python automation and scripting",

        "AI applications in cybersecurity",

        "Cloud computing and cloud security",

        "Data science and data visualization",

        "Web application security and OWASP",

        "Advanced Python programming techniques",

        "Artificial intelligence in cybersecurity"
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
        "Artificial Intelligence",
        "Cloud Computing",
        "Data Science",
        "Cybersecurity",
        "Programming",
        "Artificial Intelligence"
    ],

    "likes": [
        120, 90, 150, 180, 110,
        220, 160, 130, 140, 200,
        170, 155, 190, 145, 230
    ],

    "views": [
        1000, 800, 1500, 2000, 900,
        2500, 1800, 1200, 1600, 2300,
        1900, 1700, 2100, 1550, 2700
    ],

    "comments": [
        30, 20, 40, 35, 25,
        50, 45, 28, 30, 48,
        38, 32, 44, 31, 55
    ],

    "shares": [
        25, 15, 35, 40, 20,
        60, 45, 30, 32, 55,
        42, 35, 50, 33, 65
    ],

    "days_old": [
        2, 5, 1, 10, 4,
        3, 7, 1, 6, 2,
        5, 8, 3, 4, 1
    ]
}


df = pd.DataFrame(data)


# =========================================================
# 2. USER PROFILES
# =========================================================

users = {

    "Aditi": {
        "age": 22,
        "interests": [
            "Cybersecurity",
            "Programming",
            "Cloud Computing"
        ]
    },

    "Rahul": {
        "age": 24,
        "interests": [
            "Programming",
            "Machine Learning"
        ]
    },

    "Priya": {
        "age": 23,
        "interests": [
            "Artificial Intelligence",
            "Machine Learning",
            "Data Science"
        ]
    },

    "Neha": {
        "age": 25,
        "interests": [
            "Cloud Computing",
            "Cybersecurity"
        ]
    }
}


# =========================================================
# 3. USER INTERACTION HISTORY
# =========================================================

interaction_history = {

    "Aditi": {
        "liked_posts": [1, 5, 8, 13],
        "viewed_posts": [1, 2, 5, 8, 13, 15],
        "commented_posts": [1, 8],
        "shared_posts": [5, 13]
    },

    "Rahul": {
        "liked_posts": [3, 4, 9, 14],
        "viewed_posts": [3, 4, 9, 14],
        "commented_posts": [3, 4],
        "shared_posts": [9]
    },

    "Priya": {
        "liked_posts": [6, 10, 12, 15],
        "viewed_posts": [6, 10, 12, 15],
        "commented_posts": [6, 10],
        "shared_posts": [15]
    },

    "Neha": {
        "liked_posts": [2, 5, 11],
        "viewed_posts": [2, 5, 11],
        "commented_posts": [11],
        "shared_posts": [5]
    }
}


# =========================================================
# 4. ENGAGEMENT SCORE
# =========================================================

df["engagement_score"] = (

    df["likes"] * 1 +

    df["comments"] * 2 +

    df["shares"] * 3

)


# =========================================================
# 5. ENGAGEMENT RATE
# =========================================================

df["engagement_rate"] = (

    (
        df["likes"] +
        df["comments"] +
        df["shares"]
    )
    /
    df["views"]

)


# =========================================================
# 6. CONTENT FEATURES
# =========================================================

df["features"] = (

    df["content"] + " " +

    df["category"]

)


# =========================================================
# 7. TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(

    stop_words="english",

    ngram_range=(1, 2)

)


tfidf_matrix = vectorizer.fit_transform(

    df["features"]

)


# =========================================================
# 8. CONTENT SIMILARITY
# =========================================================

similarity_matrix = cosine_similarity(

    tfidf_matrix

)


# =========================================================
# 9. SENTIMENT ANALYSIS
# =========================================================

positive_words = [

    "learn",
    "latest",
    "advanced",
    "applications",
    "automation",
    "deep",
    "security",
    "protect",
    "programming",
    "techniques"

]


negative_words = [

    "attack",
    "threat",
    "danger",
    "problem",
    "risk"

]


def calculate_sentiment(text):

    text = text.lower()

    positive = 0

    negative = 0

    for word in positive_words:

        if word in text:

            positive += 1


    for word in negative_words:

        if word in text:

            negative += 1


    if positive > negative:

        return "Positive"

    elif negative > positive:

        return "Negative"

    else:

        return "Neutral"


df["sentiment"] = df["content"].apply(

    calculate_sentiment

)


# =========================================================
# 10. RECENCY SCORE
# =========================================================

df["recency_score"] = np.exp(

    -0.15 * df["days_old"]

)


# =========================================================
# 11. NORMALIZATION FUNCTION
# =========================================================

def normalize(series):

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(

            [1] * len(series),

            index=series.index

        )

    return (

        (series - minimum)

        /

        (maximum - minimum)

    )


df["engagement_normalized"] = normalize(

    df["engagement_score"]

)


df["popularity_score"] = normalize(

    df["views"]

)


# =========================================================
# 12. USER INTEREST SCORE
# =========================================================

def calculate_interest_score(

    category,

    interests

):

    if category in interests:

        return 1.0

    return 0.0


# =========================================================
# 13. USER BEHAVIOR SCORE
# =========================================================

def calculate_behavior_score(

    post_id,

    username

):

    history = interaction_history.get(

        username,

        {}

    )


    score = 0


    if post_id in history.get(

        "liked_posts", []

    ):

        score += 0.5


    if post_id in history.get(

        "viewed_posts", []

    ):

        score += 0.2


    if post_id in history.get(

        "commented_posts", []

    ):

        score += 0.3


    if post_id in history.get(

        "shared_posts", []

    ):

        score += 0.5


    return min(score, 1.0)


# =========================================================
# 14. PERSONALIZED RECOMMENDATION ENGINE
# =========================================================

def recommend_content(

    username,

    number_of_recommendations=5

):


    # -----------------------------------------------------
    # Check user
    # -----------------------------------------------------

    if username not in users:

        print(

            "\nNew user detected."

        )

        print(

            "Using popular content recommendation."

        )


        interests = []

    else:

        interests = users[username][

            "interests"

        ]


    recommendations = []


    # -----------------------------------------------------
    # Calculate scores
    # -----------------------------------------------------

    for index, row in df.iterrows():


        # User interest

        interest_score = (

            calculate_interest_score(

                row["category"],

                interests

            )

        )


        # Content similarity

        content_score = (

            similarity_matrix[index].mean()

        )


        # Engagement

        engagement_score = (

            row["engagement_normalized"]

        )


        # Popularity

        popularity_score = (

            row["popularity_score"]

        )


        # Recency

        recency_score = (

            row["recency_score"]

        )


        # User behavior

        behavior_score = (

            calculate_behavior_score(

                row["post_id"],

                username

            )

        )


        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        final_score = (

            interest_score * 0.30 +

            content_score * 0.15 +

            engagement_score * 0.15 +

            popularity_score * 0.10 +

            recency_score * 0.15 +

            behavior_score * 0.15

        )


        recommendations.append({

            "index": index,

            "score": final_score,

            "interest": interest_score,

            "engagement": engagement_score,

            "popularity": popularity_score,

            "recency": recency_score

        })


    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    recommendations = sorted(

        recommendations,

        key=lambda x: x["score"],

        reverse=True

    )


    # -----------------------------------------------------
    # Diversity control
    # -----------------------------------------------------

    selected = []

    categories_used = set()


    for recommendation in recommendations:

        index = recommendation["index"]

        category = df.iloc[index]["category"]


        # Don't recommend own post

        if (

            username in df.iloc[index]["user"]

        ):

            continue


        # Avoid too many same-category posts

        if category in categories_used:

            same_category_count = sum(

                1

                for item in selected

                if item["category"] == category

            )

            if same_category_count >= 2:

                continue


        selected.append({

            "index": index,

            "score": recommendation["score"],

            "category": category

        })


        categories_used.add(category)


        if len(selected) >= number_of_recommendations:

            break


    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print("\n")

    print("=" * 70)

    print(

        "PERSONALIZED RECOMMENDATIONS"

    )

    print(

        "USER:", username

    )

    print("=" * 70)


    for rank, item in enumerate(

        selected,

        start=1

    ):

        index = item["index"]

        row = df.iloc[index]


        print("\n")

        print(

            f"#{rank} {row['content']}"

        )

        print(

            "Category:",

            row["category"]

        )

        print(

            "Sentiment:",

            row["sentiment"]

        )

        print(

            "Likes:",

            row["likes"]

        )

        print(

            "Views:",

            row["views"]

        )

        print(

            "Comments:",

            row["comments"]

        )

        print(

            "Shares:",

            row["shares"]

        )

        print(

            "Engagement Rate:",

            round(

                row["engagement_rate"] * 100,

                2

            ),

            "%"

        )

        print(

            "Recommendation Score:",

            round(

                item["score"],

                3

            )

        )

        print("-" * 70)


# =========================================================
# 15. TRENDING CONTENT
# =========================================================

def show_trending_content():

    print("\n")

    print("=" * 70)

    print("🔥 TRENDING CONTENT")

    print("=" * 70)


    trending = df.sort_values(

        by=[

            "engagement_score",

            "views"

        ],

        ascending=False

    )


    for _, row in trending.head(5).iterrows():

        print(

            f"\n📌 {row['content']}"

        )

        print(

            "Category:",

            row["category"]

        )

        print(

            "Views:",

            row["views"]

        )

        print(

            "Engagement:",

            row["engagement_score"]

        )


# =========================================================
# 16. ANALYTICS DASHBOARD
# =========================================================

def show_analytics():

    print("\n")

    print("=" * 70)

    print("📊 SOCIAL MEDIA ANALYTICS")

    print("=" * 70)


    print(

        "\nTotal Posts:",

        len(df)

    )


    print(

        "Total Views:",

        df["views"].sum()

    )


    print(

        "Total Likes:",

        df["likes"].sum()

    )


    print(

        "Total Comments:",

        df["comments"].sum()

    )


    print(

        "Total Shares:",

        df["shares"].sum()

    )


    print(

        "\nAverage Engagement Rate:",

        round(

            df["engagement_rate"].mean() * 100,

            2

        ),

        "%"

    )


    print(

        "\n📂 Category Performance:"

    )


    category_stats = (

        df.groupby("category")

        .agg({

            "views": "sum",

            "likes": "sum",

            "comments": "sum",

            "shares": "sum"

        })

        .sort_values(

            "likes",

            ascending=False

        )

    )


    print(

        category_stats

    )


    print(

        "\n😊 Sentiment Distribution:"

    )


    print(

        df["sentiment"]

        .value_counts()

    )


# =========================================================
# 17. USER PROFILE
# =========================================================

def show_user_profile(username):

    if username not in users:

        print(

            "\nUser not found."

        )

        return


    user = users[username]

    history = interaction_history.get(

        username,

        {}

    )


    print("\n")

    print("=" * 70)

    print("👤 USER PROFILE")

    print("=" * 70)


    print(

        "Name:",

        username

    )


    print(

        "Age:",

        user["age"]

    )


    print(

        "Interests:",

        ", ".join(user["interests"])

    )


    print(

        "Liked Posts:",

        len(history.get(

            "liked_posts",

            []

        ))

    )


    print(

        "Viewed Posts:",

        len(history.get(

            "viewed_posts",

            []

        ))

    )


    print(

        "Comments:",

        len(history.get(

            "commented_posts",

            []

        ))

    )


    print(

        "Shares:",

        len(history.get(

            "shared_posts",

            []

        ))

    )


# =========================================================
# 18. MAIN PROGRAM
# =========================================================

if __name__ == "__main__":


    # Show analytics

    show_analytics()


    # Show profile

    show_user_profile(

        "Aditi"

    )


    # Show recommendations

    recommend_content(

        "Aditi",

        number_of_recommendations=5

    )


    # Show trending content

    show_trending_content()