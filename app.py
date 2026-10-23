import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://raw.githubusercontent.com/ronaldleonardo/Data_Transformer_Datasets/refs/heads/main/bbc_text_cls.csv"

st.set_page_config(page_title="News Article Classifier", page_icon="📰", layout="wide")

st.title("📰 News Article Classification")
st.write("Classify a news article into a category using **TF-IDF and Multinomial Naive Bayes**.")

@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_URL)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(columns, candidates):
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for c in columns:
        cl = str(c).strip().lower()
        if any(candidate in cl for candidate in candidates):
            return c
    return None

try:
    raw_df = load_dataset()
except Exception as exc:
    st.error("I couldn't load the dataset. Check your internet connection or the dataset URL.")
    st.exception(exc)
    st.stop()

text_col = find_column(raw_df.columns, ["text", "article", "content", "news", "body"])
label_col = find_column(raw_df.columns, ["category", "label", "class", "topic", "type"])

if text_col is None or label_col is None:
    st.error("Couldn't automatically identify the article-text and category columns.")
    st.write("Columns found in the CSV:", list(raw_df.columns))
    st.info("Update the text_col and label_col selection in app.py to match the dataset columns.")
    st.stop()

df = raw_df[[text_col, label_col]].copy()
df.columns = ["article_text", "category"]
df["article_text"] = df["article_text"].fillna("").astype(str).str.strip()
df["category"] = df["category"].fillna("").astype(str).str.strip()
df = df[(df["article_text"] != "") & (df["category"] != "")]
df = df.drop_duplicates(subset=["article_text"]).reset_index(drop=True)

if df["category"].nunique() < 2:
    st.error("The dataset needs at least two different categories to train a classifier.")
    st.stop()

# Use a stratified hold-out set when each class has enough examples.
class_counts = df["category"].value_counts()
stratify_labels = df["category"] if class_counts.min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    df["article_text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))),
    ("nb", MultinomialNB()),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

tab1, tab2, tab3 = st.tabs(["🔎 Predict a category", "📊 Dataset overview", "🧪 Model evaluation"])

with tab1:
    st.subheader("Try your own news article")
    article = st.text_area(
        "Paste a headline or article text here:",
        height=180,
        placeholder="Paste a news article to predict its category...",
    )
    if st.button("Classify article", type="primary"):
        if len(article.strip()) < 10:
            st.warning("Please enter a little more text so the model has enough information.")
        else:
            prediction = model.predict([article.strip()])[0]
            st.success(f"Predicted category: **{prediction}**")
            if hasattr(model.named_steps["nb"], "predict_log_proba"):
                probs = model.predict_proba([article.strip()])[0]
                probability_df = pd.DataFrame({
                    "Category": model.named_steps["nb"].classes_,
                    "Model score (%)": probs * 100,
                }).sort_values("Model score (%)", ascending=False)
                st.write("Category scores (these are model probabilities, not guarantees):")
                st.dataframe(
                    probability_df.style.format({"Model score (%)": "{:.2f}%"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.bar_chart(probability_df.set_index("Category"))

    st.divider()
    st.subheader("Try an article from the dataset")
    if st.button("Load a random example"):
        sample = df.sample(1, random_state=None).iloc[0]
        st.session_state["sample_article"] = sample["article_text"]
        st.session_state["sample_category"] = sample["category"]
    if "sample_article" in st.session_state:
        st.text_area("Example article", key="sample_article_display",
                     value=st.session_state["sample_article"], height=150, disabled=True)
        st.caption(f"Actual dataset category: {st.session_state['sample_category']}")

with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles", f"{len(df):,}")
    col2.metric("Categories", df["category"].nunique())
    col3.metric("Test accuracy", f"{accuracy:.1%}")
    st.subheader("Articles by category")
    counts = df["category"].value_counts().rename_axis("Category").to_frame("Articles")
    st.bar_chart(counts)
    with st.expander("Preview cleaned dataset"):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
    st.caption(f"Text column used: {text_col} · Category column used: {label_col}")

with tab3:
    st.metric("Hold-out test accuracy", f"{accuracy:.2%}")
    st.write(f"Training articles: **{len(X_train):,}** · Test articles: **{len(X_test):,}**")
    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    st.subheader("Confusion matrix")
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    st.info("Accuracy depends on the dataset and the random train/test split. The model is a learning project, not a source-verification tool.")

st.sidebar.header("About this project")
st.sidebar.write("A text-classification demo built with Streamlit and scikit-learn.")
st.sidebar.markdown("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.markdown("**Text features:** TF-IDF")
st.sidebar.markdown("**Evaluation:** 80/20 train-test split")
