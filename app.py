# ==========================================
# FeedTrack — Company Review Analytics Dashboard
# Author: Rohit Pal
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="FeedTrack - Review Insights Dashboard", layout="wide")

st.title("📊 FeedTrack — Company Review Analytics Dashboard")
st.markdown("""
Analyze company review datasets and generate insights on sentiment, ratings, and trends.  
Upload your `.csv` file below to explore results interactively.
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload Company Review CSV", type=["csv"])

if uploaded_file is not None:
    # --- Read CSV using pandas ---
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["Date"], dayfirst=False)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    # --- Clean column names ---
    df.columns = df.columns.str.strip().str.title()

    if "Review" not in df.columns:
        st.error("❌ The uploaded file must contain a 'Review' column.")
        st.stop()

    company_name = df["Company"].iloc[0] if "Company" in df.columns else "Uploaded Dataset"
    st.success(f"✅ Loaded dataset for **{company_name}** with {len(df)} reviews.")
    st.dataframe(df.head())

    # ---------------- SENTIMENT ANALYSIS ----------------
    st.subheader("🧠 Sentiment Analysis")

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        if not isinstance(text, str):
            return 0
        return analyzer.polarity_scores(text)["compound"]

    df["Sentiment_Score"] = df["Review"].apply(get_sentiment)
    df["Sentiment_Label"] = pd.cut(
        df["Sentiment_Score"],
        bins=[-1.1, -0.05, 0.05, 1.1],
        labels=["Negative", "Neutral", "Positive"]
    )

    st.write(df[["Review", "Sentiment_Score", "Sentiment_Label"]].head())

    # ---------------- METRICS ----------------
    st.subheader("📈 Key Metrics")

    col1, col2, col3 = st.columns(3)
    avg_rating = df["Rating"].mean() if "Rating" in df.columns else np.nan
    pos_percent = (df["Sentiment_Label"] == "Positive").mean() * 100
    neg_percent = (df["Sentiment_Label"] == "Negative").mean() * 100

    col1.metric("Average Rating", f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A")
    col2.metric("Positive Sentiment", f"{pos_percent:.1f}%")
    col3.metric("Negative Sentiment", f"{neg_percent:.1f}%")

    # ---------------- VISUALIZATIONS ----------------
    st.subheader("📊 Visual Insights")

    tab1, tab2, tab3 = st.tabs(["Rating & Sentiment", "Wordcloud", "Department & Source"])

    with tab1:
        col1, col2 = st.columns(2)

        # Rating Distribution
        if "Rating" in df.columns:
            fig1 = px.histogram(
                df, x="Rating", color="Sentiment_Label", nbins=10,
                title="Rating Distribution by Sentiment", barmode="overlay"
            )
            col1.plotly_chart(fig1, use_container_width=True)

        # Sentiment Pie
        fig2 = px.pie(
            df, names="Sentiment_Label",
            title="Overall Sentiment Composition",
            color_discrete_sequence=["#ff6b6b", "#feca57", "#1dd1a1"]
        )
        col2.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### ☁️ Review WordCloud")
        text = " ".join(df["Review"].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No review text found for wordcloud generation.")

    with tab3:
        col1, col2 = st.columns(2)

        # --- Department Chart ---
        if "Department" in df.columns:
            fig3 = px.box(
                df, x="Department", y="Rating", color="Department",
                title="Rating by Department", points="all"
            )
            col1.plotly_chart(fig3, use_container_width=True)
        else:
            col1.info("No Department column found.")

        # --- Fixed Source Chart (bug-free) ---
        if "Source" in df.columns:
            source_df = df["Source"].value_counts().reset_index()
            source_df.columns = ["Source", "Count"]
            fig4 = px.bar(
                source_df,
                x="Source", y="Count",
                title="Review Source Count",
                color="Source",
                text="Count"
            )
            fig4.update_traces(textposition="outside")
            col2.plotly_chart(fig4, use_container_width=True)
        else:
            col2.info("No Source column found.")

    # ---------------- TIME SERIES ----------------
    if "Date" in df.columns:
        st.subheader("📆 Rating Over Time")
        df_sorted = df.sort_values("Date")
        fig5 = px.line(
            df_sorted, x="Date", y="Rating",
            color="Sentiment_Label",
            markers=True,
            title=f"Rating Trend Over Time — {company_name}"
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ---------------- AUTO INSIGHTS ----------------
    st.subheader("💡 Automated Insights")

    insights = []
    insights.append(f"Average rating is **{avg_rating:.2f}** with {pos_percent:.1f}% positive sentiment.")
    if neg_percent > 30:
        insights.append("⚠️ High negative sentiment detected — potential employee dissatisfaction.")
    if avg_rating >= 4.0:
        insights.append("🌟 Excellent overall satisfaction — company culture seems strong.")
    if "Department" in df.columns:
        dept_avg = df.groupby("Department")["Rating"].mean().sort_values()
        if not dept_avg.empty:
            insights.append(f"Lowest-rated department: **{dept_avg.index[0]} ({dept_avg.iloc[0]:.2f})**.")
    if "Source" in df.columns:
        top_source = df["Source"].value_counts().idxmax()
        insights.append(f"Most reviews collected from **{top_source}**.")

    for insight in insights:
        st.markdown(f"- {insight}")

    # ---------------- EXPORT ----------------
    st.subheader("📥 Export Processed Data")

    csv_out = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Processed CSV",
        data=csv_out,
        file_name=f"feedtrack_processed_{company_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a company review dataset CSV to begin analysis.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ by **Rohit Pal** — FeedTrack Project (Data Analytics Final Year) 🚀")
