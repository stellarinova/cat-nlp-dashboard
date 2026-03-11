import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="r/Cats Sentiment Discovery", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("sentiment_results_rcats.csv")
    df['created'] = pd.to_datetime(df['created'])
    return df

def main():
    st.title("🐱 r/Cats Community Insights")
    st.markdown("""
    This dashboard analyzes the emotional pulse of the Reddit cat community. 
    Use the search feature to see how users feel about specific topics.
    """)

    df = load_data()

    # --- SIDEBAR: SEARCH FEATURE ---
    st.sidebar.header("🔍 Reddit Search")
    search_query = st.sidebar.text_input("Enter a keyword (e.g., 'Vet', 'Cancer', 'Adopted')", "")

    if search_query:
        # Filter data based on search
        filtered_df = df[df['combined_text'].str.contains(search_query, case=False, na=False)]
        
        if not filtered_df.empty:
            st.subheader(f"Results for: '{search_query}'")
            
            # Metric Row
            col1, col2, col3 = st.columns(3)
            avg_sentiment = filtered_df['compound_score'].mean()
            col1.metric("Total Posts Found", len(filtered_df))
            col2.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            col3.metric("Supportive Threads", len(filtered_df[filtered_df['flair_group'] == 'Supportive/Crisis']))

            # Analysis Columns
            left_col, right_col = st.columns(2)

            with left_col:
                st.write("### Sentiment Distribution")
                fig = px.histogram(filtered_df, x="compound_score", 
                                   nbins=20, 
                                   color_discrete_sequence=['#22d3ee'],
                                   labels={'compound_score': 'Sentiment Intensity'})
                st.plotly_chart(fig, use_container_width=True)

            with right_col:
                st.write("### Top Associated Flairs")
                flair_counts = filtered_df['flair'].value_counts().reset_index()
                fig_pie = px.pie(flair_counts, values='count', names='flair', hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Show Raw Data Sample
            with st.expander("View Sample Posts"):
                st.table(filtered_df[['flair', 'title', 'compound_score']].head(10))
        else:
            st.warning(f"No posts found containing '{search_query}'. Try a different keyword.")
    else:
        # DEFAULT VIEW: Global Trends
        st.subheader("Global Community Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Average Sentiment by Category")
            group_avg = df.groupby('flair_group')['compound_score'].mean().reset_index()
            fig_bar = px.bar(group_avg, x='flair_group', y='compound_score', color='flair_group')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col2:
            st.write("#### Post Volume by Flair")
            fig_volume = px.treemap(df, path=['flair_group', 'flair'], values='compound_score')
            st.plotly_chart(fig_volume, use_container_width=True)

if __name__ == "__main__":
    main()
