import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="r/Cats Community Insights",
    page_icon="🐱",
    layout="wide"
)

# 2. DATA LOADING (with caching for performance)
@st.cache_data
def load_data():
    """
    Loads the processed sentiment results.
    Ensure 'sentiment_results_rcats.csv' is in your project directory.
    """
    try:
        df = pd.read_csv("sentiment_results_rcats.csv")
        df['created'] = pd.to_datetime(df['created'])
        print('original data size is ')
        print(len(df))        
        # We create a positive column for bubble size to avoid the Plotly ValueError
        # but we won't show this raw number to the user.
        df['bubble_size'] = df['compound_score'] + 1.1 
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    # --- NAVIGATION MENU ---
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.selectbox(
        "Select a Page", 
        ["r/Cats Community Insights", "Distribution of Sentiment", "Community Engagement", "Supportive Pulse"]
    )

    df = load_data()
    print('data size : ' )
    print(len(df))    
    if df.empty:
        st.warning("Please ensure 'sentiment_results_rcats.csv' is in your project folder.")
        return

    # --- PAGE 1: r/Cats COMMUNITY INSIGHTS (Landing Page) ---
    if page == "r/Cats Community Insights":
        st.title("🐱 r/Cats Community Insights")
        
        # Metrics Row - Original Format
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts Analyzed", len(df))
        col2.metric("Overall Sentiment", round(df['compound_score'].mean(), 2))
        col3.metric("Top Activity Category", df['flair'].mode()[0])
        
        st.markdown("---")
        
        # Main Layout: Text on left, Chart/Search on right
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("""
            ### Welcome to the Empathy Dashboard
            This research project analyzes the emotional landscape of the `r/Cats` subreddit. 
            Using over 5,000 posts, we examine how digital communities interact during times of joy and crisis.
            
            **Key Research Goals:**
            - Identifying patterns of digital support.
            - Quantifying community response to medical crises.
            - Visualizing the 'Supportiveness Gap' across flairs.
            """)
            st.info("💡 Use the sidebar to navigate between different data views.")

        with right_col:
            st.subheader("🔍 Sentiment Search")
            search_query = st.text_input("Enter keyword (e.g., 'Vet', 'Adopt', 'Mourning')", "")
            
            if search_query:
                filtered_df = df[df['combined_text'].str.contains(search_query, case=False, na=False)]
                if not filtered_df.empty:
                    st.write(f"Results for '{search_query}':")
                    st.dataframe(filtered_df[['flair', 'title', 'compound_score']].head(10), use_container_width=True)
                else:
                    st.error("No matches found.")
            else:
                # Original Landing Page Chart: Avg Sentiment by Flair
                avg_sentiment = df.groupby('flair')['compound_score'].mean().sort_values().reset_index()
                fig_landing = px.bar(
                    avg_sentiment, 
                    x='compound_score', 
                    y='flair', 
                    orientation='h',
                    title="Average Sentiment by Flair",
                    color='compound_score',
                    color_continuous_scale='RdYlGn',
                    template="plotly_dark"
                )
                st.plotly_chart(fig_landing, use_container_width=True)

    # --- PAGE 2: DISTRIBUTION OF SENTIMENT ---
    elif page == "Distribution of Sentiment":
        st.title("📊 Distribution of Sentiment")
        st.write("Visualizing the spread of emotional polarity across all community posts.")
        
        fig = px.histogram(
            df, 
            x="compound_score", 
            nbins=50, 
            color="flair_group",
            title="Emotional Polarity Distribution",
            labels={'compound_score': 'Sentiment Score (-1.0 to 1.0)'},
            template="plotly_dark",
            color_discrete_map={
                'Supportive/Crisis': '#FF4B4B', 
                'Casual/Entertainment': '#00CC96',
                'Other/General': '#636EFA'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 3: COMMUNITY ENGAGEMENT ---
    elif page == "Community Engagement":
        st.title("🤝 Community Engagement")
        st.write("Does sentiment influence how many comments or upvotes a post receives?")
        
        # REVERTED: Original Scatter Chart Rollback
        # Note: 'size' uses bubble_size to prevent errors, but 'hover_data' shows real sentiment
        fig_scatter = px.scatter(
            df, 
            x="score", 
            y="num_comments", 
            size="bubble_size", 
            color="flair_group",
            hover_data={'title': True, 'compound_score': True, 'bubble_size': False},
            title="Upvotes vs. Comment Density",
            template="plotly_dark",
            labels={'score': 'Upvotes', 'comms_num': 'Number of Comments', 'flair_group': 'Category'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- PAGE 4: SUPPORTIVE PULSE ---
    elif page == "Supportive Pulse":
        st.title("💓 Supportive Pulse")
        supportive_df = df[df['flair_group'] == 'Supportive/Crisis']
        
        if not supportive_df.empty:
            l_col, r_col = st.columns(2)
            with l_col:
                avg_support = supportive_df.groupby('flair')['compound_score'].mean().sort_values().reset_index()
                fig_bar = px.bar(avg_support, x='compound_score', y='flair', orientation='h', 
                                 color='compound_score', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_bar, use_container_width=True)
            with r_col:
                text = " ".join(supportive_df['combined_text'].astype(str))
                wc = WordCloud(background_color="#0e1117", max_words=100, width=800, height=500).generate(text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                st.pyplot(fig_wc)

if __name__ == "__main__":
    main()
