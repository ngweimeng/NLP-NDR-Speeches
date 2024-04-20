import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.ticker as ticker
import numpy as np

# Page configuration
st.set_page_config(page_title="NLP - PM Lee's Speeches", layout='wide', page_icon='🇸🇬')

# Column setup for image and main title
col1, col2 = st.columns([1, 3])

# Using the first column for the image
with col1:
    st.image("images/image1.jpeg")

# Using the second column for the main title
with col2:
    st.title("Leadership in Words - Unpacking PM Lee’s National Day Speeches with Natural Language Processing")

# About the project section
st.header('About the Project')
st.markdown("""
            In anticipation of Prime Minister Lee Hsien Loong’s handover of leadership to DPM Lawrence Wong in May 2024, this analysis has been crafted to honor his significant contributions to the nation of Singapore. The project dissects the series of PM Lee’s National Day Rally speeches, showcasing his leadership and influence through the lens of Natural Language Processing.
            
            As we approach this moment of transition, we express our sincere appreciation for PM Lee's years of unwavering dedication to Singapore’s growth and prosperity. 

            **Thank you Prime Minister Lee Hsien Loong Sir**!
""")

# Interactive slider for selecting a year
#title = 'Slide to select a year:'
#year = st.slider(title, min_value=2004, max_value=2023)
#st.write('You selected:', year)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/output.csv")

df = load_data()

# Vectorize all speeches at once with adjusted max_df for single grams
vectorizer_single = TfidfVectorizer(max_df=0.7)
X_single = vectorizer_single.fit_transform(df['processed_speech'])
feature_names_single = vectorizer_single.get_feature_names_out()
dense_single = X_single.todense()

# Vectorize all speeches for bigrams
vectorizer_bigram = TfidfVectorizer(max_df=0.7, ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(df['processed_speech'])
feature_names_bigram = vectorizer_bigram.get_feature_names_out()
dense_bigram = X_bigram.todense()

# Select a year for display and analysis
year_to_display = st.slider('Select a year to display NLP Analytics:', 
                            min_value=int(df['Year'].min()), 
                            max_value=int(df['Year'].max()), 
                            value=int(df['Year'].min()))

# Display word clouds and sentiment analysis
def display_wordclouds(dense, feature_names, title_suffix):
    idx = df.index[df['Year'] == year_to_display].tolist()[0]
    denselist = dense[idx].tolist()[0]
    df_tfidf = pd.DataFrame([denselist], columns=feature_names)

    word_weights = df_tfidf.sum(axis=0).sort_values(ascending=False).to_dict()

    wordcloud = WordCloud(
        width=400, height=200, background_color='black',
        colormap="Paired", max_font_size=150, max_words=20
    ).generate_from_frequencies(word_weights)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"{title_suffix} for Year {year_to_display}")
    return fig

### Display Text Summarization section
st.header(f"Text Analysis for {year_to_display}")

# Column setup for text summarization and word count analysis
col1, col2 = st.columns(2)

# Text Summarization column
with col1:
    st.subheader("Text Summarization")
    st.info("""
    The summaries below are generated using the OpenAI API, employing language modeling to distill key information from each speech.
    """)
    selected_summary = df[df['Year'] == year_to_display]['Summary'].iloc[0]
    st.write(selected_summary)

# Word Count Analysis and Word Count over the years column
with col2:
    st.subheader(f"Word Count Analysis")
    st.info("""
    The word count metric serves as a quantitative measure of the speech's extent and the comprehensiveness of the topics addressed by PM Lee within the given year.
    """)
    word_count = df[df['Year'] == year_to_display]['Word_Count'].iloc[0]
    st.write(f"The length of the speech for the year {year_to_display} is **{word_count} words**.")

    # Plot the changes in word count over the years
    st.subheader("Word Count Trend Over Years")
    fig, ax = plt.subplots()
    ax.plot(df['Year'], df['Word_Count'], marker='o', linestyle='-')
    ax.set_xlabel('Year')
    ax.set_ylabel('Word Count')
    ax.set_title('Word Count Trend Over Years')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure x-axis has only integer labels
    ax.grid(True)
    st.pyplot(fig)

### Word Cloud Section
st.header(f"Word Clouds for {year_to_display}")

# Explanation of the word cloud generation process
st.info("""
The word clouds generated here are visual representations of the most prominent words found in PM Lee's speeches for the selected year. They are created using a method known as TF-IDF (Term Frequency-Inverse Document Frequency) which evaluates how relevant a word is to a document in a collection of documents. This relevance is shown by the size of the word in the visualization.

Here's the process:

1. **Term Frequency (TF)**: We count how many times each word appears in the speech.
2. **Inverse Document Frequency (IDF)**: We calculate a score that diminishes the weight of terms that occur very frequently across the speech corpus and increases the weight of terms that occur rarely.
3. **TF-IDF**: The two scores are multiplied to determine the importance of each word within the speech for the selected year.
4. **Visualization**: The most important words are then displayed in the word cloud, with larger sizes indicating higher TF-IDF scores.

This technique helps us to extract key themes and terms that Prime Minister Lee emphasized during that year's address.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Single Grams")
    fig_single = display_wordclouds(dense_single, feature_names_single, "Single Gram")
    st.pyplot(fig_single)

with col2:
    st.subheader("Bi-grams")
    fig_bigram = display_wordclouds(dense_bigram, feature_names_bigram, "Bi-gram")
    st.pyplot(fig_bigram)

# Sentiment analysis
# Function to perform sentiment analysis across all speeches
def get_sentiment_over_time(df):
    df['sentiment_polarity'] = df['processed_speech'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['processed_speech'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df[['Year', 'sentiment_polarity', 'sentiment_subjectivity']]

# Calculate sentiment over time
sentiment_over_time = get_sentiment_over_time(df)

# Display sentiment analysis
st.header(f"Sentiment Analysis for {year_to_display}")

# Display detailed sentiment for the selected year
selected_speech_sentiment = sentiment_over_time[sentiment_over_time['Year'] == year_to_display]
polarity = selected_speech_sentiment['sentiment_polarity'].values[0]
subjectivity = selected_speech_sentiment['sentiment_subjectivity'].values[0]

# Explain sentiment terms
st.info("""
**Sentiment Polarity** indicates the positivity or negativity of the text. A score of -1 signifies extreme negativity, 0 neutrality, and 1 extreme positivity.

**Sentiment Subjectivity** quantifies the amount of personal opinion and factual information contained in the text. A score of 0 is very objective, and 1 is very subjective.
""")

st.markdown(f"**Sentiment Polarity (Negative to Positive):** `{polarity:.2f}` (Scale: -1 to 1)")
st.markdown(f"**Sentiment Subjectivity (Objective to Subjective):** `{subjectivity:.2f}` (Scale: 0 to 1)")

# Make sure 'Year' column is of integer type for proper x-axis labeling
df['Year'] = df['Year'].astype(int)
sentiment_over_time['Year'] = sentiment_over_time['Year'].astype(int)

# Plot sentiment polarity and subjectivity over time as line charts
st.subheader("Sentiment Trends Over Time")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Sentiment Polarity Over Time")
    fig1, ax1 = plt.subplots()
    ax1.plot(sentiment_over_time['Year'], sentiment_over_time['sentiment_polarity'], marker='o', color='blue')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Sentiment Polarity')
    ax1.set_ylim([-1, 1])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set x-axis major ticks to integer values
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    st.markdown("#### Sentiment Subjectivity Over Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(sentiment_over_time['Year'], sentiment_over_time['sentiment_subjectivity'], marker='o', color='orange')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Sentiment Subjectivity')
    ax2.set_ylim([0, 1])
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set x-axis major ticks to integer values
    ax2.grid(True)
    st.pyplot(fig2)
