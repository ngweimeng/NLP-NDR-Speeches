import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Page configuration
st.set_page_config(page_title="NLP - PM Lee's Speeches", layout='wide', page_icon='🇸🇬')

# Sidebar with information about the data source and disclaimer

st.sidebar.markdown("""
### Data Source
This analysis uses data collected via web scraping from the [Prime Minister's Office website](https://www.pmo.gov.sg/). The dataset comprises speeches delivered by Prime Minister Lee Hsien Loong at his National Day Rallies.
""")

# Using a warning to highlight the disclaimer
st.sidebar.warning("""
**Disclaimer:** The analysis and visualizations are for academic purposes only and do not represent political opinions or endorsements. Interpretations are strictly computational and do not reflect personal views.
""")

st.sidebar.markdown("""
### GitHub Repository
For more details on the project or to contribute, please visit the [GitHub repository](https://github.com/ngweimeng/NLP-NDR-Speeches).
""")

# Column setup for image and main title
col1, col2 = st.columns([1, 1.2])

# Using the first column for the image
with col1:
    st.image("images/image1.jpeg")

# Using the second column for the main title
with col2:
    st.title("Leadership in Words - Unpacking PM Lee’s National Day Speeches with Natural Language Processing")

# About the project section
st.markdown("## About This Project")
st.markdown("""
            In anticipation of Prime Minister Lee Hsien Loong’s handover of leadership to DPM Lawrence Wong in May 2024, this project has been crafted to honor his significant contributions to Singapore. The project dissects the series of PM Lee’s National Day Rally speeches (2004-2023), showcasing his leadership and influence through the lens of Natural Language Processing.
            
            As we approach this moment of transition, we express our sincere appreciation for PM Lee's years of unwavering dedication to Singapore’s growth and prosperity. 

            **Thank you Prime Minister Lee Hsien Loong Sir**!
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/output.csv")

df = load_data()

# Year and Analysis Type selection
col_year, col_analysis = st.columns([1, 2])
with col_year:
    year_to_display = st.selectbox(
        'Select Year:',
        options=sorted(df['Year'].unique()),  # Ensure years are sorted
        index=len(df['Year'].unique()) - 1  # Default to the most recent year
    )

# Display warning for 2020
if year_to_display == 2020:
    st.warning(
        "The 2020 rally was cancelled due to the COVID-19 pandemic. Instead, PM Lee's National Day Message is analyzed. "
        "Please note, this may cause deviations from typical data patterns."
    )

with col_analysis:
    analysis_type = st.selectbox(
        'Select Analysis:',
        options=['Speech Summary','Word Count','Word Cloud', 'Sentiment Analysis','Topic Modeling'],
        index=0  # Default to the first analysis type
    )

# Vectorize all speeches for single grams
vectorizer_single = TfidfVectorizer(max_df=0.7)
X_single = vectorizer_single.fit_transform(df['processed_speech'])
feature_names_single = vectorizer_single.get_feature_names_out()
dense_single = X_single.todense()

# Vectorize all speeches for bigrams
vectorizer_bigram = TfidfVectorizer(max_df=0.7, ngram_range=(2, 2))
X_bigram = vectorizer_bigram.fit_transform(df['processed_speech'])
feature_names_bigram = vectorizer_bigram.get_feature_names_out()
dense_bigram = X_bigram.todense()

# Function to display word cloud
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

# Function to perform sentiment analysis across all speeches
def get_sentiment_over_time(df):
    df['sentiment_polarity'] = df['processed_speech'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['processed_speech'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df[['Year', 'sentiment_polarity', 'sentiment_subjectivity']]

# Calculate sentiment over time
sentiment_over_time = get_sentiment_over_time(df)

### Speech Summary section
if analysis_type == 'Speech Summary':
    st.header(f"Speech Summary for {year_to_display}")
    st.info("""
    **Understanding Summary Generation:**

    The summaries are derived from an automated analysis using the OpenAI API, which applies natural language processing techniques to distill core information from PM Lee's speeches. This method systematically summarizes key content, providing a concise overview for better readability.
    """)
    # Fetch the URL for the full speech
    speech_url = df[df['Year'] == year_to_display]['URL'].iloc[0]
    st.markdown(f"[Click here for the full speech]({speech_url})", unsafe_allow_html=True)

    # Fetch Speech Summary
    selected_summary = df[df['Year'] == year_to_display]['Summary'].iloc[0]
    st.write(selected_summary)

### Word Count section
elif analysis_type == 'Word Count':
    st.header(f"Word Count for {year_to_display}")
    st.info("""
    **Understanding Word Count Metrics:**
    
    The word count analysis provides insights into the length and detail of each speech by PM Lee. Analyzing word count helps identify trends in speech length over time, which can reflect changes in policy focus, communication strategy, or event context. For instance, longer speeches might indicate more comprehensive coverage of topics, significant announcements, or detailed explanations of complex issues. Conversely, shorter speeches might suggest a more focused or concise communication approach.
    
    This analysis can be particularly useful for understanding shifts in governance and communication strategy across different periods or in response to specific events.
    """)
    # Setup columns for metric and chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        current_word_count = df[df['Year'] == year_to_display]['Word_Count'].iloc[0]
        previous_year = year_to_display - 1
        if previous_year in df['Year'].values:
            previous_word_count = df[df['Year'] == previous_year]['Word_Count'].iloc[0]
            delta = current_word_count - previous_word_count
        else:
            delta = None  # No data for the previous year

        # Use Streamlit's metric for a cleaner display with delta
        if delta is not None:
            st.metric(label=f"Word Count for {year_to_display}", value=f"{current_word_count} words", delta=f"{delta} words")
        else:
            st.metric(label=f"Word Count for {year_to_display}", value=f"{current_word_count} words", delta="No previous data")
    
    with col2:
        # Plot Word Count Trend using Plotly for an interactive chart
        fig = px.line(
            df, 
            x='Year', 
            y='Word_Count', 
            markers=True,
            title='Word Count Trend Over Years'
        )
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=sorted(df['Year'].unique())),
            yaxis_title='Word Count',
            xaxis_title='Year'
        )
        st.plotly_chart(fig, use_container_width=True)

### Word Cloud Section
elif analysis_type == 'Word Cloud':
    st.header(f"Word Clouds for {year_to_display}")

    # Explanation of the word cloud generation process
    st.info("""
    **Understanding Word Clouds:**
            
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

### Sentiment Analysis Section
elif analysis_type == 'Sentiment Analysis':

    st.header(f"Sentiment Analysis for {year_to_display}")

    # Explain sentiment terms
    st.info("""
**Understanding Sentiment Analysis:**

Sentiment Analysis is employed to gauge the emotional tone behind a series of speeches by PM Lee, offering insights into the mood and communicative intent of his addresses during various national events.

- **Sentiment Polarity**: This metric calculates the overall sentiment of the text from highly negative (-1), neutral (0), to highly positive (+1), based on the use of language and phrasing. The analysis is done through natural language processing algorithms that evaluate word and phrase choices to determine sentiment.
- **Sentiment Subjectivity**: This measure assesses whether the text is more subjective (opinionated) or objective (factual), with scores ranging from 0 (fully objective) to 1 (highly subjective). This is crucial for distinguishing between factual reporting and personal opinion in speeches.

Understanding sentiment polarity helps stakeholders gauge the optimism or concern in the leadership's messaging, which can be crucial for predicting policy directions or public sentiment. Knowing the level of subjectivity in speeches can also aid communications teams in aligning future speeches with desired tones, whether aiming for more factual presentations or engaging narratives that resonate on a personal level.
""")

    # Fetching sentiment data for the selected year
    selected_speech_sentiment = sentiment_over_time[sentiment_over_time['Year'] == year_to_display]
    polarity = selected_speech_sentiment['sentiment_polarity'].values[0]
    subjectivity = selected_speech_sentiment['sentiment_subjectivity'].values[0]

    # Calculate delta values
    previous_year = year_to_display - 1
    if previous_year in sentiment_over_time['Year'].values:
        previous_sentiment = sentiment_over_time[sentiment_over_time['Year'] == previous_year]
        previous_polarity = previous_sentiment['sentiment_polarity'].values[0]
        previous_subjectivity = previous_sentiment['sentiment_subjectivity'].values[0]
        delta_polarity = polarity - previous_polarity
        delta_subjectivity = subjectivity - previous_subjectivity
        delta_polarity_text = f"{delta_polarity:+.2f}"
        delta_subjectivity_text = f"{delta_subjectivity:+.2f}"
    else:
        delta_polarity_text = "N/A"
        delta_subjectivity_text = "N/A"

    # Display sentiment metrics with deltas
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sentiment Polarity (Negative to Positive)", f"{polarity:.2f}", delta=delta_polarity_text)
    with col2:
        st.metric("Sentiment Subjectivity (Objective to Subjective)", f"{subjectivity:.2f}", delta=delta_subjectivity_text)

    # Prepare data for Plotly
    import plotly.graph_objs as go

    # Prepare data for Plotly
    df_polarity = sentiment_over_time[['Year', 'sentiment_polarity']]
    df_subjectivity = sentiment_over_time[['Year', 'sentiment_subjectivity']]

    # Sentiment Polarity Over Time Plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_polarity['Year'], y=df_polarity['sentiment_polarity'], mode='lines+markers', name='Polarity'))
    fig1.update_layout(
        title="Sentiment Polarity Over Time",
        xaxis_title='Year',
        yaxis_title='Sentiment Polarity',
        xaxis=dict(tickmode='array', tickvals=sorted(df['Year'].unique())),  
        yaxis=dict(range=[-0.2, 0.2])  
    )

    # Sentiment Subjectivity Over Time Plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_subjectivity['Year'], y=df_subjectivity['sentiment_subjectivity'], mode='lines+markers', name='Subjectivity'))
    fig2.update_layout(
        title="Sentiment Subjectivity Over Time",
        xaxis_title='Year',
        yaxis_title='Sentiment Subjectivity',
        xaxis=dict(tickmode='array', tickvals=sorted(df['Year'].unique())),  
        yaxis=dict(range=[0.2, 0.6])  
    )
    # Display plots in columns
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

### Topic Modeling Section
elif analysis_type == 'Topic Modeling':
    st.header(f"Topic Modeling for {year_to_display}")
    st.info("""
    **Understanding Topic Modeling with NMF:**

    We utilize Non-negative Matrix Factorization (NMF), a group of algorithms in multivariate analysis and linear algebra, to analyze speeches. NMF identifies topics by decomposing the large document-term matrix, derived from the transcripts, into two meaningful smaller matrices - one representing the relationship between documents and topics, the other between topics and terms. Read more about [NMF here](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html).

    Through trial and error, we chose to focus on five topics because this number made the themes easiest to understand and most meaningful. Here's a peek into what each topic covers and their co-occuring words/terms:

    - **Global Engagement**: This includes discussions on how Singapore interacts with the rest of the world, including trade and technological advancements. *(International relations, trade, technological competition...)*
    - **National Crisis Response**: Here, we see strategies for tackling emergencies that affect the entire country, like public health scares or legal issues. *(Strategies for public health emergencies, legal frameworks, operational logistics...)*
    - **Community Development**: This theme is about building a better society, like improving neighborhoods, creating housing policies, and supporting community members, especially the elderly. *(Urban planning, housing policy, community welfare, support for the elderly...)*
    - **Health & Education**: These discussions revolve around keeping the nation healthy and well-educated, from preventing diseases to supporting families. *(Public health, preventive healthcare education, family support programs...)*
    - **Cultural Identity**: This is all about what makes Singapore unique, including its diverse social makeup, religious practices, and sense of unity. *(Social fabric, religious harmony, multiculturalism, national unity...)*

    The heatmap below visually tracks how these topics have come into and out of focus over the years, shedding light on what matters most to Singapore at different times.
""")

    # Topic names mapping
    topic_names = {
        "Topic 1": "Global Engagement",
        "Topic 2": "National Crisis Response",
        "Topic 3": "Community Development",
        "Topic 4": "Health & Education",
        "Topic 5": "Cultural Identity"
    }

    # Perform TF-IDF vectorization 
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.7, stop_words='english')
    dtm = vectorizer.fit_transform(df['processed_speech'])

    nmf_model = NMF(n_components=5, random_state=42)
    nmf_model.fit(dtm)

    document_index = df.index[df['Year'] == year_to_display].tolist()[0]
    document_topics = nmf_model.transform(dtm[document_index])

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for index, (col, (topic_key, topic_name)) in enumerate(zip(columns, topic_names.items())):
        current_year_topic_weight = document_topics.flatten()[index]
        previous_year = year_to_display - 1

        # Check if there is data for the previous year
        if previous_year in df['Year'].values:
            previous_document_index = df.index[df['Year'] == previous_year].tolist()[0]
            previous_year_topics = nmf_model.transform(dtm[previous_document_index])
            previous_year_topic_weight = previous_year_topics.flatten()[index]
            delta = current_year_topic_weight - previous_year_topic_weight
        else:
            delta = None

        delta_color = "inverse" if (delta is not None and delta < 0) else "normal"

        # Display the metric for the current year
        with col:
            st.metric(label=topic_name,
                    value=f"{current_year_topic_weight:.2f}",
                    delta=f"{delta:.2f}" if delta is not None else "N/A",
                    delta_color=delta_color)
            
    # Create a DataFrame to store topic distribution for each document
    topic_distributions = nmf_model.transform(dtm)  # This gets the topic distribution for each document
    topic_df = pd.DataFrame(topic_distributions, columns=[topic_names[f"Topic {i+1}"] for i in range(nmf_model.n_components)])
    topic_df['Year'] = df['Year']

    # Create a DataFrame suitable for the heatmap
    heatmap_df = topic_df.groupby('Year').mean().T
    heatmap_df.columns = heatmap_df.columns.astype(str)  # Convert the year column names to string if they are not already

    # Create the heatmap using Plotly
    fig = px.imshow(
    heatmap_df.T,  # Transpose the DataFrame
    labels=dict(x="Topic", y="Year", color="Relevance"),
    x=heatmap_df.index,  # Topics are now on the x-axis
    y=heatmap_df.columns,  # Years are now on the y-axis
    aspect="auto",
    color_continuous_scale='YlGnBu'
    )

    # Rotate the x-axis labels (topics) for better readability
    fig.update_xaxes(tickangle=-45)

    # Update the y-axis ticks to ensure all years are displayed
    fig.update_yaxes(
        tickmode='linear',
        tick0=heatmap_df.columns.min(),
        dtick=1  # Display a tick for every year increment
    )

    # Add titles and adjust layout
    fig.update_layout(
        title_text='Trend of Topics Over the Years', 
        title_x=0.5, 
        autosize=False, 
        margin=dict(t=50, l=10, r=10, b=10)
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)