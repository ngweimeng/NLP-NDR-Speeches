# Leadership in Words: Analyzing PM Lee's Speeches

## About This Project
This project, "Leadership in Words", focuses on dissecting the series of National Day Rally speeches delivered by Prime Minister Lee Hsien Loong, using advanced Natural Language Processing (NLP) techniques. The aim is to understand the evolving leadership themes and sentiments expressed over the years as Singapore navigates through various challenges and milestones.

See Streamlit app [here][https://pm-national-day-speeches.streamlit.app/]

### Rationale
The project was initiated in anticipation of PM Lee Hsien Loong’s handover of leadership, to explore his impact through his speeches at the National Day Rallies. By applying NLP, we aim to uncover deeper insights into the thematic and emotional undertones of his leadership narrative.

## Data Collection
Data for this project comprises speeches delivered by PM Lee, sourced from the official Prime Minister's Office website. Speeches were systematically scraped using Python libraries `requests` and `BeautifulSoup`. The data spans from 2004 to 2023, providing a rich corpus for longitudinal analysis.

### Data Preprocessing
The speeches were preprocessed to enhance the NLP analysis. This involved:
- Removing stopwords pertinent to Singapore's context.
- Lemmatization to reduce words to their base or dictionary form.
- Removal of special characters and numbers to focus on textual data.

## NLP Analysis
Several NLP techniques were employed to analyze the speeches:

### Sentiment Analysis
Using `TextBlob`, we evaluated the sentiment polarity and subjectivity of each speech, aiming to capture the emotional and subjective content over the years.

### Topic Modeling
Non-negative Matrix Factorization (NMF) was used to extract key themes from the speeches. This method helps identify prevalent topics and their evolution over time.

### Word Clouds
Generated using `WordCloud` in Python, these visualizations highlight the most significant words from each year's speeches, providing intuitive insights into the focal themes.

### TF-IDF Analysis
To quantitatively measure the importance of words within the speeches across the corpus, TF-IDF vectorization was utilized, highlighting words that are frequent in a speech but not across all speeches.

## Data Source and Disclaimer

The analysis uses data collected via web scraping from the [Prime Minister's Office website](https://www.pmo.gov.sg/). The dataset comprises speeches delivered by Prime Minister Lee Hsien Loong at his National Day Rallies. **Disclaimer:** The analysis and visualizations are for academic purposes only and do not represent political opinions or endorsements. Interpretations are strictly computational and do not reflect personal views.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
