# Data manipulation and analysis
import pandas as pd
from collections import Counter

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Display utilities
from IPython.display import display

# Natural Language Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams

# Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model

# Named Entity Recognition
import spacy
from collections import Counter

# Sentiment Analysis
from textblob import TextBlob

# Text Complexity Analysis
import re
from math import ceil

# Word Cloud
from wordcloud import WordCloud

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and display the first few rows.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Display the first few rows of the DataFrame
    display(df.head())

    return df

def info(dataframe: pd.DataFrame) -> None:
    """
    Display detailed information about each column in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to display information about.

    Returns:
        None
    """
    for column in dataframe.columns:
        print(f"\nColumn: {column}")
        print("-" * 40)
        
        # Display data type
        print(f"Data Type: {dataframe[column].dtype}")
        
        # Display non-null count and percentage
        non_null_count = dataframe[column].count()
        non_null_percentage = (non_null_count / len(dataframe)) * 100
        print(f"Non-Null Count: {non_null_count} ({non_null_percentage:.2f}%)")
        
        # Display unique value count
        unique_count = dataframe[column].nunique()
        print(f"Unique Values: {unique_count}")
        
        # Display basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            print("Basic Statistics:")
            print(dataframe[column].describe().to_string())
        
        # Display most common values for non-numeric columns
        else:
            print("Most Common Values:")
            print(dataframe[column].value_counts().head().to_string())
        
        print("\n")

    return None

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the categories in the DataFrame. Any category that is not in the predefined list
    is considered 'physics'.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'category' column.

    Returns:
        pd.DataFrame: The DataFrame with cleaned categories.
    """
    valid_categories = {
        'physics', 'mathematics', 'computer science', 'quantitative biology',
        'quantitative finance', 'statistics', 'electrical engineering and systems science',
        'economics'
    }
    
    df['category'] = df['category'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['category'] = df['category'].apply(lambda x: x if x in valid_categories else 'physics')
    
    # Remove duplicates based on title, summary, or comment
    df = df.drop_duplicates(subset=['title', 'summary', 'comment'], keep='first')
    
    return df


def create_basic_plot(figsize=(10, 6)):
    """
    Create and return a basic plot figure and axes using seaborn style.

    Args:
        figsize (tuple): The figure size (width, height) in inches. Default is (10, 6).

    Returns:
        tuple: A tuple containing the figure and axes objects.
    """
    # Set the seaborn style
    sns.set_style("whitegrid")

    # Create a new figure with specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Set a subtle grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    return fig, ax


def plot_category_distribution(df: pd.DataFrame) -> None:
    """
    Display the distribution of categories in a bar plot using seaborn.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'category' column.

    Returns:
        None
    """
    # Count the occurrences of each category
    category_counts = df['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    # Create a default plot
    plt.figure(figsize=(14, 8))

    # Create a bar plot of category counts using seaborn with default colors
    sns.barplot(x='category', y='count', data=category_counts)

    # Set the title and labels for the plot
    plt.title('Distribution of Categories', fontsize=20, fontweight='bold')
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Rotate x-axis labels for better readability and adjust their position
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Add count labels on top of each bar
    for i, v in enumerate(category_counts['count']):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()

    return None

def plot_text_analysis_distributions(df: pd.DataFrame) -> None:
    """
    Plot the distributions of title, summary, and comment lengths in the DataFrame using seaborn.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.

    Returns:
        None
    """
    # Calculate text lengths
    df['title_length'] = df['title'].str.len()
    df['summary_length'] = df['summary'].str.len()
    df['comment_length'] = df['comment'].str.len()

    # Plot title length distribution
    plot_distribution(df, 'title_length', 'Distribution of Title Lengths')

    # Plot summary length distribution
    plot_distribution(df, 'summary_length', 'Distribution of Summary Lengths')

    # Plot comment length distribution
    plot_distribution(df, 'comment_length', 'Distribution of Comment Lengths')

    return None

def plot_distribution(data: pd.DataFrame, column: str, title: str) -> None:
    """
    Create and display a single distribution plot.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to plot.
        column (str): The name of the column to plot.
        title (str): The title of the plot.

    Returns:
        None
    """
    # Create a basic plot
    fig, ax = create_basic_plot(figsize=(12, 6))

    # Create a histogram plot with kernel density estimation
    sns.histplot(data=data, x=column, kde=True, ax=ax)

    # Set the title and labels for the plot with custom fonts
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{title.split()[-2]} Length (characters)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()

    return None

def plot_text_analysis_boxplots(df: pd.DataFrame) -> None:
    """
    Create and display separate box plots for title, summary, and comment lengths,
    showing key statistical values including whisker values.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        None
    """
    # Create separate plots for title, summary, and comment lengths
    for column in ['title_length', 'summary_length', 'comment_length']:
        # Create a basic plot
        fig, ax = create_basic_plot(figsize=(12, 8))

        # Create a box plot for the current column
        sns.boxplot(data=df, y=column, ax=ax)

        # Calculate key statistical values
        min_val = df[column].min()
        q1 = df[column].quantile(0.25)
        median = df[column].median()
        q3 = df[column].quantile(0.75)
        max_val = df[column].max()

        # Calculate whisker values
        iqr = q3 - q1
        lower_whisker = max(min_val, q1 - 1.5 * iqr)
        upper_whisker = min(max_val, q3 + 1.5 * iqr)

        # Add text annotations for key statistical values including whiskers
        ax.text(0.05, 0.95, f'Min: {min_val:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.90, f'Lower Whisker: {lower_whisker:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.85, f'Q1: {q1:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.80, f'Median: {median:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.75, f'Q3: {q3:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.70, f'Upper Whisker: {upper_whisker:.0f}', transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.65, f'Max: {max_val:.0f}', transform=ax.transAxes, verticalalignment='top')

        # Set the title and labels for the plot with custom fonts
        ax.set_title(f'Distribution of {column.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        ax.set_ylabel('Length (characters)', fontsize=12)

        # Adjust the layout to prevent clipping of labels
        plt.tight_layout()

        # Display the plot
        plt.show()

    return None


def plot_text_length_analysis_distribution_by_category(df: pd.DataFrame) -> None:
    """
    Create and display violin plots for title, summary, and comment lengths,
    showing the distribution per category.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        None
    """
    # Create separate plots for title, summary, and comment lengths
    for column in ['title_length', 'summary_length', 'comment_length']:
        # Create a basic plot
        fig, ax = create_basic_plot(figsize=(14, 8))

        # Create a violin plot for the current column, grouped by category
        sns.violinplot(data=df, x='category', y=column, ax=ax)

        # Set the title and labels for the plot with custom fonts
        ax.set_title(f'Distribution of {column.replace("_", " ").title()} by Category', fontsize=16, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Length (characters)', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust the layout to prevent clipping of labels
        plt.tight_layout()

        # Display the plot
        plt.show()

    return None

def plot_text_length_analysis_average_by_category(df: pd.DataFrame) -> None:
    """
    Create and display bar plots for average title, summary, and comment lengths by category.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.

    Returns:
        None
    """
    # Calculate average lengths for each category
    avg_lengths = df.groupby('category')[['title_length', 'summary_length', 'comment_length']].mean().reset_index()

    # Create separate plots for title, summary, and comment lengths
    for column in ['title_length', 'summary_length', 'comment_length']:
        # Create a basic plot
        fig, ax = create_basic_plot(figsize=(14, 8))

        # Create a bar plot for the current column
        sns.barplot(data=avg_lengths, x='category', y=column, ax=ax)

        # Set the title and labels for the plot with custom fonts
        ax.set_title(f'Average {column.replace("_", " ").title()} by Category', fontsize=16, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Average Length (characters)', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of each bar
        for i, v in enumerate(avg_lengths[column]):
            ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        # Adjust the layout to prevent clipping of labels
        plt.tight_layout()

        # Display the plot
        plt.show()

    return None

def get_word_freq(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return Counter(words)

def plot_word_frequency_analysis(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Create and display bar plots for word frequency analysis of title, summary, and comment columns,
    excluding stop words.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        top_n (int): Number of top words to display. Default is 20.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']

    for column in columns:
        # Combine all text in the column
        all_text = ' '.join(df[column].dropna().astype(str))
        
        # Get word frequencies
        word_freq = get_word_freq(all_text)
        
        # Get top N words
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        # Create a basic plot
        fig, ax = create_basic_plot(figsize=(12, 8))
        
        # Create a bar plot
        sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), ax=ax)
        
        # Set the title and labels
        ax.set_title(f'Top {top_n} Most Common Words in {str(column).title()} (Excluding Stop Words)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Words', fontsize=12)
        
        # Add value labels on the bars
        for i, v in enumerate(top_words.values()):
            ax.text(v, i, f' {v}', va='center', fontweight='bold')
        
        # Adjust the layout
        plt.tight_layout()
        
        # Display the plot
        plt.show()

    return None

def plot_word_frequency_by_category(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Plot word frequency analysis for each column by category.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        top_n (int): Number of top words to display for each category. Default is 20.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']
    categories = df['category'].unique()

    for column in columns:
        for category in categories:
            # Filter data for the current category
            category_data = df[df['category'] == category]
            
            # Combine all text in the column for the current category
            all_text = ' '.join(category_data[column].dropna().astype(str))
            
            # Get word frequencies
            word_freq = get_word_freq(all_text)
            
            # Get top N words
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
            
            # Create a basic plot
            fig, ax = create_basic_plot(figsize=(12, 8))
            
            # Create a bar plot
            sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), ax=ax)
            
            # Set the title and labels
            ax.set_title(f'Top {top_n} Most Common Words in {str(column).title()} for {str(category).title()} (Excluding Stop Words)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_ylabel('Words', fontsize=12)
            
            # Add value labels on the bars
            for i, v in enumerate(top_words.values()):
                ax.text(v, i, f' {v}', va='center', fontweight='bold')
            
            # Adjust the layout
            plt.tight_layout()
            
            # Display the plot
            plt.show()

    return None

def get_top_ngrams(text: str, n: int, top_k: int = 20) -> list:
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum()]
    n_grams = ngrams(tokens, n)
    n_gram_counts = Counter(n_grams)
    return n_gram_counts.most_common(top_k)

def plot_ngrams(ngrams: list, title: str) -> None:
    fig, ax = create_basic_plot(figsize=(15, 10))
    words, counts = zip(*ngrams)
    ax.barh([' '.join(w) for w in words], counts)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("N-grams", fontsize=12)
    ax.invert_yaxis()
    for i, v in enumerate(counts):
        ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_ngram_analysis(df: pd.DataFrame, n_range: range = range(2, 6), top_k: int = 20) -> None:
    """
    Perform n-gram analysis on the titles, summaries, and comments of the DataFrame and plot the results separately.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.
        n_range (range): The range of n-grams to analyze. Default is 2 to 5.
        top_k (int): The number of top n-grams to display. Default is 20.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']
    
    for column in columns:
        all_text = ' '.join(df[column].dropna().astype(str).str.lower())
        top_ngrams = {n: get_top_ngrams(all_text, n, top_k) for n in n_range}

        for n, ngrams in top_ngrams.items():
            plot_ngrams(ngrams, f"Top {top_k} {n}-grams in {column.capitalize()}")

    plt.close('all')  # Close all figures to free up memory

def perform_lda_topic_modeling(df: pd.DataFrame, num_topics: int = 5, num_words: int = 10) -> None:
    """
    Perform Topic Modeling using Latent Dirichlet Allocation (LDA) on title, summary, and comment columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        num_topics (int): The number of topics to extract. Default is 5.
        num_words (int): The number of words to display for each topic. Default is 10.

    Returns:
        None
    """

    columns = ['title', 'summary', 'comment']

    for column in columns:
        print(f"\nPerforming LDA Topic Modeling on '{column}' column:")
        
        # Prepare the data
        text_data = df[column].dropna().astype(str).tolist()

        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(text_data)

        # Create and fit the LDA model
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_output = lda_model.fit_transform(doc_term_matrix)

        # Print the top words for each topic
        print(f"Top {num_words} words for each of the {num_topics} topics in '{column}':")
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
            print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
            print()

        # Visualize the topics
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer)
        display(vis)

        # Assign dominant topic to each document
        dominant_topics = lda_output.argmax(axis=1)

        # Plot distribution of topics
        plt.figure(figsize=(10, 6))
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        ax = topic_counts.plot(kind='bar')
        plt.title(f'Distribution of Dominant Topics in {column.capitalize()}')
        plt.xlabel('Topic')
        plt.ylabel('Number of Documents')

        # Add value labels on the bars
        for i, v in enumerate(topic_counts):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

        # Expand y-axis limit by 10%
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.1)

        plt.tight_layout()
        plt.show()

    return None

def perform_lda_topic_modeling_by_category(df: pd.DataFrame, num_topics: int = 5, num_words: int = 10) -> None:
    """
    Perform Topic Modeling using Latent Dirichlet Allocation (LDA) on title, summary, and comment columns of the DataFrame,
    grouped by category.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data and 'category' column.
        num_topics (int): The number of topics to extract for each category. Default is 5.
        num_words (int): The number of words to display for each topic. Default is 10.

    Returns:
        None
    """

    columns = ['title', 'summary', 'comment']

    for category in df['category'].unique():
        print(f"\nPerforming LDA Topic Modeling for category: {category}")
        category_df = df[df['category'] == category]

        for column in columns:
            print(f"\nAnalyzing '{column}' column:")
            
            # Prepare the data
            text_data = category_df[column].dropna().astype(str).tolist()

            # Create document-term matrix
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(text_data)

            # Create and fit the LDA model
            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_output = lda_model.fit_transform(doc_term_matrix)

            # Print the top words for each topic
            print(f"Top {num_words} words for each of the {num_topics} topics in '{column}':")
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
                print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
                print()

            # Visualize the topics
            pyLDAvis.enable_notebook()
            vis = pyLDAvis.lda_model.prepare(lda_model, doc_term_matrix, vectorizer)
            display(vis)

            # Assign dominant topic to each document
            dominant_topics = lda_output.argmax(axis=1)

            # Plot distribution of topics
            plt.figure(figsize=(10, 6))
            topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
            ax = topic_counts.plot(kind='bar')
            plt.title(f'Distribution of Dominant Topics in {column.capitalize()} for {category}')
            plt.xlabel('Topic')
            plt.ylabel('Number of Documents')

            # Add value labels on the bars
            for i, v in enumerate(topic_counts):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')

            # Expand y-axis limit by 10%
            y_max = ax.get_ylim()[1]
            ax.set_ylim(0, y_max * 1.1)

            plt.tight_layout()
            plt.show()

    return None

def perform_named_entity_recognition(df: pd.DataFrame) -> None:
    """
    Perform Named Entity Recognition (NER) on the title, summary, and comment columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.

    Returns:
        None
    """

    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")

    columns = ['title', 'summary', 'comment']

    for column in columns:
        print(f"\nPerforming Named Entity Recognition on '{column}' column:")

        # Function to process text and extract named entities
        def extract_entities(text):
            doc = nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]

        # Apply NER to the specified column
        entities = df[column].dropna().astype(str).apply(extract_entities)

        # Flatten the list of entities
        all_entities = [entity for sublist in entities for entity in sublist]

        # Count the occurrences of each entity
        entity_counts = Counter(all_entities)

        # Display the most common entities
        print("\nMost common named entities:")
        for (entity, label), count in entity_counts.most_common(20):
            print(f"{entity} ({label}): {count}")

        # Visualize entity distribution
        entity_types = [label for _, label in all_entities]
        entity_type_counts = Counter(entity_types)

        fig, ax = create_basic_plot()
        sns.barplot(x=list(entity_type_counts.keys()), y=list(entity_type_counts.values()), ax=ax)
        ax.set_title(f"Distribution of Named Entity Types in '{column}'")
        ax.set_xlabel("Entity Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    return None


def perform_ner_by_category(df: pd.DataFrame) -> None:
    """
    Perform Named Entity Recognition (NER) on the text data, grouped by category.

    Args:
        df (pd.DataFrame): The DataFrame containing the text data and 'category' column.

    Returns:
        None
    """
    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")

    columns = ['title', 'summary', 'comment']
    categories = df['category'].unique()

    for category in categories:
        print(f"\nPerforming Named Entity Recognition for category: {category}")
        
        category_df = df[df['category'] == category]
        
        for column in columns:
            print(f"\nAnalyzing '{column}' column:")

            # Function to process text and extract named entities
            def extract_entities(text):
                doc = nlp(text)
                return [(ent.text, ent.label_) for ent in doc.ents]

            # Apply NER to the specified column for the current category
            entities = category_df[column].dropna().astype(str).apply(extract_entities)

            # Flatten the list of entities
            all_entities = [entity for sublist in entities for entity in sublist]

            # Count the occurrences of each entity
            entity_counts = Counter(all_entities)

            # Display the most common entities
            print("\nMost common named entities:")
            for (entity, label), count in entity_counts.most_common(10):
                print(f"{entity} ({label}): {count}")

            # Visualize entity distribution
            entity_types = [label for _, label in all_entities]
            entity_type_counts = Counter(entity_types)

            fig, ax = create_basic_plot()
            sns.barplot(x=list(entity_type_counts.keys()), y=list(entity_type_counts.values()), ax=ax)
            ax.set_title(f"Distribution of Named Entity Types in '{column}' for {category}")
            ax.set_xlabel("Entity Type")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    return None

def perform_sentiment_analysis(df: pd.DataFrame) -> None:
    """
    Perform sentiment analysis on the title, summary, and comment columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.

    Returns:
        None
    """
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity

    # Calculate sentiment for titles, summaries, and comments
    df['title_sentiment'] = df['title'].apply(get_sentiment)
    df['summary_sentiment'] = df['summary'].apply(get_sentiment)
    df['comment_sentiment'] = df['comment'].apply(get_sentiment)

    # Calculate average sentiment for each category
    category_sentiment = df.groupby('category')[['title_sentiment', 'summary_sentiment', 'comment_sentiment']].mean()

    # Distribution of sentiment scores for each column
    columns = ['title_sentiment', 'summary_sentiment', 'comment_sentiment']
    for column in columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Distribution of {column.split("_")[0].capitalize()} Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    # Plotting average sentiment by category for each column
    for column in columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_sentiment.index, y=column, data=category_sentiment)
        plt.title(f'Average {column.split("_")[0].capitalize()} Sentiment by Category')
        plt.xlabel('Category')
        plt.ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return None

def calculate_ari(text):
    if not isinstance(text, str) or not text.strip():
        return None
    words = len(re.findall(r'\w+', text))
    sentences = len(re.findall(r'\w+[.!?]', text)) or 1  # Ensure at least 1 sentence
    characters = len(re.findall(r'\S', text))
    if words == 0 or sentences == 0:
        return None
    ari = ceil(4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43)
    return max(1, min(ari, 14))  # Clamp ARI between 1 and 14

def plot_text_complexity_analysis(df: pd.DataFrame) -> None:
    """
    Perform text complexity analysis using Automated Readability Index (ARI) on the title, summary, and comment columns of the DataFrame and plot the results.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']
    for column in columns:
        df[f'{column}_ari'] = df[column].apply(calculate_ari)

    for i, column in enumerate(columns):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='category', y=f'{column}_ari', data=df.dropna(subset=[f'{column}_ari']))
        plt.title(f'Text Complexity Analysis: {column.capitalize()} ARI by Category')
        plt.xlabel('Category')
        plt.ylabel('ARI Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Overall ARI distribution
    plt.figure(figsize=(12, 6))
    for column in columns:
        sns.kdeplot(data=df[f'{column}_ari'].dropna(), label=column.capitalize())
    plt.title('Overall ARI Score Distribution')
    plt.xlabel('ARI Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None

def plot_text_complexity_by_category(df: pd.DataFrame) -> None:
    """
    Plot text complexity analysis using Automated Readability Index (ARI) for each category.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns and their ARI scores.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']
    categories = df['category'].unique()

    for category in categories:
        category_df = df[df['category'] == category]
        
        for column in columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=f'{column}_ari', data=category_df.dropna(subset=[f'{column}_ari']))
            plt.title(f'Text Complexity Analysis: {category} - {column.capitalize()} ARI')
            plt.ylabel('ARI Score')
            
            # Calculate and display mean ARI score
            mean_ari = category_df[f'{column}_ari'].mean()
            if not pd.isna(mean_ari):
                plt.axhline(mean_ari, color='r', linestyle='--')
                plt.text(0.05, 0.95, f'Mean ARI: {mean_ari:.2f}', 
                         transform=plt.gca().transAxes, 
                         verticalalignment='top',
                         color='r')

            plt.tight_layout()
            plt.show()

    # Overall ARI distribution by category
    for column in columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='category', y=f'{column}_ari', data=df.dropna(subset=[f'{column}_ari']))
        plt.title(f'Overall ARI Score Distribution by Category: {column.capitalize()}')
        plt.xlabel('Category')
        plt.ylabel('ARI Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return None

def plot_word_cloud(df: pd.DataFrame) -> None:
    """
    Plot word clouds for title, summary, and comment columns.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.

    Returns:
        None
    """

    columns = ['title', 'summary', 'comment']
    
    for column in columns:
        # Combine all text in the column
        text = ' '.join(df[column].dropna().astype(str))
        
        # Create and generate a word cloud image
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        # Display the generated image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {column.capitalize()}')
        plt.tight_layout(pad=0)
        plt.show()

    return None

def plot_word_cloud_by_category(df: pd.DataFrame) -> None:
    """
    Plot word clouds for title, summary, and comment columns, grouped by category.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'category' columns.

    Returns:
        None
    """
    columns = ['title', 'summary', 'comment']
    categories = df['category'].unique()
    num_categories = len(categories)

    for column in columns:
        rows = (num_categories + 2) // 3  # Calculate number of rows needed
        fig, axes = plt.subplots(rows, 3, figsize=(20, 6 * rows))
        fig.suptitle(f'Word Clouds for {column.capitalize()} by Category', fontsize=16)
        
        for idx, category in enumerate(categories):
            # Filter data for the current category
            category_text = ' '.join(df[df['category'] == category][column].dropna().astype(str))
            
            # Create and generate a word cloud image
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate(category_text)
            
            # Plot in the corresponding subplot
            row = idx // 3
            col = idx % 3
            if rows > 1:
                axes[row, col].imshow(wordcloud, interpolation='bilinear')
                axes[row, col].axis('off')
                axes[row, col].set_title(f'{category}')
            else:
                axes[col].imshow(wordcloud, interpolation='bilinear')
                axes[col].axis('off')
                axes[col].set_title(f'{category}')

        # Remove any unused subplots
        for i in range(num_categories, rows * 3):
            row = i // 3
            col = i % 3
            if rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return None
