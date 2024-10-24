# Data manipulation and analysis
import pandas as pd

# Natural Language Processing
import spacy
from transformers import AutoTokenizer, AutoModel

# PyTorch for deep learning
import torch

# Text processing
import re
from textblob import TextBlob

# Operating system interfaces
import os

def tokenize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenize the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with tokenized text in specified columns.
    """
    nlp = spacy.load('en_core_web_sm')
    text_columns = ['title', 'summary', 'comment', 'authors']

    def tokenize_text(text):
        if isinstance(text, str):
            return [token.text for token in nlp(text)]
        return text

    for column in text_columns:
        df[column] = df[column].apply(tokenize_text)

    return df


def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lemmatize the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with lemmatized text in specified columns.
    """
    nlp = spacy.load('en_core_web_sm')
    text_columns = ['title', 'summary', 'comment', 'authors']

    for column in text_columns:
        df[column] = df[column].apply(lambda text: ' '.join([token.lemma_ for token in nlp(text)]) if isinstance(text, str) else text)

    return df


def vectorize(df: pd.DataFrame, model_name: str = 'bert-base-uncased') -> pd.DataFrame:
    """
    Vectorize the text data in the 'title', 'summary', 'comment', and 'authors' columns of the DataFrame using a transformer model.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns to be vectorized.
        model_name (str): The name of the transformer model to use. Default is 'bert-base-uncased'.

    Returns:
        pd.DataFrame: The DataFrame with the vectorized text added as new columns for each input column.
    """

    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    columns_to_vectorize = ['title', 'summary', 'comment', 'authors']

    for column in columns_to_vectorize:
        # Ensure the column contains string data
        df[column] = df[column].astype(str)

        # Tokenize the text and get the model output
        inputs = tokenizer(df[column].tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Convert the embeddings to a DataFrame
        embedding_df = pd.DataFrame(embeddings, columns=[f'{column}_emb_{i}' for i in range(embeddings.shape[1])])

        # Concatenate the original DataFrame with the embedding DataFrame
        df = pd.concat([df, embedding_df], axis=1)

    return df


def word_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of words in the 'title', 'summary', 'comment', and 'authors' columns
    and add these counts as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for word counts.
    """
    columns_to_count = ['title', 'summary', 'comment', 'authors']

    for column in columns_to_count:
        df[f'{column}_word_count'] = df[column].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)

    return df

def named_entity_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of named entities by category in the 'title', 'summary', 'comment', and 'authors' columns
    and add these counts as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for named entity counts by category.
    """

    # Load the English NER model
    nlp = spacy.load("en_core_web_sm")

    columns_to_process = ['title', 'summary', 'comment', 'authors']

    for column in columns_to_process:
        # Ensure the column contains string data
        df[column] = df[column].astype(str)

        # Process each text and count entities
        entity_counts = []
        for text in df[column]:
            doc = nlp(text)
            counts = {ent_type: 0 for ent_type in nlp.pipe_labels['ner']}
            for ent in doc.ents:
                counts[ent.label_] += 1
            entity_counts.append(counts)

        # Add new columns for each entity category
        for ent_type in nlp.pipe_labels['ner']:
            df[f'{column}_ner_{ent_type}_count'] = [count[ent_type] for count in entity_counts]

    return df

def sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis on the 'title', 'summary', 'comment', and 'authors' columns
    and add the sentiment scores as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for sentiment scores.
    """

    # Disable tokenizers parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    columns_to_analyze = ['title', 'summary', 'comment', 'authors']

    for column in columns_to_analyze:
        # Ensure the column contains string data
        df[column] = df[column].astype(str)

        # Function to calculate sentiment
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity

        # Apply the function to calculate sentiment
        df[f'{column}_sentiment'] = df[column].apply(get_sentiment)

    return df

def text_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Automated Readability Index (ARI) for the 'title', 'summary', 'comment', and 'authors' columns
    and add the ARI scores as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for ARI scores.
    """

    columns_to_analyze = ['title', 'summary', 'comment', 'authors']

    for column in columns_to_analyze:
        # Ensure the column contains string data
        df[column] = df[column].astype(str)

        # Apply the function to calculate ARI
        df[f'{column}_ari'] = df[column].apply(__calculate_ari)

    return df

def __calculate_ari(text):
    """
    Calculate the Automated Readability Index (ARI) for a given text.

    Args:
        text (str): The text to calculate the ARI for.

    Returns:
        float: The ARI score for the text.
    """
    # Count characters (excluding spaces)
    characters = len(re.findall(r'\S', text))
    
    # Count words
    words = len(text.split())
    
    # Count sentences (simple approximation)
    sentences = len(re.findall(r'\w+[.!?]', text)) or 1  # Ensure at least 1 sentence
    
    # Calculate ARI
    if words == 0:
        return 0
    ari = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    return max(1, min(ari, 14))  # Clamp value between 1 and 14


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all columns except for 'title', 'summary', 'comment', and 'authors',
    and move 'category' and 'split' to be the last two columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with only relevant columns and 'category' and 'split' as the last two columns.
    """
    # Columns to exclude
    exclude_columns = ['title', 'summary', 'comment', 'authors']
    
    # Get all column names except the excluded ones
    keep_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Remove 'category' and 'split' from keep_columns if they exist
    if 'category' in keep_columns:
        keep_columns.remove('category')
    if 'split' in keep_columns:
        keep_columns.remove('split')
    
    # Add 'category' and 'split' to the end of keep_columns
    keep_columns.extend(['category', 'split'])
    
    # Return the DataFrame with only the kept columns
    return df[keep_columns]

