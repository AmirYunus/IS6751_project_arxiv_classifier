# Data manipulation and analysis
import pandas as pd

# Natural Language Processing
import re
from nltk.corpus import stopwords as __stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def categories(df: pd.DataFrame) -> pd.DataFrame:
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
    
    return df


def duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on 'title', 'summary', or 'comment' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', and 'comment' columns.

    Returns:
        pd.DataFrame: The DataFrame with duplicate rows removed.
    """
    # Create a list of columns to check for duplicates
    columns_to_check = ['title', 'summary', 'comment']
    
    # Remove duplicates, keeping the first occurrence
    df_no_duplicates = df.drop_duplicates(subset=columns_to_check, keep='first')
    
    # Reset the index of the DataFrame
    df_no_duplicates = df_no_duplicates.reset_index(drop=True)
    
    return df_no_duplicates

def null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with null or NA values from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with rows containing null or NA values removed.
    """
    # Drop rows with any null or NA values
    df_cleaned = df.dropna()
    
    # Reset the index of the DataFrame
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned


def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with lowercased text data.
    """
    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].str.lower()
    
    return df

def punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove punctuation from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with punctuation removed from text data.
    """
    import string
    
    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
    
    return df

def numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove numbers from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with numbers removed from text data.
    """
    import re
    
    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x) if isinstance(x, str) else x)
    
    return df

def whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove extra whitespace from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with extra whitespace removed from text data.
    """
    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)
    
    return df

def stopwords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove stopwords from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with stopwords removed from text data.
    """

    stop_words = set(__stopwords.words('english'))
    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]) if isinstance(x, str) else x)
    
    return df

def contractions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand contractions in the text data of 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with contractions expanded in text data.
    """

    contraction_dict = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "I would", "i'll": "I will", "i'm": "I am", "i've": "I have",
        "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
        "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
        "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
        "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are",
        "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
    }

    contractions_pattern = re.compile('|'.join(contraction_dict.keys()), re.IGNORECASE)

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(lambda x: contractions_pattern.sub(lambda m: contraction_dict[m.group(0).lower()], x) if isinstance(x, str) else x)
    
    return df

def diacritics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert specific diacritics to their non-diacritic equivalents while preserving other non-English characters in the text data of 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with specific diacritics converted and other characters preserved in text data.
    """

    def process_text(text):
        if not isinstance(text, str):
            return text
        
        # Define a dictionary of diacritic characters and their non-diacritic equivalents
        diacritic_map = {
            'ń': 'n', 'Ń': 'N',
            'ñ': 'n', 'Ñ': 'N',
            'ó': 'o', 'Ó': 'O',
            'ś': 's', 'Ś': 'S',
            'ć': 'c', 'Ć': 'C',
            'ź': 'z', 'Ź': 'Z',
            'ż': 'z', 'Ż': 'Z',
            'ł': 'l', 'Ł': 'L',
            'ą': 'a', 'Ą': 'A',
            'ę': 'e', 'Ę': 'E'
        }
        
        # Replace diacritic characters
        for diacritic, replacement in diacritic_map.items():
            text = text.replace(diacritic, replacement)
        
        return text

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(process_text)
    
    return df

def special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove special characters from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with special characters removed from text data.
    """
    def process_text(text):
        if not isinstance(text, str):
            return text
        
        # Remove special characters while preserving spaces and alphanumeric characters
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(process_text)
    
    return df

def emojis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove emojis and emoticons from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with emojis and emoticons removed from text data.
    """
    def remove_emojis_emoticons(text):
        if not isinstance(text, str):
            return text
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove emoticons
        emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
        text = emoticon_pattern.sub(r'', text)
        
        return text

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(remove_emojis_emoticons)
    
    return df

def html(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove HTML tags and URLs from the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with HTML tags and URLs removed from text data.
    """
    def clean_text(text):
        if not isinstance(text, str):
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(clean_text)
    
    return df

def utf8(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force UTF-8 encoding for the text data in 'title', 'summary', 'comment', and 'authors' columns of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'title', 'summary', 'comment', and 'authors' columns.

    Returns:
        pd.DataFrame: The DataFrame with text data forced to UTF-8 encoding.
    """
    def encode_utf8(text):
        if isinstance(text, str):
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        return text

    text_columns = ['title', 'summary', 'comment', 'authors']
    
    for column in text_columns:
        df[column] = df[column].apply(encode_utf8)
    
    return df