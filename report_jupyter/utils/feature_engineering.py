# Data manipulation and analysis
import pandas as pd
import numpy as np

# Natural Language Processing
import spacy
from transformers import BertTokenizer, BertModel

# PyTorch for deep learning
import torch
import torch.nn as nn

# Text processing
import re
from textblob import TextBlob

# Operating system interfaces
import os

# Progress bar
from tqdm import tqdm

# Scaling
from sklearn.preprocessing import MinMaxScaler

def __clear_cuda_memory():
    """Clear CUDA memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def __get_device() -> torch.device:
    """
    Determine the best available device for computation.
    Returns CUDA if available with DataParallel for multiple GPUs,
    else MPS if available, else CPU.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            return torch.device("cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def __load_bert_model(model_name: str):
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return tokenizer, model
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __load_bert_model(model_name)
        raise e

def __process_text_column(column: pd.Series, process_func: callable, batch_size: int) -> pd.Series:
    """
    Process a text column using the given function in batches.

    Args:
        column (pd.Series): The column to process.
        process_func (callable): The function to apply to each text.
        batch_size (int): The number of samples to process at once.

    Returns:
        pd.Series: The processed column.
    """
    try:
        processed = []
        for i in tqdm(range(0, len(column), batch_size), desc="Processing text column"):
            batch = column.iloc[i:i+batch_size]
            processed.extend([process_func(text) if isinstance(text, str) else text for text in batch])
        return pd.Series(processed)
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __process_text_column(column, process_func, batch_size)
        raise e

def __bert_lemmatize(text: str, tokenizer: BertTokenizer, model: BertModel, device: torch.device) -> str:
    try:
        if not isinstance(text, str):
            text = str(text)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model(**inputs)
        
        lemmatized_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        lemmatized_text = ' '.join([token for token in lemmatized_tokens if token not in ['[CLS]', '[SEP]', '[PAD]']])
        
        return lemmatized_text
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __bert_lemmatize(text, tokenizer, model, device)
        raise e

def __lemmatize_column(column: pd.Series, tokenizer: BertTokenizer, model: BertModel, device: torch.device, batch_size: int) -> pd.Series:
    try:
        tokenized = __process_text_column(column, tokenizer.tokenize, batch_size)
        return __process_text_column(tokenized, lambda text: __bert_lemmatize(text, tokenizer, model, device), batch_size)
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __lemmatize_column(column, tokenizer, model, device, batch_size)
        raise e

def __get_embeddings(texts: pd.Series, tokenizer: BertTokenizer, model: BertModel, device: torch.device) -> np.ndarray:
    try:
        inputs = tokenizer(texts.tolist(), return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __get_embeddings(texts, tokenizer, model, device)
        raise e

def __vectorize_column(df: pd.DataFrame, column: str, tokenizer: BertTokenizer, model: BertModel, device: torch.device, batch_size: int) -> pd.DataFrame:
    try:
        df[column] = df[column].astype(str)
        embeddings = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Vectorizing {column}"):
            batch = df[column].iloc[i:i+batch_size]
            embeddings.extend(__get_embeddings(batch, tokenizer, model, device))
        embedding_df = pd.DataFrame(embeddings, columns=[f'{column}_emb_{i}' for i in range(embeddings[0].shape[0])])
        return pd.concat([df, embedding_df], axis=1)
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return __vectorize_column(df, column, tokenizer, model, device, batch_size)
        raise e

def lemmatize(df: pd.DataFrame, model_name: str = 'deepset/bert-large-uncased-whole-word-masking-squad2', batch_size: int = 32) -> pd.DataFrame:
    """
    Lemmatize the text data in specified columns of the DataFrame using BERT lemmatizer.

    Args:
        df (pd.DataFrame): The DataFrame containing text columns to lemmatize.
        model_name (str): The name of the BERT model to use for lemmatization. Default is 'bert-large-uncased-whole-word-masking-squad2'.
        batch_size (int): The number of samples to process at once. Default is 32.

    Returns:
        pd.DataFrame: The DataFrame with lemmatized text in specified columns.
    """
    try:
        tokenizer, model = __load_bert_model(model_name)
        device = __get_device()
        model = model.to(device)
        
        text_columns = ['title', 'summary', 'comment', 'authors']

        for column in tqdm(text_columns, desc="Lemmatizing columns"):
            df[column] = __lemmatize_column(df[column], tokenizer, model, device, batch_size)

        return df
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return lemmatize(df, model_name, batch_size)
        raise e

def vectorize(df: pd.DataFrame, model_name: str = 'deepset/bert-large-uncased-whole-word-masking-squad2', batch_size: int = 32) -> pd.DataFrame:
    """
    Vectorize the text data in specified columns of the DataFrame using a BERT model.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns to be vectorized.
        model_name (str): The name of the BERT model to use. Default is 'bert-large-uncased-whole-word-masking-squad2'.
        batch_size (int): The number of samples to process at once. Default is 32.

    Returns:
        pd.DataFrame: The DataFrame with the vectorized text added as new columns for each input column.
    """
    try:
        tokenizer, model = __load_bert_model(model_name)
        device = __get_device()
        model = model.to(device)

        columns_to_vectorize = ['title', 'summary', 'comment', 'authors']

        for column in tqdm(columns_to_vectorize, desc="Vectorizing columns"):
            df = __vectorize_column(df, column, tokenizer, model, device, batch_size)

        return df
    except RuntimeError as e:
        if "out of memory" in str(e):
            __clear_cuda_memory()
            return vectorize(df, model_name, batch_size)
        raise e

def word_count(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Count the number of words in the 'title', 'summary', 'comment', and 'authors' columns
    and add these counts as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.
        batch_size (int): The number of samples to process at once. Default is 1000.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for word counts.
    """
    columns_to_count = ['title', 'summary', 'comment', 'authors']

    for column in tqdm(columns_to_count, desc="Counting words"):
        word_counts = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {column}", leave=False):
            batch = df[column].iloc[i:i+batch_size]
            word_counts.extend([len(str(x).split()) if isinstance(x, str) else 0 for x in batch])
        df[f'{column}_word_count'] = word_counts

    return df

def named_entity_recognition(df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """
    Count the number of named entities by category in the 'title', 'summary', 'comment', and 'authors' columns
    and add these counts as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.
        batch_size (int): The number of samples to process at once. Default is 100.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for named entity counts by category.
    """

    nlp = spacy.load("en_core_web_sm")
    columns_to_process = ['title', 'summary', 'comment', 'authors']

    for column in tqdm(columns_to_process, desc="Processing NER"):
        df[column] = df[column].astype(str)
        entity_counts = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {column}", leave=False):
            batch = df[column].iloc[i:i+batch_size]
            docs = list(nlp.pipe(batch))
            batch_counts = [{ent.label_: 1 for ent in doc.ents} for doc in docs]
            entity_counts.extend(batch_counts)

        for ent_type in nlp.pipe_labels['ner']:
            df[f'{column}_ner_{ent_type}_count'] = [count.get(ent_type, 0) for count in entity_counts]

    return df

def sentiment_analysis(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Perform sentiment analysis on the 'title', 'summary', 'comment', and 'authors' columns
    and add the sentiment scores as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.
        batch_size (int): The number of samples to process at once. Default is 1000.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for sentiment scores.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    columns_to_analyze = ['title', 'summary', 'comment', 'authors']

    for column in tqdm(columns_to_analyze, desc="Analyzing sentiment"):
        df[column] = df[column].astype(str)
        sentiments = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {column}", leave=False):
            batch = df[column].iloc[i:i+batch_size]
            sentiments.extend(batch.apply(lambda text: TextBlob(text).sentiment.polarity))
        df[f'{column}_sentiment'] = sentiments

    return df

def __calculate_ari(text):
    """
    Calculate the Automated Readability Index (ARI) for a given text.

    Args:
        text (str): The text to calculate the ARI for.

    Returns:
        float: The ARI score for the text.
    """
    characters = len(re.findall(r'\S', text))
    words = len(text.split())
    sentences = len(re.findall(r'\w+[.!?]', text)) or 1
    
    if words == 0:
        return 0
    ari = 4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43
    return max(1, min(ari, 14))

def text_complexity(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Calculate the Automated Readability Index (ARI) for the 'title', 'summary', 'comment', and 'authors' columns
    and add the ARI scores as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the text columns.
        batch_size (int): The number of samples to process at once. Default is 1000.

    Returns:
        pd.DataFrame: The DataFrame with additional columns for ARI scores.
    """

    columns_to_analyze = ['title', 'summary', 'comment', 'authors']

    for column in tqdm(columns_to_analyze, desc="Calculating text complexity"):
        df[column] = df[column].astype(str)
        ari_scores = []
        for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {column}", leave=False):
            batch = df[column].iloc[i:i+batch_size]
            ari_scores.extend(batch.apply(__calculate_ari))
        df[f'{column}_ari'] = ari_scores

    return df

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep all columns except for 'title', 'summary', 'comment', and 'authors',
    and move 'category' and 'split' to be the last two columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with only relevant columns and 'category' and 'split' as the last two columns.
    """
    exclude_columns = ['title', 'summary', 'comment', 'authors']
    keep_columns = [col for col in df.columns if col not in exclude_columns]
    
    for col in ['category', 'split']:
        if col in keep_columns:
            keep_columns.remove(col)
    
    keep_columns.extend(['category', 'split'])
    
    return df[keep_columns]

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the numerical columns of the DataFrame using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with normalized numerical columns.
    """
    # Identify numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Exclude 'category' and 'split' columns if they are present
    columns_to_normalize = [col for col in numerical_columns if col not in ['category', 'split']]

    # Create a MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df, scaler

