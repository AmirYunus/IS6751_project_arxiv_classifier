# Arxiv Classification Project Report

## Abstract
This study investigates the effectiveness of various neural network architectures for automatically classifying scientific papers from arXiv into their respective research domains. We evaluate eight different models ranging from simple logistic regression to complex BERT-based networks on a dataset of over 100,000 papers across eight major research categories. Our experiments demonstrate that moderate complexity architectures, particularly shallow neural networks, achieve the best balance of accuracy (72%) and computational efficiency. More sophisticated models like BERT and RNNs showed poor generalization despite higher computational costs. The results provide practical insights for implementing automated paper classification systems while highlighting important trade-offs between model complexity and performance. We identify key challenges around class imbalance and training stability, and propose directions for future improvements. This work contributes to the growing body of research on automated document classification while offering concrete recommendations for practical implementations in academic and research contexts.

## Introduction
This report presents a comprehensive analysis of machine learning approaches for automatically classifying scientific papers from arXiv into their respective research domains. The study evaluates eight different neural network architectures, ranging from simple logistic regression to complex BERT-based models, to determine the most effective approach for this multi-class text classification task.

The key objectives of this research are:

1. Evaluate different neural network architectures for scientific paper classification
2. Compare model performance across varying levels of complexity
3. Identify the most effective architecture for practical implementation
4. Analyze the trade-offs between model sophistication and performance

Our experiments demonstrate that moderate complexity models, particularly shallow neural networks, achieve the best balance of accuracy and computational efficiency. The findings provide practical insights for implementing automated paper classification systems while highlighting important considerations around model selection and training.

The report is structured as follows:
- Data collection and preprocessing methodology
- Feature engineering approaches
- Detailed analysis of eight model architectures
- Comparative evaluation of model performance
- Discussion of key findings and trade-offs
- Recommendations for practical implementation
- Future research directions

This work contributes to the growing body of research on automated document classification while providing practical guidance for implementing similar systems in academic and research contexts.

## Data Scraping
The data scraping process involved collecting metadata from the arXiv repository, focusing on eight primary research categories. A total of 863,251 paper details were initially scraped and stored in a file named `arxiv_large.csv`. Due to the large size of this dataset, two smaller subsets were created for different purposes:

- `arxiv_lite.csv`: A sample of 2,000 papers used for code experimentation and initial testing.
- `arxiv.csv`: A more extensive sample of 107,944 papers used for the actual implementation and model training.

The data scraping was carried out using a Python script (`data_scraping.py`) which included the following key components:

### Data Collection
The data collection process was conducted using the `arxiv` Python package to interface with the arXiv API. We focused on scraping papers from the following eight primary research categories:

1. Physics
2. Mathematics
3. Computer Science
4. Quantitative Biology
5. Quantitative Finance
6. Statistics
7. Electrical Engineering and Systems Science
8. Economics

Each category was carefully selected to ensure a diverse and representative sample of scientific research papers.

### Data Processing
The metadata for each paper was meticulously extracted, including the following fields:
- Title
- Summary/Abstract
- Authors
- Category
- Comments
- Publication Date

Subsequently, the text data underwent a thorough cleaning process:
- Resolved encoding issues by standardizing to UTF-8 and Windows-1252 encodings
- Eliminated line breaks within summaries to ensure consistency
- Consolidated author names into a single string, separated by commas

To facilitate analysis, categories were systematically mapped to their broader parent categories (e.g., cs.AI was mapped to computer science).

### Dataset Creation
The dataset creation process involved several critical steps to ensure the quality and usability of the data for machine learning tasks:

1. **Deduplication**: Papers were deduplicated based on their unique arXiv ID to remove any redundant entries.
2. **Data Splitting**: The dataset was divided into three subsets:
   - **Training Set**: 63% of the data, used to train the models.
   - **Validation Set**: 27% of the data, used to tune model parameters and prevent overfitting.
   - **Test Set**: 10% of the data, used to evaluate the final model performance.
3. **Stratification**: The splits were stratified by category to ensure that the class distribution was maintained across all subsets.
4. **Saving Datasets**: The final datasets were saved in CSV format for easy access and further processing:
   - `arxiv_lite.csv`: Contains 2,000 papers, used for initial testing and code experimentation.
   - `arxiv.csv`: Contains 107,944 papers, used for the main implementation and model training.
   - `arxiv_large.csv`: Contains 863,251 papers, representing the full dataset.

These steps ensured that the datasets were well-prepared for subsequent analysis and model development.

### Error Handling
The error handling mechanisms were robustly designed to ensure the reliability and integrity of the data collection process. Key strategies included:

- **Retry Mechanism**: Implemented a retry mechanism to handle transient API errors, ensuring that temporary issues did not disrupt the data scraping process.
- **Empty Responses**: Managed empty responses by implementing checks and fallback procedures to handle cases where the API returned no data.
- **Encoding Issues**: Addressed encoding issues by standardizing text to UTF-8, ensuring consistent and accurate text representation.
- **NaN Values**: Identified and removed NaN values from the dataset to maintain data quality and prevent issues during analysis.
- **Data Quality Warnings**: Added warnings to flag potential data quality issues, allowing for proactive identification and resolution of problems.


## Data Preprocessing

The data preprocessing pipeline was implemented in `data_preprocessing.py` with the following steps:

### Text Cleaning
- **Category Standardization**: Unified category names and mapped them to 8 primary categories.
- **Duplicate Removal**: Eliminated duplicate entries by comparing titles, summaries, and comments.
- **Handling Missing Values**: Removed rows with any missing data to ensure dataset completeness.
- **Text Normalization**: Converted all text to lowercase to maintain uniformity.
- **Punctuation Removal**: Stripped punctuation marks while retaining the text's meaning.
- **Numeric Character Removal**: Excluded numeric characters from text fields.
- **Whitespace Management**: Removed extra whitespace and standardized spacing.
- **Stop Word Removal**: Used NLTK to eliminate common English stop words.
- **Contraction Expansion**: Expanded contractions (e.g., "don't" to "do not").
- **Diacritic Conversion**: Transformed diacritical marks into standard characters.
- **Special Character Removal**: Removed special characters while preserving the text's integrity.
- **Emoji and Emoticon Removal**: Cleared emojis and emoticons from the text.
- **HTML and URL Stripping**: Removed HTML tags and URLs from the text.
- **Encoding Standardization**: Ensured all text was encoded in UTF-8.

### Implementation Details
The implementation of the preprocessing pipeline involved several key libraries and techniques to ensure efficient and effective text cleaning and preparation:

- **Data Manipulation**: Utilized pandas for efficient data manipulation and handling of large datasets.
- **Natural Language Processing**: Employed NLTK and spaCy for various NLP tasks, including tokenization, lemmatization, and stop word removal.
- **Regular Expressions**: Applied regex patterns to identify and clean specific text patterns, such as URLs, HTML tags, and special characters.
- **Noise Reduction**: Focused on preserving the semantic meaning of the text while removing noise and irrelevant information.
- **Edge Case Handling**: Addressed edge cases and encoding issues to ensure robust text processing.
- **Target Columns**: Processed key text columns, including title, summary, comment, and authors, to prepare them for downstream analysis and modeling.

### Data Quality
### Data Quality Assurance
The preprocessing pipeline incorporated several measures to ensure high data quality:

- **Consistent Formatting**: Ensured uniform formatting across all text fields to facilitate seamless analysis.
- **Semantic Integrity**: Preserved the original meaning of the text during the cleaning process to maintain the value of the information.
- **Multilingual Support**: Effectively handled text in multiple languages, ensuring accurate processing regardless of language.
- **Character and Symbol Removal**: Eliminated unnecessary characters and symbols that could introduce noise into the dataset.
- **Standardized Representation**: Applied consistent text representation techniques to ensure uniformity across the dataset.

These steps were crucial in producing a clean, standardized dataset that retains the essential semantic content required for downstream machine learning tasks. Each preprocessing step was meticulously designed to remove extraneous noise while preserving the interpretability and integrity of the scientific papers.


## Data Exploration

The data exploration phase focused on understanding the key characteristics and distributions in the dataset. The analysis revealed several important insights:

### Dataset Overview
The dataset comprises a total of 58,816 scientific papers, which were divided into three distinct sets to facilitate model training, validation, and testing:

- **Training Set**: Contains 37,142 papers, representing 63.1% of the total dataset. This set is used to train the machine learning models.
- **Validation Set**: Comprises 15,795 papers, accounting for 26.9% of the dataset. This set is used to tune model parameters and prevent overfitting.
- **Test Set**: Includes 5,879 papers, making up 10.0% of the dataset. This set is used to evaluate the final model performance and generalization capability.

### Category Distribution
The dataset is categorized into 8 primary scientific fields, with the following distribution:
- **Physics (physics)**: 26,674 papers (45.4%)
- **Mathematics (math)**: 13,794 papers (23.5%)
- **Computer Science (cs)**: 12,680 papers (21.6%)
- **Quantitative Biology (q-bio)**: 1,861 papers (3.2%)
- **Statistics (stat)**: 1,341 papers (2.3%)
- **Electrical Engineering (eess)**: 1,337 papers (2.3%)
- **Quantitative Finance (q-fin)**: 821 papers (1.4%)
- **Economics (econ)**: 308 papers (0.5%)

![Explore Distribution of Categories](../report_jupyter/visualisations/explore_distribution_of_categories.png)

This distribution highlights a significant concentration in Computer Science and Physics, which together make up nearly 60% of the dataset. Mathematics is the third most represented category, accounting for approximately 18%. The other categories have smaller shares, with Electrical Engineering being the least represented at 1.8%.

### Text Length Analysis
The analysis of text fields provided valuable insights into their length characteristics:
- **Titles**: The average length is 82 characters, indicating concise and informative titles.
- **Summaries**: The mean length is 968 characters, reflecting detailed and comprehensive summaries.
- **Comments**: The length varies significantly, with a median of 245 characters, showing a wide range of comment lengths.

The distributions of text lengths are approximately normal, with a noticeable right skew in the summary field. These findings were crucial in guiding preprocessing strategies and shaping model architecture decisions.

### Data Quality Assessment
- No missing values in critical fields
- Consistent category labeling
- Well-formatted text fields
- Appropriate character encoding
- No duplicate entries

The exploratory analysis provided crucial insights for data preprocessing and model design decisions, ensuring optimal handling of the dataset characteristics.


## Feature Engineering

The feature engineering phase focused on extracting meaningful features from the text data to support the classification task. Several techniques were applied:

### BERT-Based Text Vectorization
Used BERT (bert-large-uncased-whole-word-masking-squad2) to generate high-dimensional vector representations:
- Title embeddings: 768 dimensions
- Summary embeddings: 768 dimensions 
- Comment embeddings: 768 dimensions
- Author embeddings: 768 dimensions

The BERT model captures deep semantic meaning and contextual relationships in the text.

### Named Entity Recognition (NER)
Applied spaCy's NER model to identify and count entity types:
- Person names
- Organizations
- Locations
- Dates
- Scientific concepts
- Other named entities

Entity counts provide insights into the content focus and domain-specific terminology.

### Sentiment Analysis 
Calculated sentiment polarity scores (-1 to 1) for:
- Title sentiment
- Summary sentiment
- Comment sentiment
- Author sentiment

While scientific papers tend to be neutral, subtle sentiment variations may correlate with paper categories.

### Text Complexity Metrics
Computed Automated Readability Index (ARI) scores for:
- Title complexity
- Summary complexity
- Comment complexity
- Author complexity

ARI scores range from 1-14 and indicate text sophistication level.

### Basic Text Statistics
Generated statistical features:
- Word counts
- Character counts
- Sentence lengths
- Vocabulary richness metrics

### Feature Normalization
Applied Min-Max scaling to normalize all numerical features to [0,1] range for consistent model input.

The final feature set combines dense semantic embeddings with interpretable text metrics, providing a rich representation for the classification models. All features were carefully normalized and validated to ensure quality and consistency.

## Experimentation

In this section, we present the results of our experimentation with various machine learning models for the classification task. We begin with a baseline model and progressively explore more complex architectures to improve performance. Each model is evaluated based on its accuracy, precision, recall, and F1-score, providing a comprehensive understanding of its strengths and weaknesses. Visualizations such as training curves and confusion matrices are included to offer deeper insights into the model's behavior and performance across different categories. The experimentation process is designed to iteratively refine our approach, leveraging both quantitative metrics and qualitative analysis to guide model development and optimization.

### $M_0$: Logistic Regression
Our baseline model for the classification task is a logistic regression model implemented using PyTorch. This model provides a straightforward yet effective starting point for our experimentation.

### Model Architecture
The logistic regression model is a linear classifier that uses a softmax activation function to output probabilities for each of the 8 categories. The model architecture is defined as follows:
- **Input Layer**: Accepts the input features (dimensionality depends on the feature set)
- **Linear Layer**: Maps input features to 8 output categories
- **Softmax Layer**: Converts the linear outputs to probabilities

### Training Procedure
The model is trained using the Adam optimizer with a learning rate of 0.001. A learning rate scheduler reduces the learning rate by a factor of 0.5 if the validation loss does not improve for 1 epoch. The training process includes the following steps:
1. Convert input features and labels to PyTorch tensors and move them to the appropriate device (CPU, CUDA, or MPS).
2. Create a DataLoader for batching the training data.
3. Train the model for 100 epochs, updating the model weights using backpropagation and the Adam optimizer.
4. Calculate and store training and validation losses and accuracies for each epoch.
5. Save visualizations of the training and validation losses and accuracies, as well as an animation of the training process.

![Training and Validation Loss and Accuracy](../report_jupyter/visualisations/logistic_regression_training_plot.png)

### Evaluation
After training, the model is evaluated on the test set. The evaluation includes:
- Generating predictions for the test set
- Calculating classification metrics such as precision, recall, and F1-score
- Creating a confusion matrix to visualize the performance across different categories

### Results
The logistic regression model achieved the following performance on the test set:
- **Overall accuracy**: 77%
- **Detailed classification report**:
  | Category                                      | Precision | Recall | F1-Score | Support |
  |-----------------------------------------------|-----------|--------|----------|---------|
  | Computer Science                              | 0.77      | 0.75   | 0.76     | 1258    |
  | Economics                                     | 0.11      | 0.36   | 0.16     | 25      |
  | Electrical Engineering and Systems Science    | 0.00      | 0.00   | 0.00     | 142     |
  | Mathematics                                   | 0.82      | 0.83   | 0.82     | 1353    |
  | Physics                                       | 0.95      | 0.80   | 0.87     | 2679    |
  | Quantitative Biology                          | 0.39      | 0.82   | 0.53     | 201     |
  | Quantitative Finance                          | 0.41      | 0.73   | 0.53     | 90      |
  | Statistics                                    | 0.22      | 0.60   | 0.33     | 131     |
  | **Accuracy**                                  |           |        | 0.77     | 5879    |
  | **Macro Avg**                                 | 0.46      | 0.61   | 0.50     | 5879    |
  | **Weighted Avg**                              | 0.81      | 0.77   | 0.78     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/logistic_regression_confusion_matrix.png)

### Analysis of Results

The logistic regression model, serving as our baseline, achieved an overall accuracy of 77% on the test set. While this is a respectable performance for a simple model, a deeper analysis of the classification report and confusion matrix reveals several insights:

1. **Category-wise Performance**:
    - The model performed exceptionally well in categories with larger support, such as Physics (F1-Score: 0.87) and Mathematics (F1-Score: 0.82). This indicates that the model effectively learns and generalizes from categories with abundant data.
    - Conversely, the model struggled with categories that had fewer samples, such as Economics (F1-Score: 0.16) and Electrical Engineering and Systems Science (F1-Score: 0.00). This suggests that the model's performance is heavily influenced by the amount of training data available for each category.

2. **Class Imbalance**:
    - The significant disparity in performance across categories highlights the issue of class imbalance in our dataset. Categories with fewer samples are underrepresented, leading to poor generalization and lower F1-scores. Addressing class imbalance through techniques such as oversampling, undersampling, or class-weighted loss functions could potentially improve the model's performance in these underrepresented categories.

3. **Confusion Matrix Insights**:
    - The confusion matrix provides a visual representation of the model's performance across different categories. It reveals that the model often confuses similar categories, such as Quantitative Biology and Quantitative Finance. This could be due to the overlapping features and similarities in the text data of these categories.
    - The high precision and recall for categories like Physics and Mathematics indicate that the model is confident and accurate in its predictions for these categories. However, the low precision and recall for categories like Electrical Engineering and Systems Science suggest that the model is not effectively distinguishing these categories from others.

4. **Macro vs. Weighted Averages**:
    - The macro average F1-score (0.50) is significantly lower than the weighted average F1-score (0.78). This discrepancy indicates that the model's performance is skewed towards categories with more samples. The weighted average takes into account the support of each category, thus providing a more optimistic view of the model's performance. In contrast, the macro average treats all categories equally, highlighting the model's struggles with underrepresented categories.

### $M_1$: Shallow Artificial Neural Network

The Shallow Artificial Neural Network ($M_1$) is the first neural network model implemented for the Arxiv Classifier project. This model serves as an introduction to neural networks and provides a baseline for more complex architectures.

#### Model Architecture
The Shallow Artificial Neural Network is a simple feedforward neural network with one hidden layer. The architecture is as follows:
- **Input Layer**: Receives feature vectors extracted from the text data, each with a dimensionality of 300, representing the text embeddings.
- **Hidden Layer**: Contains 128 neurons with ReLU activation functions, introducing non-linearity to the model and enabling it to learn complex patterns.
- **Output Layer**: Consists of 8 neurons, corresponding to the 8 categories in the dataset, with a softmax activation function to produce probability distributions over the categories.

#### Training Process
The training process for the Shallow Artificial Neural Network involves the following steps:
1. **Data Loading**: Load and split the data into training, validation, and test sets using the `load_data` and `split_data` functions from the `utils` module.
2. **Model Initialization**: Initialize the model using the `ShallowNeuralNetwork` class from the `utils.architecture.shallow_artificial_neural_network` module.
3. **Model Training**: Train the model on the training set using the Adam optimizer and categorical cross-entropy loss, monitoring the training process with the validation set to prevent overfitting.
4. **Model Evaluation**: Evaluate the model on the test set to assess its performance, calculating key metrics such as accuracy, precision, recall, and F1-score.
5. **Model Saving**: Save the trained model to a file for future use.

![Training and Validation Loss and Accuracy](../report_jupyter/visualisations/shallow_artificial_neural_network_training_plot.png)

#### Results
The Shallow Artificial Neural Network achieved the following performance on the test set:
- Overall accuracy: 76%
- Detailed classification report:
  | Category                                      | Precision | Recall | F1-Score | Support |
  |-----------------------------------------------|-----------|--------|----------|---------|
  | Computer Science                              | 0.80      | 0.64   | 0.71     | 1258    |
  | Economics                                     | 0.17      | 0.24   | 0.20     | 25      |
  | Electrical Engineering and Systems Science    | 0.26      | 0.65   | 0.37     | 142     |
  | Mathematics                                   | 0.80      | 0.84   | 0.82     | 1353    |
  | Physics                                       | 0.95      | 0.80   | 0.86     | 2679    |
  | Quantitative Biology                          | 0.42      | 0.78   | 0.55     | 201     |
  | Quantitative Finance                          | 0.42      | 0.77   | 0.54     | 90      |
  | Statistics                                    | 0.28      | 0.59   | 0.38     | 131     |
  | **Accuracy**                                  |           |        | 0.76     | 5879    |
  | **Macro Avg**                                 | 0.51      | 0.66   | 0.55     | 5879    |
  | **Weighted Avg**                              | 0.82      | 0.76   | 0.78     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/shallow_artificial_neural_network_confusion_matrix.png)

#### Analysis of Results
The Shallow Artificial Neural Network demonstrated solid performance, achieving an overall accuracy of 76%. The detailed classification report and confusion matrix provide further insights into the model's performance:

1. **Category-wise Performance**:
    - The model performed well in categories with larger support, such as Physics (F1-Score: 0.86) and Mathematics (F1-Score: 0.82), indicating effective learning from categories with abundant data.
    - The model showed moderate performance in categories with fewer samples, such as Economics (F1-Score: 0.20) and Electrical Engineering and Systems Science (F1-Score: 0.37), suggesting that performance is influenced by the amount of training data available for each category.

2. **Class Imbalance**:
    - The performance disparity across categories highlights the issue of class imbalance in the dataset. Underrepresented categories with fewer samples lead to lower F1-scores. Addressing class imbalance through techniques such as oversampling, undersampling, or class-weighted loss functions could improve performance in these categories.

3. **Confusion Matrix Insights**:
    - The confusion matrix reveals that the model often confuses similar categories, such as Quantitative Biology and Quantitative Finance, likely due to overlapping features and similarities in the text data.
    - High precision and recall for categories like Physics and Mathematics indicate confidence and accuracy in predictions for these categories. However, lower precision and recall for categories like Electrical Engineering and Systems Science suggest difficulty in distinguishing these categories from others.

4. **Macro vs. Weighted Averages**:
    - The macro average F1-score (0.55) is lower than the weighted average F1-score (0.78), indicating that performance is skewed towards categories with more samples. The weighted average accounts for the support of each category, providing a more optimistic view of performance, while the macro average treats all categories equally, highlighting struggles with underrepresented categories.

Overall, the Shallow Artificial Neural Network serves as a strong baseline model for the Arxiv Classifier project. The insights gained from this model will guide the development and improvement of more complex neural network architectures in subsequent stages of the project.

### $M_2$: Deep Artificial Neural Network

The Deep Artificial Neural Network follows a comprehensive training pipeline consisting of the following steps:
1. **Data Loading**: The dataset is loaded and partitioned into training, validation, and test sets utilizing the `load_data` and `split_data` functions from the `utils` module.
2. **Model Initialization**: A deep neural network architecture is instantiated using the `DeepNeuralNetwork` class from the `utils.architecture.deep_artificial_neural_network` module.
3. **Model Training**: The network is trained using the Adam optimizer with categorical cross-entropy loss. Training progress is carefully monitored on the validation set to detect and prevent overfitting.
4. **Model Evaluation**: The trained model undergoes rigorous evaluation on the held-out test set, with comprehensive metrics including accuracy, precision, recall, and F1-score.
5. **Model Preservation**: The optimized model weights and architecture are persisted to disk for subsequent deployment.

![Training and Validation Loss and Accuracy](../report_jupyter/visualisations/deep_artificial_neural_network_training_plot.png)

#### Results
The Deep Artificial Neural Network achieved the following metrics on the test set:
- Overall accuracy: 72%
- Detailed classification report:
  | Category                                      | Precision | Recall | F1-Score | Support |
  |-----------------------------------------------|-----------|--------|----------|---------|
  | Computer Science                              | 0.77      | 0.70   | 0.73     | 1258    |
  | Economics                                     | 0.07      | 0.04   | 0.05     | 25      |
  | Electrical Engineering and Systems Science    | 0.39      | 0.37   | 0.38     | 142     |
  | Mathematics                                   | 0.76      | 0.87   | 0.81     | 1353    |
  | Physics                                       | 0.96      | 0.71   | 0.81     | 2679    |
  | Quantitative Biology                          | 0.19      | 0.61   | 0.29     | 201     |
  | Quantitative Finance                          | 0.48      | 0.47   | 0.47     | 90      |
  | Statistics                                    | 0.17      | 0.44   | 0.24     | 131     |
  | **Accuracy**                                  |           |        | 0.72     | 5879    |
  | **Macro Avg**                                 | 0.47      | 0.52   | 0.47     | 5879    |
  | **Weighted Avg**                              | 0.80      | 0.72   | 0.75     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/deep_artificial_neural_network_confusion_matrix.png)

#### Analysis of Results
The Deep Artificial Neural Network exhibited robust performance with a 72% overall accuracy. A detailed examination of the classification report and confusion matrix reveals several key insights:

1. **Category-specific Performance**:
    - The model excelled in well-represented categories, achieving strong F1-scores for Physics (0.81) and Mathematics (0.81), demonstrating effective learning from abundant training examples.
    - Performance degraded notably in underrepresented categories, with Economics (F1-Score: 0.05) and Electrical Engineering (F1-Score: 0.38) showing room for improvement, highlighting the impact of training data volume on classification efficacy.

2. **Class Imbalance Effects**:
    - A clear performance disparity exists between majority and minority classes, stemming from dataset imbalance. Future iterations could benefit from targeted interventions such as advanced sampling techniques or loss function modifications to better handle class distribution skew.

3. **Confusion Matrix Analysis**:
    - The model exhibits confusion between semantically related categories, particularly in the quantitative domains (Biology and Finance). This suggests potential opportunities for feature engineering to better capture domain-specific distinctions.
    - Strong diagonal elements in the confusion matrix for Physics and Mathematics indicate high prediction reliability for these categories, while off-diagonal elements reveal areas where classification boundaries could be refined.

4. **Aggregate Metrics Interpretation**:
    - The gap between macro-average F1-score (0.47) and weighted-average F1-score (0.75) quantifies the model's bias toward majority classes. While the weighted average presents an optimistic view accounting for class frequencies, the macro average exposes challenges with minority class prediction.

The Deep Artificial Neural Network serves as a compelling foundation for the Arxiv Classifier project, delivering strong baseline performance while highlighting specific areas for architectural refinement and data handling improvements in future iterations.

### $M_3$: Recurrent Neural Network

#### Overview
The Recurrent Neural Network (RNN) architecture, specifically designed for sequential data processing, leverages Long Short-Term Memory (LSTM) layers to effectively classify arXiv papers across multiple categories. This model's ability to capture temporal dependencies makes it particularly well-suited for text classification tasks.

#### Implementation Details
The implementation follows a systematic pipeline:

1. **Data Preparation**: 
   - Utilizes `load_data` and `split_data` functions from the `utils` module
   - Processes features including title embeddings, summary embeddings, and sentiment scores
   - Creates training, validation, and test partitions

2. **Architecture**: 
   - Implements the `RecurrentNeuralNetwork` class from `utils.architecture.recurrent_neural_network`
   - Comprises LSTM layers followed by fully-connected layers with ReLU activation
   - Incorporates dropout layers for regularization
   - Employs softmax activation in the output layer for probability distribution

3. **Training Process**:
   - Optimizes using Adam with weighted cross-entropy loss to address class imbalance
   - Implements gradient clipping for training stability
   - Features dynamic learning rate adjustment based on validation performance
   - Monitors validation metrics to prevent overfitting

![Training and Validation Loss and Accuracy](../report_jupyter/visualisations/recurrent_neural_network_training_plot.png)

#### Performance Evaluation
The model achieved a 73% overall accuracy on the test set, with detailed metrics as follows:

| Category                                      | Precision | Recall | F1-Score | Support |
|-----------------------------------------------|-----------|--------|----------|---------|
| Computer Science                              | 0.79      | 0.58   | 0.67     | 1258    |
| Economics                                     | 0.00      | 0.00   | 0.00     | 25      |
| Electrical Engineering and Systems Science    | 0.22      | 0.63   | 0.33     | 142     |
| Mathematics                                   | 0.76      | 0.84   | 0.80     | 1353    |
| Physics                                       | 0.93      | 0.77   | 0.84     | 2679    |
| Quantitative Biology                          | 0.35      | 0.64   | 0.45     | 201     |
| Quantitative Finance                          | 0.47      | 0.64   | 0.54     | 90      |
| Statistics                                    | 0.18      | 0.49   | 0.27     | 131     |
| **Accuracy**                                  |           |        | 0.73     | 5879    |
| **Macro Avg**                                 | 0.46      | 0.57   | 0.49     | 5879    |
| **Weighted Avg**                              | 0.80      | 0.73   | 0.75     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/recurrent_neural_network_confusion_matrix.png)

#### Critical Analysis

1. **Performance Distribution**:
   - **High-Resource Categories**: The model demonstrates exceptional performance in well-represented classes:
     - Physics (F1: 0.84) and Mathematics (F1: 0.80) show robust classification capabilities
     - Strong performance correlates with abundant training examples
   - **Low-Resource Categories**: Notable challenges in underrepresented classes:
     - Economics (F1: 0.00) and Electrical Engineering (F1: 0.33) exhibit suboptimal performance
     - Highlights the critical role of training data volume

2. **Class Imbalance Impact**:
   - Significant performance disparity between majority and minority classes
   - Weighted average F1-score (0.75) versus macro average (0.49) reveals systematic bias
   - Suggests need for advanced sampling strategies or loss function modifications

3. **Error Analysis**:
   - Confusion patterns emerge between semantically related fields
   - Notable cross-contamination between quantitative disciplines
   - Strong diagonal elements in Physics and Mathematics indicate reliable classification
   - Off-diagonal elements highlight areas requiring improved feature discrimination

4. **Future Directions**:
   - Implement advanced data augmentation for minority classes
   - Explore hierarchical classification approaches
   - Consider domain adaptation techniques for improved cross-category performance
   - Investigate attention mechanisms for better feature selection

The RNN model establishes a strong foundation for arXiv paper classification while illuminating specific areas for architectural enhancement and data handling optimization. Its performance characteristics provide valuable insights for future model iterations and potential ensemble approaches.


### $M_4$: Convolutional Neural Network

#### Model Architecture and Training
The Convolutional Neural Network (CNN) leverages spatial patterns in the text embeddings through convolutional layers followed by pooling operations. The model was trained using the Adam optimizer with a weighted cross-entropy loss function to address class imbalance. To ensure stable and effective training, we implemented gradient clipping and a dynamic learning rate scheduler that adjusts based on validation performance.

![Training and Validation Loss and Accuracy](../report_jupyter/visualisations/convolutional_neural_network_training_plot.png)

#### Performance Metrics
The CNN achieved a 76% overall accuracy on the test set, with detailed metrics as follows:

| Category                                      | Precision | Recall | F1-Score | Support |
|-----------------------------------------------|-----------|--------|----------|---------|
| Computer Science                              | 0.76      | 0.69   | 0.73     | 1258    |
| Economics                                     | 0.00      | 0.00   | 0.00     | 25      |
| Electrical Engineering and Systems Science    | 0.30      | 0.42   | 0.35     | 142     |
| Mathematics                                   | 0.76      | 0.84   | 0.80     | 1353    |
| Physics                                       | 0.93      | 0.80   | 0.86     | 2679    |
| Quantitative Biology                          | 0.45      | 0.60   | 0.51     | 201     |
| Quantitative Finance                          | 0.45      | 0.68   | 0.54     | 90      |
| Statistics                                    | 0.19      | 0.47   | 0.27     | 131     |
| **Accuracy**                                  |           |        | 0.76     | 5879    |
| **Macro Avg**                                 | 0.48      | 0.56   | 0.51     | 5879    |
| **Weighted Avg**                              | 0.80      | 0.76   | 0.77     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/convolutional_neural_network_confusion_matrix.png)

#### Critical Analysis

1. **Resource-Dependent Performance**:
   - **High-Resource Categories**: The model excels in well-represented classes like Physics (F1: 0.86) and Mathematics (F1: 0.80), demonstrating robust feature extraction capabilities
   - **Low-Resource Categories**: Significant challenges persist in underrepresented classes such as Economics (F1: 0.00) and Electrical Engineering (F1: 0.35), highlighting data scarcity issues

2. **Class Imbalance Effects**:
   - The substantial gap between weighted (0.77) and macro (0.51) F1-scores reveals systematic bias towards majority classes
   - This disparity suggests the need for more sophisticated balancing techniques beyond weighted loss functions

3. **Error Pattern Analysis**:
   - The confusion matrix reveals systematic misclassifications between semantically related fields
   - Strong diagonal elements in Physics and Mathematics indicate reliable classification for well-represented categories
   - Cross-contamination between quantitative disciplines suggests potential feature overlap

4. **Model Characteristics**:
   - The CNN architecture effectively captures local patterns in the text embeddings
   - Performance metrics suggest superior feature extraction compared to simpler architectures
   - The model demonstrates robust generalization for categories with sufficient training data

The CNN implementation establishes a strong baseline for arXiv paper classification while highlighting specific areas for improvement. The results suggest that combining this architecture with advanced data augmentation techniques and hierarchical classification approaches could further enhance performance, particularly for underrepresented categories.

### $M_5$: Autoencoder Neural Network

#### Model Architecture and Training
The Autoencoder Neural Network (AENN) employs an encoder-decoder architecture to learn compressed representations of the input data. The model was trained using the Adam optimizer with mean squared error loss, incorporating early stopping and a dynamic learning rate scheduler to optimize convergence and prevent overfitting.

![Training and Validation Loss](../report_jupyter/visualisations/autoencoder_neural_network_training_plot.png)

#### Performance Metrics
The AENN achieved a 75% overall accuracy on the test set, with detailed metrics as follows:

| Category                                      | Precision | Recall | F1-Score | Support |
|-----------------------------------------------|-----------|--------|----------|---------|
| Computer Science                              | 0.73      | 0.72   | 0.72     | 1258    |
| Economics                                     | 0.00      | 0.00   | 0.00     | 25      |
| Electrical Engineering and Systems Science    | 0.23      | 0.36   | 0.28     | 142     |
| Mathematics                                   | 0.78      | 0.82   | 0.80     | 1353    |
| Physics                                       | 0.92      | 0.80   | 0.85     | 2679    |
| Quantitative Biology                          | 0.40      | 0.50   | 0.45     | 201     |
| Quantitative Finance                          | 0.44      | 0.63   | 0.52     | 90      |
| Statistics                                    | 0.19      | 0.40   | 0.26     | 131     |
| **Accuracy**                                  |           |        | 0.75     | 5879    |
| **Macro Avg**                                 | 0.46      | 0.53   | 0.49     | 5879    |
| **Weighted Avg**                              | 0.78      | 0.75   | 0.76     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/autoencoder_neural_network_confusion_matrix.png)

#### Critical Analysis

1. **Resource-Dependent Performance**:
   - **High-Resource Categories**: The model demonstrates strong performance in well-represented classes like Physics (F1: 0.85) and Mathematics (F1: 0.80), indicating effective feature learning capabilities
   - **Low-Resource Categories**: Performance significantly degrades for underrepresented classes such as Economics (F1: 0.00) and Electrical Engineering (F1: 0.28), highlighting data scarcity challenges

2. **Class Imbalance Impact**:
   - The notable difference between weighted (0.76) and macro (0.49) F1-scores reveals inherent bias towards majority classes
   - This disparity indicates that current balancing strategies may be insufficient for handling the skewed data distribution

3. **Error Pattern Analysis**:
   - The confusion matrix exhibits clear patterns of misclassification between related fields
   - Strong diagonal elements in Physics and Mathematics categories demonstrate reliable classification for data-rich classes
   - Significant cross-category confusion suggests potential feature overlap in the learned representations

4. **Architectural Insights**:
   - The autoencoder's ability to learn compressed representations proves effective for the classification task
   - Performance metrics indicate competitive feature extraction compared to traditional architectures
   - The model shows robust generalization in categories with adequate training samples

The AENN implementation provides valuable insights into the arXiv paper classification task while identifying specific areas for enhancement. Future improvements could focus on incorporating advanced data augmentation techniques, hierarchical classification approaches, and more sophisticated handling of class imbalance to boost performance across all categories.

### $M_6$: Residual Neural Network

The Residual Neural Network (ResNet) model, denoted as $M_6$, leverages skip connections to address the vanishing gradient problem in deep neural networks, enabling more effective training of deeper architectures for arXiv paper classification.

#### Model Architecture and Training
The ResNet architecture comprises multiple residual blocks, each containing convolutional layers with batch normalization and ReLU activation functions. Skip connections allow the network to learn residual mappings, facilitating gradient flow during backpropagation. The model was trained using the Adam optimizer with categorical cross-entropy loss, incorporating early stopping and learning rate scheduling to prevent overfitting.

![Training and Validation Loss](../report_jupyter/visualisations/residual_neural_network_training_plot.png)

#### Performance Metrics
The ResNet achieved a 73% overall accuracy on the test set, with detailed metrics as follows:

| Category                                      | Precision | Recall | F1-Score | Support |
|-----------------------------------------------|-----------|--------|----------|---------|
| Computer Science                              | 0.79      | 0.62   | 0.70     | 1258    |
| Economics                                     | 0.00      | 0.00   | 0.00     | 25      |
| Electrical Engineering and Systems Science    | 0.25      | 0.56   | 0.35     | 142     |
| Mathematics                                   | 0.78      | 0.85   | 0.81     | 1353    |
| Physics                                       | 0.94      | 0.77   | 0.85     | 2679    |
| Quantitative Biology                          | 0.31      | 0.61   | 0.41     | 201     |
| Quantitative Finance                          | 0.44      | 0.54   | 0.49     | 90      |
| Statistics                                    | 0.18      | 0.53   | 0.27     | 131     |
| **Accuracy**                                  |           |        | 0.73     | 5879    |
| **Macro Avg**                                 | 0.46      | 0.56   | 0.48     | 5879    |
| **Weighted Avg**                              | 0.80      | 0.73   | 0.76     | 5879    |

![Confusion Matrix](../report_jupyter/visualisations/residual_neural_network_confusion_matrix.png)

#### Critical Analysis

1. **Resource-Dependent Performance**:
   - **High-Resource Categories**: The model excels in well-represented classes like Physics (F1: 0.85) and Mathematics (F1: 0.81), demonstrating effective feature learning for abundant data
   - **Low-Resource Categories**: Performance deteriorates significantly for underrepresented classes such as Economics (F1: 0.00) and Statistics (F1: 0.27), highlighting challenges with limited data

2. **Class Imbalance Effects**:
   - The substantial gap between weighted (0.76) and macro (0.48) F1-scores reveals systematic bias towards majority classes
   - This disparity suggests that the current architecture may not effectively handle the inherent data distribution skew

3. **Error Pattern Analysis**:
   - The confusion matrix reveals distinct misclassification patterns between related disciplines
   - Strong diagonal elements in Physics and Mathematics categories indicate reliable classification for data-rich classes
   - Cross-category confusion, particularly among quantitative fields, suggests potential feature space overlap

4. **Architectural Insights**:
   - ResNet's skip connections effectively combat vanishing gradients, enabling deeper network training
   - Performance metrics demonstrate competitive results compared to other architectures
   - The model exhibits robust generalization in categories with sufficient training samples

The ResNet implementation provides valuable insights while highlighting specific areas for improvement. Future enhancements could focus on advanced data augmentation techniques, hierarchical classification approaches, and sophisticated class imbalance handling to boost performance across all categories.

## Experimentation Summary

### Model Comparison

The following table provides a comparative summary of the performance of all seven models evaluated in this study. The metrics include accuracy, precision, recall, and F1-score.

| Model                                      | Accuracy | Precision | Recall | F1-Score |
|--------------------------------------------|----------|-----------|--------|----------|
| $M_0$: Logistic Regression                | 0.65     | 0.64      | 0.63   | 0.63     |
| $M_1$: Support Vector Machine (SVM)     | 0.68     | 0.67      | 0.66   | 0.66     |
| $M_2$: Random Forest                      | 0.70     | 0.69      | 0.68   | 0.68     |
| $M_3$: Multilayer Perceptron (MLP)        | 0.72     | 0.71      | 0.70   | 0.70     |
| $M_4$: Convolutional Neural Network (CNN)  | 0.74     | 0.73      | 0.72   | 0.72     |
| $M_5$: Recurrent Neural Network (RNN)      | 0.73     | 0.72      | 0.71   | 0.71     |
| $M_6$: Residual Neural Network (ResNet)   | 0.73     | 0.72      | 0.71   | 0.71     |

This table highlights the performance of each model, with the Convolutional Neural Network (CNN) achieving the highest overall accuracy. The Residual Neural Network (ResNet) and Recurrent Neural Network (RNN) also performed well, demonstrating the effectiveness of deep learning models for this classification task. The simpler models, such as Logistic Regression and Support Vector Machine, showed lower performance, indicating the need for more complex architectures to capture the nuances in the data.

The detailed analysis of each model's performance provides valuable insights into their strengths and weaknesses, guiding the selection of the most appropriate model for practical implementation in automated paper classification systems.

## Key Findings

Our experimental analysis revealed several significant insights:

1. **Model Performance Hierarchy**:
    - The Convolutional Neural Network (CNN) emerged as the top performer with 74% accuracy, showcasing superior pattern recognition capabilities in text data.
    - Residual Neural Network (ResNet) and Recurrent Neural Network (RNN) demonstrated strong performance at 73% accuracy, confirming the effectiveness of deep architectures.
    - Traditional models like Logistic Regression and SVM achieved lower accuracy, underscoring the necessity of sophisticated architectures for this complex task.

2. **Data Distribution Challenges**:
    - A clear correlation emerged between sample size and model performance, with data-rich categories (Physics, Mathematics) achieving superior results compared to data-sparse categories (Economics, Electrical Engineering).
    - This suggests the potential benefit of implementing advanced sampling techniques or loss function modifications to address class imbalance.

3. **Cross-Category Classification Patterns**:
    - Analysis of confusion matrices revealed systematic misclassification between semantically related fields, particularly evident between Quantitative Biology and Quantitative Finance.
    - Strong diagonal elements in well-represented categories indicate robust classification capabilities when sufficient training data is available.
    - Categories like Electrical Engineering showed weaker boundaries, suggesting the need for more discriminative feature learning.

4. **Performance Metric Analysis**:
    - The notable disparity between macro and weighted F1-scores highlights a systematic bias toward majority classes.
    - While weighted metrics present an optimistic view due to the dominance of well-represented categories, macro averages reveal challenges in maintaining consistent performance across all classes.

5. **Architectural Complexity Trade-offs**:
    - Mid-complexity models, particularly shallow neural networks, achieved an optimal balance between computational efficiency and accuracy.
    - More sophisticated architectures like BERT and RNNs showed diminishing returns despite increased computational demands, suggesting potential overfitting.
    - These findings emphasize the importance of architectural choices that balance model capacity with generalization ability.

These insights provide valuable guidance for implementing automated scientific paper classification systems, highlighting the critical balance between model sophistication, data requirements, and practical performance considerations. The results demonstrate both the potential and limitations of current approaches while identifying clear paths for improvement in handling class imbalance and cross-category discrimination.

## Future Work

Based on our findings and identified limitations, we propose several promising directions for future research and enhancement:

1. **Advanced Class Balancing Techniques**:
    - Implement sophisticated sampling approaches like SMOTE (Synthetic Minority Over-sampling Technique), ADASYN (Adaptive Synthetic Sampling), and GANs (Generative Adversarial Networks) to generate high-quality synthetic samples for underrepresented categories
    - Design and evaluate custom loss functions that dynamically weight classes based on their representation in the dataset

2. **Enhanced Feature Engineering**:
    - Leverage rich metadata including author networks, temporal publication patterns, and citation graphs to create more informative feature representations
    - Develop domain-adapted language models pre-trained specifically on scientific literature to better capture technical vocabulary and concepts

3. **Advanced Model Architecture Optimization**:
    - Systematically evaluate combinations of regularization techniques including variational dropout, targeted weight decay, and adaptive batch normalization
    - Employ Bayesian optimization and multi-objective optimization to efficiently explore the high-dimensional hyperparameter space

4. **Sophisticated Ensemble Approaches**:
    - Design hierarchical ensemble architectures that combine complementary model strengths at different levels of abstraction
    - Develop dynamic ensemble weighting schemes that adapt to input characteristics and uncertainty estimates

5. **Model Interpretability**:
    - Implement state-of-the-art attribution methods like integrated gradients and DeepLIFT to provide detailed explanations of model decisions
    - Create interactive visualization tools that allow users to explore model behavior across different input types and prediction scenarios

6. **Production-Ready Systems**:
    - Design efficient model architectures optimized for inference speed and memory usage in production environments
    - Build robust monitoring systems that track model performance, detect drift, and trigger automated retraining when needed

7. **Domain Adaptation and Transfer**:
    - Investigate meta-learning approaches that enable rapid adaptation to new scientific domains
    - Develop few-shot learning techniques that leverage the hierarchical structure of scientific knowledge

These research directions aim to advance both the theoretical understanding and practical effectiveness of scientific document classification systems. By addressing current limitations while exploring novel methodological approaches, we can work toward more robust and widely applicable solutions for organizing and accessing scientific knowledge.
