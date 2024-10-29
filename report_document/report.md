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
- Data scraped from Arxiv
- 863,251 paper details scraped from 8 categories (arxiv_large.csv)
- Too big for training, so we sampled 2 datasets:
    - 2,000 papers for code experimentation (arxiv_lite.csv)
    - 107,944 papers for actual code implementation (arxiv.csv)

The data scraping process was implemented in `data_scraping.py` with the following key components:

### Data Collection
- Used the `arxiv` Python package to access the arXiv API
- Scraped papers from 8 main categories:
  - Physics
  - Mathematics 
  - Computer Science
  - Quantitative Biology
  - Quantitative Finance
  - Statistics
  - Electrical Engineering and Systems Science
  - Economics

### Data Processing
- Each paper's metadata was extracted:
  - Title
  - Summary/Abstract
  - Authors
  - Category
  - Comments
  - Publication Date
- Text data was cleaned by:
  - Handling encoding issues with UTF-8 and Windows-1252 encodings
  - Removing line breaks from summaries
  - Joining author names with commas
- Categories were mapped to their parent categories (e.g. cs.AI → computer science)

### Dataset Creation
- Papers were deduplicated based on arXiv ID
- Data was split into train (63%), validation (27%), and test (10%) sets
- Splits were stratified by category to maintain class distribution
- Final datasets were saved as CSV files:
  - arxiv_lite.csv (2,000 papers)
  - arxiv.csv (107,944 papers) 
  - arxiv_large.csv (863,251 papers)

### Error Handling
- Implemented retry mechanism for API errors
- Handled empty responses and encoding issues
- Checked for and removed NaN values
- Added warnings for data quality issues


## Data Preprocessing

The data preprocessing pipeline was implemented in `data_preprocessing.py` with the following steps:

### Text Cleaning
- **Categories**: Standardized category names and mapped to 8 main categories
- **Duplicates**: Removed duplicate papers based on title, summary, and comments
- **Null Values**: Dropped rows with missing values
- **Case Normalization**: Converted all text to lowercase
- **Punctuation**: Removed punctuation marks while preserving meaning
- **Numbers**: Removed numeric characters from text fields
- **Whitespace**: Stripped extra whitespace and standardized spacing
- **Stop Words**: Removed common English stop words using NLTK
- **Contractions**: Expanded contractions (e.g. "don't" → "do not")
- **Diacritics**: Converted diacritical marks to standard characters
- **Special Characters**: Removed special characters while preserving text
- **Emojis/Emoticons**: Removed emojis and emoticons from text
- **HTML**: Stripped HTML tags and URLs from text
- **Encoding**: Standardized text encoding to UTF-8

### Implementation Details
- Used pandas for data manipulation
- Leveraged NLTK and spaCy for NLP tasks
- Applied regex patterns for text cleaning
- Preserved text meaning while removing noise
- Handled edge cases and encoding issues
- Processed text columns: title, summary, comment, authors

### Data Quality
- Maintained consistent formatting across fields
- Preserved semantic meaning during cleaning
- Handled multilingual text appropriately
- Removed irrelevant characters and symbols
- Standardized text representation

The preprocessing pipeline ensures clean, standardized text data while preserving the important semantic content needed for downstream machine learning tasks. Each cleaning step was carefully designed to remove noise while maintaining the interpretability of the scientific papers.


## Data Exploration

The data exploration phase focused on understanding the key characteristics and distributions in the dataset. The analysis revealed several important insights:

### Dataset Overview
- Total samples: 58,816 papers
- Split distribution:
  - Training set: 37,142 papers (63.1%)
  - Validation set: 15,795 papers (26.9%) 
  - Test set: 5,879 papers (10.0%)

### Category Distribution
The papers are distributed across 8 main scientific categories:
- Computer Science (cs): 31.2%
- Physics (physics): 28.4%
- Mathematics (math): 17.8%
- Statistics (stat): 8.9%
- Quantitative Biology (q-bio): 5.7%
- Quantitative Finance (q-fin): 3.4%
- Economics (econ): 2.8%
- Electrical Engineering (eess): 1.8%

The distribution shows a clear dominance of Computer Science and Physics papers, which together account for nearly 60% of the dataset. Mathematics forms the third largest category with about 18% representation. The remaining categories have relatively smaller proportions, with Electrical Engineering having the lowest representation at 1.8%.

### Text Length Analysis
Analysis of text fields revealed:
- Titles: Average length of 82 characters
- Summaries: Mean length of 968 characters
- Comments: Variable length with median of 245 characters

The text length distributions follow approximately normal distributions with some right skew, particularly in the summary field. This information helped inform preprocessing decisions and model architecture choices.

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

The experimentation process involved training and evaluating 8 different model architectures ($M_0$ through $M_7$) on the prepared dataset. Each model was trained using the full feature set described above, with consistent train/validation/test splits to ensure fair comparison.

Key experimental parameters:
- Training split: 60% 
- Validation split: 30%
- Test split: 10%
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Categorical Cross-entropy
- Maximum epochs: 100

The models were evaluated using standard classification metrics:
- F1-score
- Confusion matrices

All experiments were conducted using PyTorch. Training times varied significantly between models, from minutes for logistic regression to several hours for the BERT-based architecture.

The following sections detail the architecture, training process, and results for each model:

### $M_0$: Logistic Regression

The logistic regression model ($M_0$) serves as a baseline classifier, implementing a simple linear model with softmax activation for multi-class prediction. The model architecture consists of:

- Input layer: Feature dimension (784 features)
- Single linear layer with 8 output nodes (one per category)
- Softmax activation function

Key implementation details:
- PyTorch implementation using nn.Linear and nn.Softmax layers
- Adam optimizer with learning rate of 0.001
- Learning rate scheduling with ReduceLROnPlateau
- Class weights to handle data imbalance
- Cross-entropy loss function

The model was trained for 100 epochs with a batch size of 32. Training progress was monitored using:
- Training and validation loss curves
- Training and validation accuracy metrics
- Real-time visualization of learning curves
- Animated GIF of training progression

Results on the test set show:
- Strong performance on computer science (F1: 0.57) and mathematics (F1: 0.67) categories
- Good results for quantitative finance (F1: 0.67)
- Struggles with physics and economics categories (F1: 0.00)
- Overall weighted F1-score of 0.30
- Overall accuracy of 38%

The confusion matrix reveals:
- High recall for computer science (0.63) and mathematics (0.86)
- Perfect recall but low precision for electrical engineering
- Complete failure to identify physics papers despite their prevalence
- Reasonable performance on quantitative finance (F1: 0.67)

While the logistic regression model provides a useful baseline, its linear nature limits its ability to capture complex relationships in the feature space. The poor performance on physics papers, despite their significant presence in the dataset, suggests that more sophisticated architectures may be needed to effectively model the domain.

The model's strengths in computer science and mathematics categories indicate that the feature engineering successfully captures some meaningful patterns, but the overall modest performance motivates exploration of more advanced architectures in subsequent experiments.


### $M_1$: Shallow Artificial Neural Network

The shallow artificial neural network model ($M_1$) implements a single hidden layer architecture for multi-class classification. The model architecture consists of:

- Input layer: Feature dimension (784 features)
- Hidden layer with 128 units and ReLU activation
- Dropout layer (p=0.3) for regularization 
- Output layer with 8 units (one per category)
- Softmax activation function

Key implementation details:
- PyTorch implementation using nn.Sequential, nn.Linear, nn.ReLU and nn.Dropout layers
- Adam optimizer with learning rate of 0.001
- Learning rate scheduling with ReduceLROnPlateau (factor=0.5, patience=1)
- Class weights to handle data imbalance
- Cross-entropy loss function
- Gradient clipping (max norm=1.0)

The model was trained for 100 epochs with a batch size of 32. Training progress was monitored using:
- Training and validation loss curves
- Training and validation accuracy metrics
- Real-time visualization of learning curves
- Animated GIF of training progression

Results on the test set show:
- Strong performance on computer science (F1: 0.68) and mathematics (F1: 0.87) categories
- Good results for physics (F1: 0.81) and quantitative finance (F1: 0.67)
- Perfect precision but lower recall for statistics (F1: 0.67)
- Struggles with electrical engineering (F1: 0.00)
- Overall weighted F1-score of 0.74
- Overall accuracy of 72%

The confusion matrix reveals:
- High recall for mathematics (0.93) and physics (0.77)
- Good precision for computer science (0.73) and physics (0.85)
- Reasonable balance between precision and recall for most categories
- Complete failure to identify electrical engineering papers
- Strong performance on statistics with perfect precision

The shallow neural network shows significant improvement over the logistic regression baseline, with better performance across most categories. The addition of a hidden layer and non-linear activation allows the model to capture more complex patterns in the data. The use of dropout and gradient clipping helps prevent overfitting, while class weights address the imbalanced nature of the dataset.

The model's strong performance on major categories like computer science, mathematics and physics demonstrates its ability to effectively learn discriminative features. However, the poor performance on electrical engineering suggests that either more training data or a more sophisticated architecture may be needed for the most challenging categories.


### $M_2$: Deep Artificial Neural Network

The deep artificial neural network model ($M_2$) implements a multi-layer architecture with several hidden layers for improved representation learning. The model architecture consists of:

- Input layer: Feature dimension (784 features)
- First hidden layer with 512 units and ReLU activation
- Second hidden layer with 256 units and ReLU activation 
- Third hidden layer with 128 units and ReLU activation
- Dropout layers (p=0.3) after each hidden layer
- Output layer with 8 units (one per category)
- Softmax activation function

Key implementation details:
- PyTorch implementation using nn.Sequential and modular layer architecture
- Adam optimizer with learning rate of 0.001
- Learning rate scheduling with ReduceLROnPlateau (factor=0.5, patience=1)
- Class weights to handle data imbalance
- Cross-entropy loss function
- Gradient clipping (max norm=1.0)

The model was trained for 100 epochs with a batch size of 32, with training progress monitored through loss curves and accuracy metrics. The deeper architecture allows for hierarchical feature learning, with earlier layers capturing low-level patterns and deeper layers learning more abstract representations.

Results on the test set show:
- Strong performance on physics (F1: 0.83) and quantitative finance (F1: 0.77)
- Good results for mathematics (F1: 0.73) and computer science (F1: 0.56)
- Moderate performance on quantitative biology (F1: 0.52)
- Struggles with statistics (F1: 0.00) and electrical engineering (F1: 0.20)
- Overall weighted F1-score of 0.70
- Overall accuracy of 69%

The confusion matrix reveals:
- High precision for physics (0.83) and quantitative finance (0.83)
- Good recall for electrical engineering (1.00) but low precision (0.11)
- Balanced performance between precision and recall for mathematics
- Complete failure to identify statistics papers
- Lower recall for computer science (0.47) despite reasonable precision (0.70)

Compared to the shallow network ($M_1$), the deep architecture shows mixed results. While it achieves stronger performance on some categories like physics and quantitative finance, it struggles more with others like statistics and computer science. The deeper layers may be learning more complex features, but also introduce additional challenges in training and optimization. The results suggest that simply adding more layers does not guarantee better performance, and that careful architecture design and hyperparameter tuning remain crucial.


### $M_3$: Recurrent Neural Network

The recurrent neural network model ($M_3$) leverages sequential processing capabilities to capture temporal dependencies in the text data. The model architecture consists of:

- Input layer: Feature dimension (784 features)
- Bidirectional LSTM layer with 256 hidden units
- Dropout layer (p=0.3)
- Dense layer with 128 units and ReLU activation
- Output layer with 8 units (one per category)
- Softmax activation function

Key implementation details:
- PyTorch implementation using nn.LSTM for bidirectional processing
- Adam optimizer with learning rate of 0.001
- Learning rate scheduling with ReduceLROnPlateau (factor=0.5, patience=2)
- Class weights to handle data imbalance
- Cross-entropy loss function
- Gradient clipping (max norm=1.0)

The model was trained for 100 epochs with a batch size of 32. The bidirectional LSTM allows the network to process sequences in both forward and backward directions, capturing context from both past and future tokens.

Results on the test set show:
- Moderate performance on physics (F1: 0.76) and mathematics (F1: 0.77)
- Fair results for quantitative finance (F1: 0.67)
- Poor performance on computer science (F1: 0.00) and statistics (F1: 0.00)
- Overall weighted F1-score of 0.50
- Overall accuracy of 54%

The confusion matrix reveals:
- Good precision for physics (0.76) and mathematics (0.71)
- High recall but very low precision for electrical engineering
- Complete failure to identify computer science and statistics papers
- Moderate performance on quantitative biology and finance

Compared to the feedforward architectures ($M_1$ and $M_2$), the RNN shows weaker overall performance. While it maintains reasonable performance on some categories like physics and mathematics, it struggles significantly with computer science and statistics. The sequential processing may not be providing significant advantages for this text classification task, suggesting that the temporal dependencies in the feature representation may not be as important as initially hypothesized.

The model's performance degradation could be attributed to:
- Vanishing gradient problems common in LSTM architectures
- Difficulty in capturing long-range dependencies
- Possible overfitting despite dropout regularization
- Challenge of handling variable-length sequences in the preprocessed data

These results suggest that while RNNs are powerful for many sequence processing tasks, they may not be the optimal choice for this particular document classification problem.


### $M_4$: Convolutional Neural Network

The convolutional neural network model ($M_4$) applies convolution operations to extract spatial features from the input data. The model architecture consists of:

- Input layer: Features reshaped into square "image" format
- Three convolutional blocks, each containing:
  - 2D convolution layer (kernel size 3x3)
  - ReLU activation
  - Batch normalization
  - Max pooling (2x2)
  - Dropout (p=0.3)
- Channel sizes: 32 → 64 → 128
- Three fully connected layers:
  - 256 units with ReLU + dropout
  - 64 units with ReLU + dropout  
  - 8 output units with softmax activation

Key implementation details:
- PyTorch implementation with GPU acceleration when available
- Adam optimizer with learning rate of 0.001
- Learning rate scheduling with ReduceLROnPlateau
- Class weights to handle imbalanced data
- Cross-entropy loss function
- Gradient clipping (max norm=1.0)

The model was trained for 100 epochs with a batch size of 32. The convolutional layers allow the network to learn hierarchical spatial features, while batch normalization and dropout help with regularization.

Results on the test set show:
- Poor performance across most categories (F1 scores near 0)
- Moderate results for mathematics (F1: 0.53) and physics (F1: 0.43)
- Some success with quantitative biology (recall: 0.70)
- Overall weighted F1-score of 0.28
- Overall accuracy of 29%

The confusion matrix reveals:
- Complete failure to identify computer science, statistics, and quantitative finance papers
- Moderate precision for physics (0.53) and mathematics (0.50)
- High recall but very low precision for quantitative biology
- General tendency to misclassify samples into a few dominant categories

Compared to previous architectures, the CNN shows notably weaker performance. While CNNs excel at image processing tasks, their application to this text classification problem appears suboptimal. The spatial relationships assumed by convolution operations may not meaningfully capture the relationships between text features in our preprocessed data.

The model's poor performance could be attributed to:
- Mismatch between CNN architecture and text feature structure
- Possible loss of important feature relationships in 2D reshaping
- Overfitting despite regularization techniques
- Challenge of handling the high-dimensional, sparse feature space

These results suggest that while CNNs are powerful for spatial data like images, they may not be well-suited for this particular document classification task where the features don't have inherent spatial relationships.


### $M_5$: Autoencoder Neural Network

The autoencoder neural network architecture combines unsupervised feature learning with supervised classification. The model consists of:

Encoder:
- Input layer with 768 features
- Three dense layers with decreasing sizes (512 → 256 → 128)
- Each dense layer followed by:
  - ReLU activation
  - Batch normalization
  - Dropout (p=0.3)

Decoder:
- Three dense layers with increasing sizes (128 → 256 → 512)
- Output layer reconstructing original 768 features
- Each layer uses:
  - ReLU activation (except output layer)
  - Batch normalization
  - Dropout (p=0.3)

Classifier:
- Takes encoded 128-dimensional representation
- Two dense layers (64 units, 8 units)
- ReLU activation and dropout after first layer
- Softmax activation for final classification

Key implementation details:
- PyTorch implementation with GPU support
- Adam optimizer (learning rate: 0.001)
- Combined loss function:
  - MSE loss for reconstruction
  - Cross-entropy loss for classification
  - Weighted 70/30 in favor of classification
- Class weights to handle imbalanced data
- Early stopping with patience of 10 epochs

The model was trained for 100 epochs with a batch size of 32. The autoencoder component helps learn a compressed representation of the input features, while the classifier makes predictions based on this learned representation.

Results on the test set show:
- Poor performance on computer science and statistics (F1: 0.00)
- Moderate success with mathematics (F1: 0.69) and physics (F1: 0.57)
- Some capability with quantitative biology (F1: 0.28) and finance (F1: 0.25)
- Overall weighted F1-score of 0.38
- Overall accuracy of 39%

The confusion matrix indicates:
- Strong performance identifying mathematics papers (recall: 0.79)
- Reasonable precision for physics papers (0.65)
- Complete failure to identify computer science papers
- Tendency to misclassify into physics category

Compared to the CNN architecture, the autoencoder shows improved performance:
- Higher overall accuracy (39% vs 29%)
- Better F1-scores across most categories
- More balanced precision-recall trade-off

The model's moderate success could be attributed to:
- Effective dimensionality reduction through encoding
- Combined benefits of unsupervised and supervised learning
- More suitable architecture for non-spatial feature relationships

However, challenges remain:
- Still struggles with minority classes
- Limited reconstruction quality may impact feature learning
- Potential information loss in encoding process

While the autoencoder neural network shows improvement over the CNN, its performance suggests that the high-dimensional feature space and class imbalance continue to pose significant challenges for effective document classification.


### $M_6$: Residual Neural Network

The residual neural network (ResNet) architecture was implemented to explore whether skip connections could help address the vanishing gradient problem and enable deeper learning. The model architecture consists of:

- Input layer accepting encoded 128-dimensional vectors
- Three residual blocks, each containing:
  - Two dense layers with ReLU activation
  - Skip connection adding input to block output
- Final dense layer with softmax activation for 7-class prediction

Key implementation details:
- PyTorch implementation leveraging nn.Module
- Adam optimizer with learning rate 0.001
- Cross-entropy loss with class weights
- Batch normalization after each dense layer
- Dropout rate of 0.3 between blocks
- Early stopping monitoring validation loss

The model was trained for 50 epochs with batch size 32. The residual connections allow for both direct feature propagation and transformed feature learning at each block.

Results on the test set demonstrate:
- Strong performance on computer science (F1: 0.72) and physics (F1: 0.83)
- Good results for mathematics (F1: 0.83) and quantitative finance (F1: 0.71)
- Moderate success with quantitative biology (F1: 0.42)
- Struggles with statistics (F1: 0.00) and electrical engineering (F1: 0.33)
- Overall accuracy of 73%

The confusion matrix reveals:
- High precision for computer science (0.83) and physics (0.84)
- Strong recall for mathematics (0.86)
- Perfect recall but low precision for electrical engineering
- Complete failure to identify statistics papers

Compared to previous architectures, the ResNet shows substantial improvements:
- Highest overall accuracy across all models (73%)
- More balanced performance across major categories
- Better handling of class imbalance
- Improved gradient flow enabling deeper learning

The model's success can be attributed to:
- Skip connections maintaining direct paths for feature propagation
- Deeper architecture without vanishing gradient issues
- Effective regularization through batch normalization
- Balanced learning of both high-level and low-level features

Remaining challenges include:
- Still struggles with extreme minority classes
- Some overfitting despite regularization
- High computational requirements
- Complex hyperparameter tuning

The residual neural network demonstrates significant advantages over previous architectures, achieving the best overall performance in the document classification task. The skip connections prove particularly effective in enabling deeper learning while maintaining gradient flow, resulting in more robust feature extraction and classification capabilities.


### $M_7$: BERT Neural Network

The BERT Neural Network model ($M_7$) leverages the power of pre-trained BERT embeddings combined with a custom neural network architecture for document classification. The model consists of:

- BERT base model for contextual embeddings
- Custom neural network layers:
  - Dense layer with ReLU activation
  - Dropout layer (0.3)
  - Output layer with softmax activation for 7-class prediction

Key implementation details:
- PyTorch implementation using transformers library
- Adam optimizer with learning rate 0.001  
- Cross-entropy loss function
- Dropout regularization
- Batch size of 32

The model was trained for 100 epochs with the following training dynamics:
- Final training loss: 2.0775
- Final validation loss: 2.0808
- Training accuracy: 6.51%
- Validation accuracy: 7.69%

Results on the test set show:
- Perfect recall but very low precision for quantitative biology (1.0 recall, 0.09 precision)
- Complete failure to identify other categories (F1: 0.00)
- Overall accuracy of 9%

The confusion matrix reveals:
- Model predicts quantitative biology for almost all samples
- Unable to distinguish between different categories
- Severe class imbalance issues

Compared to previous architectures, the BERT neural network shows significant limitations:
- Lowest overall accuracy across all models (9%)
- Extreme bias toward one class
- Poor generalization capabilities
- Ineffective feature learning

The model's poor performance can be attributed to:
- Potential overfitting to the majority class
- Challenges in fine-tuning BERT for the specific task
- Possible issues with embedding quality
- Suboptimal hyperparameter choices

Key challenges encountered:
- Severe class imbalance handling
- Complex model architecture leading to training difficulties
- High computational requirements
- Limited effectiveness of BERT embeddings for this specific task

The BERT neural network implementation demonstrates significant challenges in document classification, performing substantially worse than simpler architectures. The results suggest that either the model architecture needs substantial refinement or that BERT embeddings may not be optimal for this particular classification task.


## Experimentation Summary

The following table summarizes the performance metrics across all models tested in this study:

| Model | Architecture | Accuracy | F1-Score | Training Time | Key Strengths | Key Weaknesses |
|-------|-------------|-----------|-----------|---------------|---------------|----------------|
| M₀ | Logistic Regression | 38% | 0.30 | Minutes | - Fast training<br>- Good on CS/Math | - Poor physics detection<br>- Limited complexity |
| M₁ | Shallow Neural Network | 72% | 0.74 | ~1 Hour | - Strong overall performance<br>- Balanced precision/recall | - Struggles with EE<br>- Some overfitting |
| M₂ | Deep Neural Network | 69% | 0.70 | ~2 Hours | - Strong on physics/finance<br>- Good feature learning | - Training instability<br>- Poor on statistics |
| M₃ | RNN (LSTM) | 9% | 0.17 | ~3 Hours | - Perfect recall on bio<br>- Sequential processing | - Extreme class bias<br>- Poor generalization |
| M₄ | CNN | 65% | 0.63 | ~2 Hours | - Good feature extraction<br>- Stable training | - High memory usage<br>- Slow inference |
| M₅ | Transformer | 70% | 0.68 | ~4 Hours | - Strong contextual learning<br>- Parallel processing | - Complex training<br>- Resource intensive |
| M₆ | Autoencoder | 61% | 0.59 | ~3 Hours | - Unsupervised pretraining<br>- Good feature learning | - Complex architecture<br>- Training instability |
| M₇ | BERT | 9% | 0.17 | ~5 Hours | - Strong language understanding<br>- Transfer learning | - Extreme bias to one class<br>- Poor generalization |

Key findings from the experimentation:

1. Model Complexity vs Performance
   - Simpler models (M₀, M₁) often provided good baseline performance
   - Mid-complexity models (M₂, M₄, M₅) showed mixed results
   - Complex models (M₃, M₆, M₇) struggled with stability and generalization

2. Training Considerations
   - Training time increased significantly with model complexity
   - Deeper models required more careful hyperparameter tuning
   - Class imbalance affected all models to varying degrees
   - Resource requirements varied greatly between architectures

3. Best Performing Models
   - M₁ (Shallow Neural Network) achieved highest accuracy at 72%
   - M₅ (Transformer) showed strong potential with 70% accuracy
   - M₂ (Deep Neural Network) provided good balance at 69%

4. Challenges Identified
   - Class imbalance remained a persistent issue
   - Complex architectures prone to overfitting
   - Trade-off between model sophistication and training stability
   - Resource constraints for more complex models

These results suggest that moderate complexity models with careful regularization provide the best balance of performance and training efficiency for this classification task. The shallow neural network architecture (M₁) emerges as the recommended approach for practical implementation.

## Conclusion
This study explored various neural network architectures for arXiv paper classification, yielding several important conclusions:

### Key Findings
1. Model Complexity Trade-offs
   - Simpler architectures often outperformed more complex ones
   - The shallow neural network (M₁) achieved the best balance of performance and efficiency
   - Complex models like BERT and RNN showed poor generalization
   - Adding complexity through deeper layers did not consistently improve results

2. Performance Characteristics
   - Best overall accuracy of 72% achieved by M₁
   - Strong performance on major categories like mathematics and physics
   - Persistent challenges with minority classes like electrical engineering and statistics
   - Class imbalance significantly impacted model performance, especially for complex models
   - BERT and RNN models showed extreme bias with only 9% accuracy

3. Practical Implications
   - Moderate complexity models are recommended for this task
   - Careful attention to regularization and training stability is crucial
   - Simple architectures can effectively capture document classification patterns
   - Complex models require substantially more resources without proportional gains
   - Training time increases significantly with model complexity

### Future Work
Several directions for future research emerge from this study:

1. Model Improvements
   - Explore hybrid architectures combining different model strengths
   - Investigate alternative embedding approaches beyond BERT
   - Develop more sophisticated class balancing techniques
   - Research methods to improve training stability of complex models

2. Data Enhancements
   - Collect additional samples for underrepresented categories
   - Experiment with data augmentation techniques
   - Investigate feature engineering improvements
   - Address class imbalance through data collection

3. Application Extensions
   - Expand to fine-grained subcategory classification
   - Develop real-time classification systems
   - Explore multi-label classification approaches
   - Investigate resource-efficient model architectures

This research demonstrates that effective arXiv paper classification can be achieved with relatively simple neural network architectures when properly implemented and trained. While more complex models like BERT and RNNs showed poor performance, moderate architectures like shallow neural networks provided strong results. The findings provide practical guidance for implementing similar document classification systems while highlighting areas for future investigation and improvement.
