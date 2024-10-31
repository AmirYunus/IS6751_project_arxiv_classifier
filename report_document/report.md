# Automated Scientific Paper Classification: A Machine Learning Approach to arXiv Category Prediction
Automated Classification of Scientific Papers Using Machine Learning: A Study of arXiv Categories


## Abstract
This report presents an automated classification system for categorizing arXiv scientific papers across eight major disciplines using machine learning techniques. Working with a dataset of over 860,000 papers (sampled to 59,000 for computational feasibility), we developed models to classify papers into Physics, Mathematics, Computer Science, Quantitative Biology, Statistics, Electrical Engineering, Quantitative Finance, and Economics categories. Our methodology covers the complete machine learning pipeline from data collection through model evaluation, aiming to enhance academic information management by improving paper organization and discovery, ultimately facilitating interdisciplinary research and literature navigation in an increasingly complex scientific landscape.


## Introduction
The classification of scientific papers into their respective research domains is a critical task in academic information management. As the volume of scientific literature continues to grow exponentially, automated classification systems become increasingly important for organizing, discovering, and analyzing research papers effectively. This project focuses on developing and evaluating machine learning models for automatically classifying scientific papers from arXiv, one of the largest repositories of electronic preprints.

Our work addresses the challenge of multi-class classification across eight major scientific disciplines: Physics, Mathematics, Computer Science, Quantitative Biology, Statistics, Electrical Engineering, Quantitative Finance, and Economics. By leveraging modern natural language processing techniques and machine learning algorithms, we aim to create a robust classification system that can accurately categorize papers based on their content, helping researchers and institutions better manage and navigate the vast landscape of scientific literature.

The significance of this project extends beyond mere organizational benefits. Accurate classification of scientific papers facilitates interdisciplinary research by helping researchers discover relevant work across different fields. It also enables better understanding of research trends and the evolution of scientific disciplines over time. Furthermore, automated classification systems can help identify emerging research areas and cross-disciplinary connections that might not be immediately apparent through traditional categorization methods.

This report presents our comprehensive approach to building and evaluating such a classification system. We detail our methodology from data collection and preprocessing to model development and evaluation, providing insights into both the technical challenges encountered and the solutions implemented. Our findings contribute to the broader understanding of automated scientific document classification and offer practical insights for similar applications in academic content management.


## Data Scraping
Our initial dataset comprised 863,251 scientific papers from arXiv. However, due to computational constraints, we sampled this down to a more manageable size of 58,816 papers. This sampled dataset was then split into training, validation, and test sets containing 37,142, 15,795, and 5,879 papers respectively, following standard machine learning practices for model development and evaluation.

| Category | Number of Papers |
|----------|-----------------|
| Physics | 26,674 |
| Mathematics | 13,794 |
| Computer Science | 12,680 |
| Quantitative Biology | 1,861 |
| Statistics | 1,341 |
| Electrical Engineering | 1,337 |
| Quantitative Finance | 821 |
| Economics | 308 |

This imbalanced distribution reflects the historical development and relative sizes of different research communities on arXiv, which originated primarily as a physics preprint server before expanding to other fields. The significant class imbalance presented an important consideration for our modeling approach, requiring careful handling to ensure fair treatment of minority classes during classification.

### Data Collection
The data collection process was conducted using the `arxiv` Python package to interface with the arXiv API. This package provided a robust and efficient way to programmatically access the vast repository of scientific papers hosted on arXiv. The package's implementation handled rate limiting and connection management, allowing us to reliably collect data at scale.

We focused our data collection efforts on eight primary research categories that represent distinct scientific domains. These categories included Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering and Systems Science, and Economics. The selection of these specific categories was deliberate, aiming to create a dataset that encompasses a broad spectrum of scientific research while maintaining clear categorical boundaries.

The chosen categories represent both traditional scientific fields like Physics and Mathematics, as well as emerging interdisciplinary areas such as Quantitative Finance and Quantitative Biology. This diversity in the selected categories was crucial for developing a classification system that could effectively handle the varied nature of modern scientific research. Each category contains numerous subcategories, providing fine-grained classification possibilities while still maintaining clear parent category distinctions.

### Data Processing
For each paper in the arXiv repository, we systematically extracted key metadata fields to build our dataset. These fields included the paper's title, summary/abstract, authors, category, comments, and publication date. This comprehensive set of metadata provided the foundation for our subsequent analysis and classification tasks.

The extracted text data required careful cleaning and standardization to ensure consistency across the dataset. We addressed encoding issues by standardizing all text to UTF-8 and Windows-1252 encodings, which helped resolve character rendering problems common in academic texts. Line breaks within paper summaries were eliminated to create uniform, continuous text blocks that would be easier to process. Additionally, we consolidated author names into single comma-separated strings to simplify the data structure while preserving all contributor information.

To streamline our classification approach, we implemented a systematic mapping of specific arXiv categories to their broader parent categories. For example, specialized subcategories like "cs.AI" (Artificial Intelligence) were mapped to their parent category "Computer Science." This hierarchical organization helped maintain clear categorical boundaries while reducing the complexity of our classification task.

### Dataset Creation
The creation of our dataset involved several critical steps to ensure the quality and usability of the data for machine learning tasks. First, we performed deduplication of papers based on their unique arXiv ID to remove any redundant entries that may have been collected during the scraping process. This step was essential to prevent data leakage and ensure the integrity of our subsequent analysis.

After deduplication, we carefully divided our dataset into three distinct subsets to support proper model development and evaluation. The training set, comprising 63% of the data, served as the primary dataset for model training. We allocated 27% of the data to the validation set, which was used to tune model parameters and prevent overfitting during the training process. The remaining 10% was reserved for the test set, which provided an unbiased evaluation of the final model performance.

To maintain the representativeness of our data across all subsets, we implemented stratified splitting based on paper categories. This stratification ensured that the distribution of research categories remained consistent between the training, validation, and test sets, preventing any potential bias in our model evaluation. The final dataset, containing 863,251 papers, was saved in CSV format to facilitate easy access and further processing during subsequent stages of our research.

Through these careful preparation steps, we established a robust foundation for our model development and analysis work. The resulting datasets were well-structured and properly balanced, enabling reliable training and evaluation of our classification models.

### Error Handling
The error handling mechanisms were robustly designed to ensure the reliability and integrity of the data collection process. We implemented a comprehensive retry mechanism to handle transient API errors, which proved essential in maintaining uninterrupted data collection despite temporary network or service issues. This mechanism automatically attempted to reconnect and resume data collection after encountering errors, significantly reducing manual intervention requirements.

Empty API responses presented another critical challenge that we addressed through careful implementation of checks and fallback procedures. When the API returned no data for a particular request, our system logged these instances and implemented appropriate fallback strategies to ensure the continuity of the data collection process. This approach helped maintain the completeness of our dataset while providing clear documentation of any gaps in the collected data.

Text encoding posed a significant challenge due to the diverse nature of scientific content. We addressed this by implementing standardized UTF-8 encoding across all collected text data. This standardization process involved careful handling of special characters and symbols common in scientific papers, ensuring that mathematical notation and technical symbols were preserved accurately in our dataset.

Data quality was maintained through rigorous handling of NaN (Not a Number) values in the dataset. Rather than allowing these null values to propagate through our analysis pipeline, we implemented systematic identification and removal procedures. This approach helped maintain the integrity of our dataset while preventing potential issues during subsequent analysis stages.

To proactively identify and address potential data quality issues, we implemented a comprehensive warning system. This system monitored various aspects of the data collection process and flagged potential problems for review. These warnings covered aspects such as unusual text patterns, unexpected category assignments, and potential data inconsistencies, allowing us to quickly identify and resolve issues before they could impact our analysis.


## Data Preprocessing

The data preprocessing pipeline was implemented in `data_preprocessing.py`, which served as the foundation for preparing our dataset for subsequent analysis and modeling tasks. This pipeline incorporated a comprehensive set of text processing and data cleaning operations designed to ensure data quality and consistency.

The preprocessing workflow was carefully structured to handle the complexities inherent in scientific text data while preserving the semantic meaning crucial for accurate classification. Our implementation focused on both efficiency and effectiveness, utilizing modern natural language processing techniques and robust error handling mechanisms.

Through this pipeline, we systematically transformed raw text data into a clean, standardized format suitable for machine learning applications. The following sections detail the specific steps and techniques employed in our preprocessing approach.

### Text Cleaning
Our text cleaning process involved several comprehensive steps to ensure data quality and consistency. First, we standardized category names by mapping them to our eight primary research categories, creating a unified classification system. We then performed thorough duplicate detection and removal by comparing titles, summaries, and comments across entries to maintain data integrity.

To ensure dataset completeness, we systematically removed rows containing missing values. The text normalization process began with converting all text to lowercase for uniformity, followed by careful removal of punctuation marks while preserving the semantic meaning of the content. We also excluded numeric characters from text fields where they weren't essential to the meaning.

Whitespace management was another crucial aspect, involving the removal of excessive spaces and standardization of spacing throughout the text. Using NLTK, we eliminated common English stop words to reduce noise in our text data. We also expanded contractions to their full forms (e.g., "don't" to "do not") to maintain consistency in word representation.

The cleaning process continued with the conversion of diacritical marks to standard characters and the careful removal of special characters while maintaining text integrity. Modern text elements such as emojis and emoticons were cleared from the dataset, as they weren't relevant to scientific content. We also stripped any HTML tags and URLs that appeared in the text.

Finally, we standardized the encoding of all text to UTF-8 format, ensuring consistent character representation across the entire dataset. This comprehensive text cleaning approach created a standardized, high-quality dataset suitable for our machine learning tasks while preserving the essential meaning of the scientific content.

### Implementation Details
The implementation of the preprocessing pipeline leveraged several key libraries and techniques to ensure efficient and effective text cleaning and preparation. At the core of our data processing workflow, we utilized pandas for efficient manipulation and handling of large datasets. This choice was crucial given the substantial size of our arXiv paper collection and the need for performant data operations.

For natural language processing tasks, we employed both NLTK and spaCy libraries, which provided comprehensive functionality for tokenization, lemmatization, and stop word removal. These NLP operations were essential for breaking down the scientific text into meaningful components that could be effectively analyzed by our models.

Regular expressions played a vital role in our text cleaning process. We implemented carefully crafted regex patterns to identify and clean specific text patterns, including URLs, HTML tags, and special characters that could introduce noise into our analysis. This systematic approach ensured consistency in how we handled various text elements across the entire dataset.

To maintain the integrity of the scientific content, we implemented sophisticated noise reduction techniques that focused on preserving the semantic meaning of the text while removing irrelevant information. This balance was particularly important given the technical nature of the papers in our dataset.

Our pipeline included robust edge case handling to address various text encoding issues and unusual patterns that emerged during processing. This comprehensive approach to edge cases ensured that our preprocessing remained reliable across the diverse range of scientific papers in our dataset.

The preprocessing steps were applied to key text columns including title, summary, comment, and authors. Each of these fields required specific consideration to prepare them appropriately for downstream analysis and modeling tasks while maintaining their distinct characteristics and importance to the classification process.

### Data Quality
The preprocessing pipeline incorporated several measures to ensure high data quality throughout our dataset preparation process. A key focus was maintaining consistent formatting across all text fields, which facilitated seamless analysis in later stages. We implemented careful text cleaning procedures that preserved the semantic integrity of the content, ensuring that the original meaning and value of the scientific information remained intact even as we standardized the format.

Our pipeline was designed with robust multilingual support capabilities, allowing us to effectively process text in multiple languages without loss of meaning or accuracy. This was particularly important given the international nature of scientific research and the diversity of our dataset. We developed sophisticated character and symbol removal procedures that eliminated unnecessary elements that could introduce noise, while being careful to retain characters essential to scientific notation and technical content.

Throughout the preprocessing steps, we maintained a standardized approach to text representation. This consistency was vital for ensuring that all documents, regardless of their source or original format, were transformed into a uniform structure suitable for machine learning applications. The standardization process was carefully calibrated to preserve the nuanced technical language common in scientific papers while removing irrelevant variations in formatting and presentation.

These comprehensive quality control measures were instrumental in producing a clean, standardized dataset that retained the essential semantic content required for downstream machine learning tasks. Each preprocessing step was meticulously designed and tested to ensure it struck the right balance between noise removal and content preservation, ultimately maintaining the interpretability and integrity of the scientific papers while preparing them for effective analysis.

## Data Exploration

The data exploration phase was a critical step in understanding the characteristics and patterns within our dataset. Through comprehensive analysis, we gained valuable insights into the distribution, composition, and unique attributes of our scientific paper collection.

Our exploration focused on several key aspects: the overall size and composition of the dataset, the distribution of papers across different scientific categories, and the statistical properties of various text fields. This systematic investigation allowed us to identify important patterns and potential challenges that would influence our subsequent modeling decisions.

The analysis revealed significant insights about class imbalances, text length distributions, and the relationships between different paper attributes. These findings were instrumental in shaping our approach to model development and evaluation, ensuring that our methodology would effectively address the specific characteristics of the dataset.

### Dataset Overview
Our dataset analysis revealed comprehensive information about each column's characteristics and data quality. The title column contains 58,816 entries with nearly unique values (58,791 unique titles), indicating minimal duplication. Among the few repeated titles, papers related to quantum mechanics, confidence intervals, and particle dynamics appeared multiple times, suggesting these are common research areas or potential variations of similar works.

The summary field demonstrates similar characteristics with 58,793 unique entries out of 58,816 total entries. Notably, there are several withdrawn papers in the dataset, with "paper withdrawn" being the most common summary text. This transparency in documenting withdrawn papers contributes to the dataset's integrity. The summaries vary significantly in length and content, from brief withdrawal notices to detailed technical descriptions of research methodologies and findings.

The comment field shows more standardized patterns, with common formatting conventions emerging. The most frequent comment type is "pages figures" (10,016 occurrences), followed by simply "pages" (6,764 occurrences). This standardization suggests a common documentation practice across submissions, though with varying levels of detail in structural descriptions.

Author distribution analysis reveals interesting patterns in academic publishing. While most authors appear infrequently, there are notable prolific contributors. Lorenzo Iorio leads with 23 papers, followed by B G Sidharth with 18 papers. Large collaboration groups, such as the OPAL and BABAR collaborations, also feature prominently in the dataset, reflecting the collaborative nature of modern scientific research.

The category distribution confirms our earlier observations about class imbalance. Physics dominates with 26,674 papers, followed by Mathematics (13,794) and Computer Science (12,680). Smaller categories like Quantitative Biology (1,861) and Statistics (1,341) have significantly less representation, highlighting the need for careful consideration in our modeling approach to handle this imbalance.

Finally, the dataset split follows a conventional machine learning practice with a train/validation/test ratio of approximately 63/27/10 (37,142/15,795/5,879 samples respectively). This split provides sufficient data for model training while maintaining adequate validation and test sets for robust performance evaluation.

### Category Distribution
Analysis of the category distribution reveals significant imbalances across different academic disciplines in our dataset. Physics emerges as the dominant category, comprising nearly half (45.4%) of all papers with 26,674 entries. This substantial representation reflects the historically strong presence of physics research in academic publishing and preprint servers. The prevalence of physics papers may be attributed to the field's long-standing culture of preprint sharing, dating back to the original arXiv platform's roots in the physics community.

Mathematics and Computer Science form the next tier, with 13,794 (23.5%) and 12,680 (21.6%) papers respectively. Together with Physics, these three fields account for over 90% of the dataset, highlighting a clear skew toward mathematical and computational sciences. This concentration suggests a strong interdisciplinary relationship between these fields, particularly in areas like theoretical physics and computational modeling. The similar volumes of mathematics and computer science papers also indicate the growing importance of computational approaches in modern research.

The remaining categories have considerably smaller representations, forming a distinct third tier in the distribution. Quantitative Biology contains 1,861 papers (3.2%), reflecting the emerging nature of computational approaches in biological sciences. Statistics and Electrical Engineering have similar volumes with 1,341 (2.3%) and 1,337 (2.2%) papers respectively, suggesting these fields may have alternative preferred publishing venues. Quantitative Finance comprises 821 papers (1.3%), while Economics has the smallest representation with just 308 papers (0.5%), potentially indicating that researchers in these fields favor traditional journal submissions over preprint platforms.

This pronounced class imbalance presents important considerations for our modeling approach, particularly in ensuring fair representation and preventing bias toward the dominant categories. Special attention will be needed in our methodology to address these distributional disparities while maintaining model performance across all categories. Potential strategies might include oversampling minority classes, implementing class weights, or employing specialized architectures designed to handle imbalanced datasets. The imbalance also suggests that evaluation metrics should be carefully chosen to provide meaningful insights across all categories, regardless of their size.

The distribution pattern also offers valuable insights into the academic publishing landscape and the adoption of preprint platforms across different disciplines. It highlights how different fields have embraced open science practices at varying rates, with some disciplines showing stronger preferences for traditional publishing routes. This understanding could be valuable for both interpreting our results and considering the broader implications of our classification system.

### Data Quality Assessment
The dataset demonstrates exceptional quality across several key dimensions. A comprehensive analysis reveals that all critical fields are complete, with no missing values detected across the 471,879 records. This completeness ensures reliable analysis and model training without the need for complex imputation strategies.

The category labeling system maintains strict consistency throughout the dataset, with each paper properly assigned to one of the eight major academic fields. This standardization is crucial for accurate classification tasks and cross-category analysis.

Text fields throughout the dataset exhibit well-formatted content, with proper character encoding that correctly handles special characters, mathematical symbols, and international author names. This formatting consistency facilitates effective text processing and analysis without the need for extensive cleaning operations.

A thorough duplicate check confirms that each entry in the dataset is unique, eliminating any concerns about data redundancy that could skew analysis results or introduce bias into model training. This uniqueness, combined with the other quality factors, provides a solid foundation for robust machine learning applications.

### Text Length Analysis
Our analysis of text length characteristics across different categories provides valuable insights into how various academic disciplines structure their papers. The following table presents the average character length for titles, summaries, and comments across the eight major categories in our dataset. These metrics offer a quantitative perspective on the varying communication styles and documentation practices across disciplines.

| Category | Title | Summary | Comment |
|----------|--------|----------|----------|
| Computer Science | 62.12 | 801.40 | 46.86 |
| Economics | 60.69 | 792.05 | 35.24 |
| Electrical Engineering | 74.83 | 882.93 | 41.30 |
| Mathematics | 52.24 | 439.31 | 38.66 |
| Physics | 61.80 | 648.08 | 44.28 |
| Quantitative Biology | 71.13 | 897.44 | 39.11 |
| Quantitative Finance | 60.88 | 708.23 | 34.48 |
| Statistics | 66.30 | 813.17 | 44.49 |

The text length analysis reveals notable variations across different academic disciplines in how they structure their titles, summaries, and comments. These differences likely reflect the distinct communication norms, complexity of concepts, and methodological approaches characteristic of each field.

In terms of title length, Electrical Engineering and Systems Science (74.83 characters) and Quantitative Biology (71.13 characters) demonstrate significantly longer titles compared to other fields. For Electrical Engineering, this may reflect the need to specify both the technical system being studied and the methodological approach, while in Quantitative Biology, longer titles likely arise from the need to specify both the biological system and the quantitative method being applied. In contrast, Mathematics shows notably shorter titles (52.24 characters), possibly reflecting the field's preference for concise, abstract representations of concepts.

Summary lengths show even more pronounced variations. Quantitative Biology leads with the longest summaries (897.44 characters), followed closely by Electrical Engineering (882.93 characters) and Statistics (813.17 characters). The extended length in Quantitative Biology summaries might be attributed to the need to describe complex biological systems alongside mathematical methodologies. Mathematics, interestingly, shows the shortest summaries (439.31 characters), which could reflect the field's reliance on formal mathematical notation (not captured in character counts) and its tendency toward precise, economical expression.

Computer Science shows the longest average comment length (46.86 characters), followed by Physics (44.28 characters) and Statistics (44.49 characters). This pattern might reflect these fields' strong preprint culture and emphasis on implementation details or experimental conditions. The longer comments in Computer Science papers could indicate additional information about code availability, computational requirements, or implementation details. Quantitative Finance shows the shortest comments (34.48 characters), possibly due to the field's more recent adoption of the preprint system and different commenting conventions.

These variations in text length metrics provide valuable insights into the communication patterns and documentation requirements of different academic disciplines. The differences likely arise from a combination of historical conventions, practical necessities, and the inherent complexity of conveying discipline-specific concepts effectively.


### Word Frequency Analysis
Word frequency analysis reveals distinctive patterns in vocabulary usage across different academic disciplines. After removing common stop words, we identified the most frequently occurring terms in each category's papers. This analysis provides insights into the key concepts, methodologies, and focus areas that characterize each field. Below are the top 5 most frequent words in each category, highlighting the unique linguistic patterns and terminology preferences:

| Physics | Frequency |
|---------|-----------|
| model | 11768 |
| results | 8395 |
| using | 7810 |
| two | 7509 |
| field | 7308 |

| Mathematics | Frequency |
|-------------|-----------|
| paper | 4551 |
| show | 4448 |
| also | 3795 |
| prove | 3765 |
| space | 3538 |

| Electrical Engineering and Systems Science | Frequency |
|------------------------------------------|-----------|
| proposed | 1119 |
| model | 965 |
| system | 818 |
| paper | 791 |
| using | 751 |

| Computer Science | Frequency |
|-----------------|-----------|
| data | 7468 |
| paper | 6730 |
| model | 5703 |
| using | 5028 |
| problem | 4980 |

| Quantitative Biology | Frequency |
|--------------------|-----------|
| model | 1614 |
| data | 1022 |
| using | 717 |
| results | 705 |
| networks | 702 |

| Economics | Frequency |
|-----------|-----------|
| model | 205 |
| data | 146 |
| economic | 143 |
| paper | 129 |
| models | 121 |

| Statistics | Frequency |
|------------|-----------|
| data | 1658 |
| model | 1279 |
| models | 851 |
| methods | 792 |
| method | 715 |

| Quantitative Finance | Frequency |
|--------------------|-----------|
| model | 797 |
| market | 638 |
| financial | 457 |
| price | 454 |
| risk | 437 |

The analysis of top words across different scientific categories reveals fascinating patterns about the focus and methodological approaches in each field.

In Physics, we see "paper" as the most common word, indicating frequent reference to prior work. Words like "quantum", "theory", and "energy" reflect the field's fundamental focus on understanding physical phenomena and developing theoretical frameworks. The high frequency of these terms aligns with physics' theoretical nature and its quest to explain fundamental properties of matter and energy.

Mathematics shows a distinct pattern with words like "prove", "space", and "show" dominating the top terms. This reflects mathematics' emphasis on formal proofs and abstract spaces. The prevalence of "prove" (3765 occurrences) particularly highlights the field's rigorous approach to establishing mathematical truths through logical argumentation.

Electrical Engineering and Systems Science demonstrates its applied nature through terms like "proposed", "model", and "system". The high frequency of "proposed" (1119 occurrences) suggests a strong focus on new methodologies and solutions, while "system" indicates the field's emphasis on integrated approaches to solving engineering challenges.

Computer Science shows a clear data-centric focus with "data" appearing 7468 times, the highest frequency among all terms across categories. The prominence of "model", "using", and "problem" reflects the field's emphasis on practical problem-solving and implementation of solutions using various computational models.

Quantitative Biology's top terms - "model", "data", "networks" - reveal its modern computational approach to biological research. The high frequency of "networks" (702 occurrences) particularly reflects the field's focus on understanding biological systems through network analysis and modeling.

Economics shows a strong theoretical foundation with "model" and "models" in its top terms, while "economic" and "data" reflect its empirical nature. The relatively lower frequencies (205 for "model") reflect the smaller dataset size but maintain similar thematic patterns to other quantitative fields.

Statistics, unsurprisingly, centers around "data" (1658 occurrences) and various types of "models". The presence of "methods" and "method" in its top terms underscores the field's focus on developing and applying analytical techniques.

Quantitative Finance shows its specialized nature with domain-specific terms like "market", "financial", "price", and "risk". The high frequency of "model" (797 occurrences) indicates the field's heavy reliance on mathematical modeling for financial analysis.

Cross-category analysis reveals interesting patterns. "Model" appears as a top term in six out of eight categories, highlighting the ubiquity of modeling approaches across modern scientific disciplines. "Data" features prominently in computer science, statistics, and quantitative biology, reflecting the increasing importance of data-driven research methodologies. The term "paper" appears frequently in physics and computer science, suggesting strong citation cultures in these fields.

These patterns reflect the evolving nature of scientific research, where traditional theoretical approaches are increasingly complemented by data-driven and computational methods across all disciplines. The analysis also reveals the distinct methodological signatures of each field while highlighting the growing convergence in analytical approaches across scientific domains.


#### Word Cloud Visualisation

### N-Gram Analysis
N-gram analysis provides deeper insights into how words are commonly used together in scientific texts across different fields. By examining frequent word combinations (2-grams, 3-grams, and 4-grams), we can better understand the key concepts, methodologies, and research focuses within each discipline. This analysis reveals common phrases and terminology that characterize the discourse in each field. Below, we present the most frequent n-grams for each scientific category, starting with Physics.

#### Physics
2-Grams
| Phrase | Frequency |
|--------|-----------|
| magnetic field | 1871 |
| et al | 1013 |
| monte carlo | 803 |
| numerical simulations | 670 |
| ground state | 646 |
| phase transition | 606 |
| black hole | 596 |
| good agreement | 565 |
| experimental data | 541 |
| star formation | 527 |
| magnetic fields | 487 |
| quantum mechanics | 447 |
| boundary conditions | 444 |
| dark matter | 443 |
| field theory | 425 |
| first time | 421 |
| power law | 414 |
| electric field | 406 |
| angular momentum | 400 |
| wide range | 384 |

3-Grams
| Phrase | Frequency |
|--------|-----------|
| monte carlo simulations | 239 |
| phys rev lett | 207 |
| density functional theory | 155 |
| et al phys | 140 |
| al phys rev | 133 |
| molecular dynamics simulations | 104 |
| external magnetic field | 103 |
| quantum field theory | 100 |
| active galactic nuclei | 89 |
| play important role | 88 |
| cosmic microwave background | 86 |
| direct numerical simulations | 85 |
| monte carlo simulation | 82 |
| nonlinear schrodinger equation | 81 |
| quantum monte carlo | 76 |
| partial differential equations | 72 |
| magnetic field strength | 72 |
| van der waals | 61 |
| hubble space telescope | 61 |
| star formation rate | 60 |

4-Grams
| Phrase | Frequency |
|--------|-----------|
| et al phys rev | 129 |
| al phys rev lett | 89 |
| sloan digital sky survey | 56 |
| phys rev lett bf | 41 |
| density matrix renormalization group | 39 |
| active galactic nuclei agn | 39 |
| cosmic microwave background cmb | 38 |
| density functional theory dft | 36 |
| auau collisions sqrtsnn gev | 35 |
| coronal mass ejections cmes | 33 |
| first order phase transition | 30 |
| digital sky survey sdss | 30 |
| markov chain monte carlo | 26 |
| good agreement experimental data | 24 |
| nonlinear partial differential equations | 24 |
| first error statistical second | 23 |
| physics beyond standard model | 23 |
| cherenkov telescope array cta | 23 |
| error statistical second systematic | 20 |
| igr j igr j | 20 |

5-Grams
| Phrase | Frequency |
|--------|-----------|
| et al phys rev lett | 87 |
| sloan digital sky survey sdss | 30 |
| first error statistical second systematic | 20 |
| et al phys rev e | 18 |
| al phys rev lett bf | 14 |
| density matrix renormalization group dmrg | 14 |
| using density matrix renormalization group | 11 |
| density matrix renormalization group method | 10 |
| et al phys rev b | 10 |
| igr j igr j igr | 10 |
| j igr j igr j | 10 |
| widefield infrared survey explorer wise | 9 |
| relativistic heavy ion collider rhic | 9 |
| density functional theory dft calculations | 9 |
| markov chain monte carlo mcmc | 8 |
| ground state first excited state | 8 |
| upsilons resonance belle detector kekb | 8 |
| observed discovery isotopes discussed isotope | 8 |
| discovery isotopes discussed isotope brief | 8 |
| first refereed publication including production | 8 |

The n-gram analysis of physics papers reveals interesting patterns in the language and focus of physics research. Looking at the 3-grams, we see a strong emphasis on experimental physics and astronomical observations, with phrases like "hubble space telescope" and references to specific measurement techniques. The high frequency of "van der waals" indicates significant research activity in molecular forces and interactions.

The 4-gram analysis provides deeper insights into the methodological and topical focus of physics papers. The most frequent 4-gram "et al phys rev" and "al phys rev lett" reflect the dominance of Physical Review journals in physics publications. There's a notable presence of astronomy-related terms like "sloan digital sky survey" and "cosmic microwave background cmb", indicating the field's strong astronomical research component. Methodological approaches are represented by phrases like "density matrix renormalization group" and "markov chain monte carlo", showing the importance of computational and theoretical methods. The presence of "density functional theory dft" suggests significant activity in quantum mechanics and materials science.

The 5-gram analysis further reinforces these patterns while revealing additional details about research practices. Publication-related phrases continue to dominate, with "et al phys rev lett" being the most frequent. Technical methodologies are elaborated in phrases like "density matrix renormalization group dmrg" and "using density matrix renormalization group". Experimental physics is represented by phrases related to particle physics facilities like "relativistic heavy ion collider rhic". The presence of phrases about statistical error reporting ("first error statistical second systematic") indicates the field's rigorous approach to experimental uncertainty.

This n-gram analysis reveals physics as a field balanced between theoretical frameworks, experimental methodologies, and observational astronomy, with a strong emphasis on rigorous publication practices and statistical analysis.

#### Mathematics
Top 20 2-grams in Summary (Mathematics)
---------------------------------------
also show: 333
paper study: 321
group g: 302
main result: 288
lie algebra: 281
differential equations: 274
finitely generated: 263
sufficient conditions: 262
necessary sufficient: 237
lie algebras: 232
also prove: 226
paper prove: 224
finite dimensional: 220
allows us: 192
space x: 190
also give: 190
locally compact: 185
upper bound: 184
fixed point: 183
let g: 181

Top 20 3-grams in Summary (Mathematics)
---------------------------------------
necessary sufficient conditions: 125
partial differential equations: 86
necessary sufficient condition: 80
give necessary sufficient: 78
algebraically closed field: 77
main result paper: 53
central limit theorem: 53
von neumann algebra: 51
finite group g: 49
give new proof: 45
mapping class group: 44
field characteristic zero: 42
ordinary differential equations: 42
upper lower bounds: 40
finite element method: 39
von neumann algebras: 36
let g finite: 36
boundary value problems: 36
vertex operator algebra: 36
lie group g: 35

Top 20 4-grams in Summary (Mathematics)
---------------------------------------
give necessary sufficient conditions: 46
algebraically closed field characteristic: 29
give necessary sufficient condition: 26
let g finite group: 20
algebraically closed field k: 19
elliptic partial differential equations: 12
compact hausdorff space x: 12
locally compact group g: 12
aleq aleq aleq aleq: 12
let r commutative noetherian: 11
locally compact hausdorff spaces: 11
partial differential equations pdes: 11
ordinary differential equations odes: 11
locally compact quantum groups: 11
commutative noetherian local ring: 10
finite dimensional weight spaces: 10
closed field characteristic zero: 10
alternating direction method multipliers: 10
system ordinary differential equations: 10
compact connected lie group: 10

Top 20 5-grams in Summary (Mathematics)
---------------------------------------
aleq aleq aleq aleq aleq: 11
algebraically closed field characteristic zero: 9
algebraically closed field characteristic p: 8
let r commutative noetherian ring: 8
algebraically closed field k characteristic: 7
modules finite dimensional weight spaces: 7
alternating direction method multipliers admm: 7
mathbb z langle x rangle: 7
integrable modules finite dimensional weight: 5
also give necessary sufficient condition: 5
let k algebraically closed field: 5
compact hausdorff spaces continuous maps: 5
mathbb rd discrete multiset lambda: 5
give necessary sufficient conditions bounded: 4
finitely generated module commutative noetherian: 4
let x smooth projective curve: 4
bounded derived categories coherent sheaves: 4
elliptic partial differential equations pdes: 4
fractional brownian motion hurst parameter: 4
locally compact hausdorff spaces continuous: 4

The n-gram analysis of Mathematics papers reveals interesting patterns in the language and concepts commonly used in mathematical research. Looking at the 3-grams, we see a strong emphasis on boundary value problems and algebraic concepts like vertex operator algebra and Lie groups, reflecting core areas of mathematical research.

The 4-gram analysis provides deeper insights into the mathematical discourse patterns. The most frequent 4-gram "give necessary sufficient conditions" (46 occurrences) indicates the formal nature of mathematical proofs and theorem statements. There's also significant presence of algebraic terminology with phrases like "algebraically closed field characteristic" (29 occurrences) and "let g finite group" (20 occurrences). The analysis reveals frequent discussion of various mathematical domains including differential equations (both partial and ordinary), topology (through phrases involving "compact Hausdorff space"), and group theory (references to "locally compact group").

The 5-gram analysis further reinforces these patterns while revealing more specific mathematical constructs. The presence of "algebraically closed field characteristic zero" and related variants suggests substantial work in abstract algebra and field theory. The phrase "modules finite dimensional weight spaces" indicates research in representation theory, while "alternating direction method multipliers admm" points to optimization theory applications. The analysis also shows frequent discussion of categorical concepts through phrases like "bounded derived categories coherent sheaves" and geometric concepts via "smooth projective curve".

Overall, the n-gram analysis effectively captures the formal, precise nature of mathematical writing while highlighting the predominant subfields and methodological approaches in mathematics research. The frequent occurrence of phrases related to conditions, proofs, and specific mathematical structures aligns well with the theoretical and rigorous nature of mathematical discourse.

#### Electrical Engineering and Systems Science
Top 20 2-grams in Summary (Electrical engineering and systems science)
----------------------------------------------------------------------
neural network: 150
proposed method: 141
results show: 130
deep learning: 112
paper propose: 104
experimental results: 103
simulation results: 96
speech recognition: 93
paper proposes: 88
neural networks: 87
speech enhancement: 80
show proposed: 80
proposed approach: 79
paper presents: 69
error rate: 61
deep neural: 60
propose novel: 55
training data: 53
convolutional neural: 53
proposed algorithm: 53

Top 20 3-grams in Summary (Electrical engineering and systems science)
----------------------------------------------------------------------
automatic speech recognition: 45
deep neural network: 35
speech recognition asr: 34
results show proposed: 33
experimental results show: 31
simulation results show: 29
convolutional neural network: 28
model predictive control: 27
word error rate: 24
convolutional neural networks: 23
deep neural networks: 19
paper propose novel: 16
base station bs: 15
internet things iot: 15
neural network cnn: 15
nonorthogonal multiple access: 14
paper proposes novel: 14
effectiveness proposed method: 14
predictive control mpc: 14
generative adversarial networks: 13

Top 20 4-grams in Summary (Electrical engineering and systems science)
----------------------------------------------------------------------
automatic speech recognition asr: 32
model predictive control mpc: 14
convolutional neural network cnn: 14
simulation results show proposed: 13
convolutional neural networks cnns: 10
word error rate wer: 9
deep neural network dnn: 9
sound event detection sed: 9
nonorthogonal multiple access noma: 8
channel state information csi: 8
automatic speaker verification asv: 8
unmanned aerial vehicles uavs: 8
synthetic aperture radar sar: 8
distributed energy resources ders: 7
intelligent reflecting surface irs: 7
deep neural networks dnns: 7
experimental results show proposed: 7
bit error rate ber: 7
deep convolutional neural networks: 6
closedform expressions outage probability: 6

Top 20 5-grams in Summary (Electrical engineering and systems science)
----------------------------------------------------------------------
minimum variance distortionless response mvdr: 5
deep convolutional neural networks cnns: 4
automatic speech recognition asr systems: 4
sound event localization detection seld: 4
minimum mean square error mmse: 4
automatic speech recognition asr system: 4
word error rate wer reductions: 3
massive multipleinput multipleoutput mimo systems: 3
simultaneous wireless information power transfer: 3
wireless information power transfer swipt: 3
orthogonal frequency division multiplexing ofdm: 3
automatic speech recognition asr model: 3
recurrent neural network transducer rnnt: 3
endtoend ee automatic speech recognition: 3
ee automatic speech recognition asr: 3
deep learningbased speech enhancement se: 3
model predictive control mpc framework: 3
convolutional neural network cnn based: 3
construct finite abstractions together corresponding: 3
experimental results show proposed system: 3

The n-gram analysis of the Electrical Engineering and Systems Science papers reveals several key research areas and methodological approaches dominant in the field. Looking at the 4-grams, we see a strong focus on speech recognition and neural networks, with "automatic speech recognition asr" being the most frequent (32 occurrences), followed by "model predictive control mpc" and "convolutional neural network cnn" (14 occurrences each). This suggests that speech processing and neural network applications are major research areas within electrical engineering.

The prevalence of terms related to deep learning and neural networks is particularly notable, with multiple variations appearing in the top 20 list: "convolutional neural networks cnns", "deep neural network dnn", and "deep neural networks dnns". This indicates the significant role of deep learning methodologies in current electrical engineering research. Additionally, the presence of terms like "sound event detection sed" and "automatic speaker verification asv" further emphasizes the field's strong focus on audio processing and recognition systems.

The 5-gram analysis provides more detailed insights into specific methodological approaches and applications. The most frequent 5-gram, "minimum variance distortionless response mvdr" (5 occurrences), is a key technique in signal processing. The strong presence of speech recognition-related 5-grams, such as "automatic speech recognition asr systems" and variations thereof, reinforces the field's emphasis on speech processing technologies. The appearance of terms like "massive multipleinput multipleoutput mimo systems" and "wireless information power transfer swipt" highlights the importance of wireless communication systems in the field.

Notably, both 4-gram and 5-gram analyses show a significant focus on experimental and simulation results, with phrases like "simulation results show proposed" and "experimental results show proposed" appearing frequently. This suggests a strong emphasis on empirical validation and practical applications in electrical engineering research. The presence of various performance metrics (like "word error rate wer" and "bit error rate ber") further underscores the field's focus on quantitative evaluation and performance optimization.

#### Computer Science
Top 20 2-grams in Summary (Computer science)
--------------------------------------------
paper propose: 689
results show: 656
neural networks: 590
neural network: 578
machine learning: 566
experimental results: 553
paper presents: 542
paper present: 493
proposed method: 437
et al: 399
deep learning: 387
propose novel: 379
lower bound: 284
paper proposes: 283
propose new: 272
reinforcement learning: 258
results demonstrate: 238
recent years: 237
show proposed: 231
convolutional neural: 231

Top 20 3-grams in Summary (Computer science)
--------------------------------------------
experimental results show: 173
paper propose novel: 121
convolutional neural networks: 117
deep neural networks: 115
convolutional neural network: 105
results show proposed: 103
experimental results demonstrate: 93
natural language processing: 91
paper propose new: 87
deep neural network: 80
large language models: 73
wireless sensor networks: 61
artificial intelligence ai: 59
upper lower bounds: 56
simulation results show: 55
internet things iot: 54
paper presents novel: 52
channel state information: 52
language models llms: 51
recurrent neural network: 50

Top 20 4-grams in Summary (Computer science)
--------------------------------------------
large language models llms: 50
convolutional neural network cnn: 47
experimental results show proposed: 42
convolutional neural networks cnns: 37
deep neural networks dnns: 32
natural language processing nlp: 30
channel state information csi: 25
deep convolutional neural networks: 25
multiagent reinforcement learning marl: 22
automatic speech recognition asr: 21
convolutional neural networks cnn: 21
experimental results demonstrate proposed: 21
simulation results show proposed: 20
long shortterm memory lstm: 19
mobile ad hoc networks: 19
recurrent neural networks rnns: 17
experimental results demonstrate effectiveness: 16
wireless sensor networks wsns: 16
theory practice logic programming: 15
partially observable markov decision: 15

Top 20 5-grams in Summary (Computer science)
--------------------------------------------
theory practice logic programming tplp: 10
minimum mean square error mmse: 9
partially observable markov decision process: 8
mobile ad hoc networks manets: 8
natural language processing nlp tasks: 7
total cpu usage clock cycles: 7
appear theory practice logic programming: 6
partially observable markov decision processes: 6
additive white gaussian noise awgn: 6
cooperative multiagent reinforcement learning marl: 6
observable markov decision process pomdp: 6
experimental results show proposed algorithm: 6
deep convolutional neural networks cnns: 6
central bank digital currency cbdc: 6
experimental results show proposed method: 6
partially ordered twoway buchi automata: 6
computer vision natural language processing: 5
channel state information transmitter csit: 5
markov chain monte carlo mcmc: 5
experimental results demonstrate effectiveness proposed: 5

The n-gram analysis of computer science abstracts reveals interesting patterns in the field's terminology and research focus areas. Looking at the 3-grams, we see a strong emphasis on neural networks and language models, with terms like "channel state information" and "recurrent neural network" appearing frequently. This suggests a significant focus on machine learning and natural language processing research.

The 4-gram analysis further reinforces this observation, with "large language models llms" being the most frequent 4-gram, followed by various neural network architectures like "convolutional neural network cnn" and "deep neural networks dnns". There's also a notable presence of reinforcement learning ("multiagent reinforcement learning marl") and natural language processing ("natural language processing nlp"). The frequency of terms related to experimental results ("experimental results show proposed", "experimental results demonstrate proposed") indicates a strong empirical focus in computer science research.

The 5-gram analysis shows a diverse range of technical concepts spanning multiple computer science subfields. While some terms continue the machine learning theme ("deep convolutional neural networks cnns"), others relate to theoretical computer science ("theory practice logic programming tplp"), wireless communications ("additive white gaussian noise awgn"), and emerging technologies ("central bank digital currency cbdc"). The presence of statistical and mathematical terms ("minimum mean square error mmse", "markov chain monte carlo mcmc") demonstrates the quantitative foundation of computer science research.

This n-gram analysis effectively captures the multifaceted nature of computer science research, highlighting its strong focus on machine learning and AI while also showing the field's breadth across theoretical, practical, and emerging technological domains.

#### Quantitative Biology
Top 20 2-grams in Summary (Quantitative biology)
------------------------------------------------
gene expression: 119
et al: 78
neural networks: 69
machine learning: 69
free energy: 67
experimental data: 65
differential equations: 55
growth rate: 53
amino acids: 52
wide range: 52
monte carlo: 45
amino acid: 45
results show: 44
gene regulatory: 42
biological systems: 41
numerical simulations: 41
neural network: 41
time series: 41
mathematical model: 41
molecular dynamics: 40

Top 20 3-grams in Summary (Quantitative biology)
------------------------------------------------
gene regulatory networks: 26
monte carlo simulations: 17
basic reproduction number: 15
ordinary differential equations: 15
play important role: 14
molecular dynamics simulations: 13
gene expression data: 11
partial differential equations: 11
protein structure prediction: 10
reproduction number r: 10
partial differential equation: 10
gene regulatory network: 10
magnetic resonance imaging: 9
slow wave sleep: 9
convolutional neural networks: 9
amino acid sequences: 8
transcription factor binding: 8
functional brain networks: 8
machine learning methods: 8
protein interaction networks: 8

Top 20 4-grams in Summary (Quantitative biology)
------------------------------------------------
basic reproduction number r: 8
severe acute respiratory syndrome: 6
roc curve auc cstatistic: 6
partial differential equation pde: 5
transcription factor binding sites: 5
functional magnetic resonance imaging: 5
ordinary differential equations odes: 5
slow wave sleep duration: 5
ordinary differential equation model: 5
acute respiratory syndrome coronavirus: 5
intrinsically disordered proteins idps: 4
molecular dynamics md simulations: 4
single nucleotide polymorphisms snps: 4
model gene regulatory networks: 4
respiratory syndrome coronavirus sarscov: 4
mutual information input output: 4
gene regulatory networks grns: 4
approximate bayesian computation abc: 4
rooted binary phylogenetic trees: 4
principal component analysis pca: 3

Top 20 4-grams in Summary (Quantitative biology)
------------------------------------------------
basic reproduction number r: 8
severe acute respiratory syndrome: 6
roc curve auc cstatistic: 6
partial differential equation pde: 5
transcription factor binding sites: 5
functional magnetic resonance imaging: 5
ordinary differential equations odes: 5
slow wave sleep duration: 5
ordinary differential equation model: 5
acute respiratory syndrome coronavirus: 5
intrinsically disordered proteins idps: 4
molecular dynamics md simulations: 4
single nucleotide polymorphisms snps: 4
model gene regulatory networks: 4
respiratory syndrome coronavirus sarscov: 4
mutual information input output: 4
gene regulatory networks grns: 4
approximate bayesian computation abc: 4
rooted binary phylogenetic trees: 4
principal component analysis pca: 3

The n-gram analysis of quantitative biology papers reveals distinct patterns that highlight the field's focus on biological systems, mathematical modeling, and medical applications. Looking at the 4-grams, we observe a strong emphasis on epidemiological research, with phrases like "basic reproduction number r" and "severe acute respiratory syndrome" appearing frequently, likely reflecting significant research activity related to infectious diseases and epidemics.

The analysis shows substantial focus on computational and statistical methods applied to biological problems. Terms like "roc curve auc cstatistic" and "approximate bayesian computation abc" indicate the importance of statistical analysis and model evaluation in the field. The presence of "partial differential equation pde" and "ordinary differential equations odes" demonstrates the field's reliance on mathematical modeling approaches to understand biological systems.

Molecular and genetic research themes are evident through phrases like "transcription factor binding sites", "single nucleotide polymorphisms snps", and "gene regulatory networks grns". This indicates significant research activity in genomics and gene regulation. The appearance of "molecular dynamics md simulations" and "intrinsically disordered proteins idps" suggests active research in protein structure and dynamics.

Neurobiological research is represented by terms like "functional magnetic resonance imaging" and "slow wave sleep duration", indicating the field's involvement in brain research and sleep studies. The presence of "protein interaction networks" and "model gene regulatory networks" highlights the importance of network analysis approaches in understanding biological systems.

The diversity of these n-grams reflects quantitative biology's interdisciplinary nature, combining biological research with mathematical modeling, statistical analysis, and computational methods. The field appears to span multiple scales of biological organization, from molecular interactions to whole-organism studies and population-level analyses.

#### Economics
Top 20 2-grams in Summary (Economics)
-------------------------------------
treatment effects: 20
time series: 15
results show: 13
panel data: 12
economic growth: 12
social norms: 12
interest rate: 11
et al: 11
machine learning: 11
macroeconomic variables: 11
treatment effect: 10
covid pandemic: 10
insurance companies: 10
fixed effects: 9
per capita: 9
nash equilibrium: 9
large number: 9
income inequality: 9
average treatment: 8
empirical application: 8

Top 20 3-grams in Summary (Economics)
-------------------------------------
average treatment effects: 7
risk stock price: 7
green technology innovation: 6
pure strategy nash: 5
greenhouse gas emissions: 4
twoway fixed effects: 4
artificial intelligence ai: 4
statistical decision theory: 4
asymmetric causality tests: 4
play important role: 4
synthetic control method: 4
dynamic transport network: 4
existence competitive equilibrium: 4
green technological innovation: 4
stock price crashes: 4
treatment effect heterogeneity: 3
machine learning ml: 3
economic time series: 3
mean squared error: 3
match value distribution: 3

Top 20 4-grams in Summary (Economics)
-------------------------------------
risk stock price crashes: 4
sens transitivity condition described: 3
dynamic stochastic general equilibrium: 3
pure strategy nash equilibria: 3
optimum pure strategy nash: 3
green technological innovation risk: 3
technological innovation risk stock: 3
innovation risk stock price: 3
quality green technology innovation: 3
quality green technological innovation: 3
risk stock price collapse: 3
number negative news reports: 3
negative news reports media: 3
news reports media listed: 3
reports media listed companies: 3
apply method estimate returns: 2
method estimate returns college: 2
twoway fixed effects estimator: 2
extended twoway fixed effects: 2
travel mode choice prediction: 2

Top 20 5-grams in Summary (Economics)
-------------------------------------
green technological innovation risk stock: 3
technological innovation risk stock price: 3
number negative news reports media: 3
negative news reports media listed: 3
news reports media listed companies: 3
apply method estimate returns college: 2
true match value distribution data: 2
optimum pure strategy nash equilibrium: 2
default risk historically underserved groups: 2
identification dynamic discrete choice model: 2
set set inclusion sense achieved: 2
cancer service access regional areas: 2
autocrats countries lowquality institutions tend: 2
countries lowquality institutions tend wealthy: 2
social choice functions satisfy gamma: 2
dynamic converges pure nash equilibrium: 2
uefa champions league group stage: 2
impact electricity blackouts poor infrastructure: 2
electricity blackouts poor infrastructure livelihood: 2
blackouts poor infrastructure livelihood residents: 2

The n-gram analysis of Economics papers reveals several interesting patterns in the research focus and methodological approaches within the field. Looking at the 4-grams, there is a notable emphasis on financial risk and stock market dynamics, with phrases like "risk stock price crashes" appearing frequently. This suggests a significant research interest in market volatility and financial crashes within the economics literature.

Game theory and equilibrium analysis also feature prominently, as evidenced by phrases such as "pure strategy nash equilibria" and "optimum pure strategy nash". This indicates the importance of game theoretical frameworks in economic research, particularly in studying strategic interactions and decision-making processes.

The data also shows a focus on green technology and innovation, with multiple related 4-grams including "green technological innovation risk" and "technological innovation risk stock". This suggests growing attention to environmental economics and the intersection of sustainability with financial markets.

The 5-gram analysis further reinforces these themes while providing additional context. The prevalence of phrases related to news media and company reporting (e.g., "negative news reports media listed") indicates research interest in how information flow affects economic outcomes. Infrastructure and development economics also appear as important themes, as shown by phrases like "impact electricity blackouts poor infrastructure" and "electricity blackouts poor infrastructure livelihood".

Methodological approaches are also evident in the n-grams, with phrases like "twoway fixed effects estimator" and "extended twoway fixed effects" suggesting the use of sophisticated econometric techniques in empirical research. The presence of terms related to discrete choice models and value distribution indicates a strong quantitative and empirical orientation in economic research methodologies.

#### Statistics
Top 20 2-grams in Summary (Statistics)
--------------------------------------
monte carlo: 195
time series: 119
markov chain: 101
proposed method: 88
chain monte: 86
real data: 85
maximum likelihood: 80
data sets: 78
machine learning: 72
simulation studies: 62
sample size: 62
et al: 59
data analysis: 58
simulation study: 55
paper propose: 54
data set: 53
bayesian inference: 53
high dimensional: 49
variable selection: 46
regression models: 44

Top 20 3-grams in Summary (Statistics)
--------------------------------------
markov chain monte: 86
chain monte carlo: 86
monte carlo mcmc: 33
sequential monte carlo: 25
maximum likelihood estimation: 23
approximate bayesian computation: 17
gene expression data: 15
principal component analysis: 15
simulated real data: 14
monte carlo methods: 13
maximum likelihood estimator: 12
hamiltonian monte carlo: 12
real data sets: 12
average treatment effect: 12
generalized linear models: 11
paper propose new: 11
time series data: 10
bayesian computation abc: 10
extensive simulation studies: 10
time series analysis: 10

Top 20 4-grams in Summary (Statistics)
--------------------------------------
markov chain monte carlo: 86
chain monte carlo mcmc: 33
approximate bayesian computation abc: 10
sequential monte carlo smc: 9
using markov chain monte: 7
chain monte carlo methods: 6
principal component analysis pca: 6
reproducing kernel hilbert space: 6
functional magnetic resonance imaging: 6
simulation studies real data: 5
monte carlo mcmc algorithms: 5
integrated nested laplace approximation: 5
monte carlo mcmc methods: 5
simulated real data sets: 4
demonstrate superior performance proposed: 4
nested laplace approximation inla: 4
gaussian process gp regression: 4
strong law large numbers: 4
magnetic resonance imaging fmri: 4
monte carlo smc methods: 4

Top 20 5-grams in Summary (Statistics)
--------------------------------------
markov chain monte carlo mcmc: 33
using markov chain monte carlo: 7
markov chain monte carlo methods: 6
chain monte carlo mcmc algorithms: 5
chain monte carlo mcmc methods: 5
integrated nested laplace approximation inla: 4
functional magnetic resonance imaging fmri: 4
sequential monte carlo smc methods: 4
efficient markov chain monte carlo: 4
markov chain monte carlo method: 4
markov chain monte carlo algorithm: 4
markov chain monte carlo algorithms: 3
extensive simulation studies real data: 3
simulation studies real data analysis: 3
chain monte carlo mcmc method: 3
fisher lecture dimension reduction regression: 3
bayesian checking second levels hierarchical: 3
checking second levels hierarchical models: 3
second levels hierarchical models arxiv: 3
approximate bayesian computation abc methods: 3

The n-gram analysis of Statistics papers reveals several key methodological and analytical themes that dominate the field. Looking at the 4-grams, Markov Chain Monte Carlo (MCMC) emerges as the most prevalent technique, with 86 occurrences of "markov chain monte carlo" and 33 occurrences of "chain monte carlo mcmc". This highlights the fundamental importance of MCMC methods in statistical computation and Bayesian inference.

Other significant methodological approaches include Approximate Bayesian Computation (ABC) and Sequential Monte Carlo (SMC), appearing 10 and 9 times respectively. The presence of "principal component analysis pca" and "reproducing kernel hilbert space" (6 occurrences each) indicates the relevance of dimensionality reduction techniques and functional analysis in statistical research.

The 5-gram analysis further reinforces the prominence of MCMC methods, with variations like "markov chain monte carlo mcmc" (33 occurrences) and "using markov chain monte carlo" (7 occurrences) appearing frequently. The analysis also reveals an emphasis on practical applications and validation, with phrases like "extensive simulation studies real data" and "simulation studies real data analysis" appearing multiple times.

Notably, there's a significant focus on hierarchical modeling and Bayesian methodology, as evidenced by phrases like "bayesian checking second levels hierarchical" and "checking second levels hierarchical models". The presence of "integrated nested laplace approximation inla" suggests the use of advanced computational methods for Bayesian inference, particularly in complex statistical models.

#### Quantitative Finance
Top 20 2-grams in Summary (Quantitative finance)
------------------------------------------------
time series: 100
stock market: 78
financial markets: 75
monte carlo: 59
deep learning: 48
risk measures: 46
stock price: 42
financial market: 39
stochastic volatility: 38
machine learning: 38
option pricing: 38
systemic risk: 38
transaction costs: 33
neural networks: 32
stock exchange: 28
stock prices: 28
implied volatility: 27
option prices: 27
et al: 26
brownian motion: 26

Top 20 3-grams in Summary (Quantitative finance)
------------------------------------------------
limit order book: 19
financial time series: 18
monte carlo simulations: 14
monte carlo simulation: 12
deep learning models: 12
stochastic differential equations: 12
systemic risk measures: 11
stochastic volatility model: 10
probability density function: 9
chinese stock market: 9
stochastic volatility models: 8
deep reinforcement learning: 8
unified growth theory: 8
partial differential equation: 8
crude oil futures: 8
markov chain monte: 7
chain monte carlo: 7
time series data: 7
stochastic control problem: 7
stock price prediction: 7

Top 20 4-grams in Summary (Quantitative finance)
------------------------------------------------
markov chain monte carlo: 7
limit order book lob: 5
fundamental theorem asset pricing: 5
optimal dividend ratcheting strategy: 4
detrended fluctuation analysis dfa: 4
good trade execution strategies: 4
long shortterm memory lstm: 4
deep deterministic policy gradient: 3
principles financial product synthesis: 3
order book lob data: 3
model limit order book: 3
observed real financial markets: 3
graph neural networks gnns: 3
laplacian based semisupervised ranking: 3
limit order book data: 3
machine learning deep learning: 3
liquid stocks traded shenzhen: 3
stocks traded shenzhen stock: 3
traded shenzhen stock exchange: 3
probability density function pdf: 3

Top 20 5-grams in Summary (Quantitative finance)
------------------------------------------------
limit order book lob data: 3
liquid stocks traded shenzhen stock: 3
stocks traded shenzhen stock exchange: 3
garch model rational error distribution: 3
deep deterministic policy gradient algorithm: 2
accurately track daily cumulative vwap: 2
sums nth degree values cnt: 2
nth degree values cnt volumes: 2
degree values cnt volumes unt: 2
values cnt volumes unt market: 2
cnt volumes unt market trades: 2
risk measures valueatrisk expected shortfall: 2
covar based xvaralphax increasing function: 2
local stochastic volatility lsv models: 2
paper proposes deep reinforcement learning: 2
wealth condensation ie convergence state: 2
subindices wheat maize soyabeans rice: 2
wheat maize soyabeans rice barley: 2
detect irregular trade behaviors stock: 2
irregular trade behaviors stock market: 2

The n-gram analysis of Quantitative Finance papers reveals interesting patterns in the research focus and methodologies within this field. Looking at the 4-grams, we see a strong emphasis on computational and statistical methods, with "markov chain monte carlo" being the most frequent (7 occurrences), highlighting the importance of stochastic modeling in financial research. The prominence of "limit order book lob" and related terms (5 occurrences) indicates significant research attention to market microstructure and order book dynamics.

The analysis also reveals a focus on fundamental financial concepts, with "fundamental theorem asset pricing" appearing frequently (5 occurrences). Machine learning and artificial intelligence approaches are well-represented, as evidenced by terms like "long shortterm memory lstm" and "deep deterministic policy gradient", suggesting the growing integration of advanced computational methods in quantitative finance research.

Examining the 5-grams provides more detailed insights into specific research areas. Market microstructure continues to be a key theme, with "limit order book lob data" appearing frequently (3 occurrences). There's notable attention to specific market analysis, particularly regarding the Shenzhen Stock Exchange, as indicated by several related 5-grams. Risk management and volatility modeling are also prominent themes, shown by phrases like "risk measures valueatrisk expected shortfall" and "local stochastic volatility lsv models".

The presence of terms related to deep reinforcement learning and behavioral analysis suggests an emerging focus on advanced algorithmic trading strategies and market behavior studies. Additionally, the appearance of agricultural commodity-related terms ("wheat maize soyabeans rice") indicates research interest in commodity markets and their dynamics.

### Topic Modelling
The topic modeling analysis for Quantitative Finance reveals several distinct research themes and methodological approaches within the field. Using Latent Dirichlet Allocation (LDA), we identified 5 main topics from the paper summaries, each representing different aspects of quantitative finance research. The analysis provides insights into the current research focus areas, methodological preferences, and emerging trends in the field. Below are the detailed findings for each topic, along with their relative prevalence in the corpus.


#### Physics
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: model, phase, field, theory, quantum, equation, states, results, state, equations 10379

Topic 2: mass, model, models, velocity, gas, formation, field, observed, density, matter 2730

Topic 3: data, emission, galaxies, observations, stars, xray, using, present, sources, high 2920

Topic 4: energy, using, results, magnetic, field, beam, electron, high, used, model 5191

Topic 5: model, quantum, data, time, results, systems, network, networks, new, using 5454

The topic modeling analysis for Physics reveals distinct research areas and methodological approaches within the field. Topic 1, with keywords like "model", "phase", "field", "theory", and "quantum", appears to represent theoretical physics research, particularly quantum mechanics and field theories. This topic has the highest document count (10,379), suggesting it's a dominant area of research. Topic 2 focuses on classical physics and astrophysics, with terms like "mass", "velocity", "gas", and "density", indicating research related to fluid dynamics and celestial body formation (2,730 documents). Topic 3 is clearly centered on observational astronomy and astrophysics, featuring terms like "emission", "galaxies", "observations", and "stars", with 2,920 documents discussing these themes.

Topic 4, with 5,191 documents, appears to concentrate on experimental physics, particularly in areas involving particle physics and electromagnetic phenomena, as evidenced by terms like "energy", "magnetic", "beam", and "electron". Topic 5, with 5,454 documents, seems to represent an intersection of quantum physics and modern computational methods, with terms like "quantum", "systems", "network", and "networks" suggesting research in quantum computing or complex systems.

The distribution of documents across these topics indicates a relatively balanced research landscape in physics, with theoretical physics (Topic 1) having the largest share, followed by computational/quantum systems (Topic 5) and experimental physics (Topic 4), while classical physics/astrophysics (Topic 2) and observational astronomy (Topic 3) have somewhat smaller but still significant representation. This distribution reflects the modern state of physics research, where theoretical and computational approaches are highly prevalent, while maintaining strong experimental and observational components.

#### Mathematics
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: problem, method, paper, equation, results, equations, solution, model, function, problems 3200

Topic 2: paper, ring, let, number, results, theorem, proof, prove, ideal, new 1908

Topic 3: space, spaces, prove, set, functions, compact, result, paper, metric, study 2603

Topic 4: group, algebra, groups, algebras, finite, category, lie, prove, theory, paper 3541

Topic 5: graph, number, graphs, paper, set, manifolds, surfaces, prove, surface, curves 2542

The topic modeling analysis for Mathematics reveals distinct research areas and methodological approaches within the field. Topic 1, with keywords like "problem", "method", "equation", and "solution", appears to represent applied mathematics and numerical analysis, focusing on problem-solving and mathematical modeling (3,200 documents). This topic emphasizes the practical and computational aspects of mathematics.

Topic 2 (1,908 documents) centers on abstract algebra and number theory, as evidenced by terms like "ring", "number", "theorem", and "ideal". The presence of "proof" and "prove" indicates the rigorous theoretical nature of this research area, highlighting the fundamental role of mathematical proofs in establishing new results.

Topic 3, with 2,603 documents, focuses on analysis and topology, featuring terms like "space", "spaces", "compact", and "metric". This topic represents research in functional analysis, metric spaces, and related theoretical frameworks that form the foundation for many branches of modern mathematics.

Topic 4 emerges as the largest topic (3,541 documents) and concentrates on algebraic structures and category theory, with keywords including "group", "algebra", "category", and "lie". This suggests a strong research focus on abstract algebraic systems and their theoretical foundations, particularly in group theory and Lie algebras.

Topic 5 (2,542 documents) appears to focus on geometric and topological aspects of mathematics, with terms like "graph", "manifolds", "surfaces", and "curves". This topic represents research in geometric topology, graph theory, and differential geometry, highlighting the spatial and structural aspects of mathematical research.

The distribution of documents across these topics shows a relatively balanced research landscape in mathematics, with a slight emphasis on algebraic structures (Topic 4) and applied mathematics (Topic 1). The presence of proof-related terms across multiple topics underscores the fundamental importance of mathematical rigor and formal demonstration in all areas of mathematical research. This distribution reflects the modern state of mathematics, where theoretical foundations continue to be developed alongside practical applications and computational methods.

### Electrical Engineering and Systems Science
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: control, power, paper, energy, proposed, systems, model, based, design, problem 193

Topic 2: proposed, model, systems, method, approach, using, paper, control, based, results 289

Topic 3: proposed, detection, method, performance, model, based, paper, noise, speech, using 140

Topic 4: speech, model, image, data, network, models, images, using, proposed, performance 511

Topic 5: channel, proposed, communication, performance, algorithm, results, paper, wireless, systems, problem 204

The topic modeling analysis for Electrical Engineering and Systems Science reveals distinct research areas and methodological approaches within the field. Topic 1, with keywords like "control", "power", "energy", and "systems", appears to focus on power systems and control engineering (193 documents). This topic emphasizes the practical applications of electrical engineering in power management and control systems design.

Topic 2, with the largest document count (289), centers on systems engineering and methodological approaches, as evidenced by terms like "systems", "method", "approach", and "using". The frequent appearance of "proposed" suggests a strong focus on novel methodological contributions in this area.

Topic 3 (140 documents) concentrates on signal processing and detection systems, particularly in speech processing, as indicated by keywords like "detection", "noise", and "speech". This topic represents research in signal detection, noise reduction, and speech processing technologies.

Topic 4 emerges as the dominant topic with 511 documents, focusing on machine learning and computer vision applications, with keywords including "speech", "image", "data", "network", and "models". This suggests a significant research emphasis on deep learning applications in speech and image processing.

Topic 5 (204 documents) appears to focus on communications engineering, with terms like "channel", "communication", "wireless", and "performance" indicating research in wireless communications and network systems. The presence of "algorithm" and "performance" suggests a strong emphasis on optimization and system performance evaluation.

The distribution of documents across these topics shows a clear emphasis on machine learning and computer vision applications (Topic 4), followed by systems engineering methodologies (Topic 2). The relatively smaller representation in signal processing (Topic 3) and power systems (Topic 1) suggests these might be more specialized research areas. This distribution reflects the modern state of electrical engineering and systems science, where data-driven approaches and machine learning applications have become increasingly prominent.

#### Computer Science
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: paper, systems, design, performance, software, research, data, using, used, approach 3321

Topic 2: paper, model, problem, using, systems, based, method, linear, new, channel 2779

Topic 3: algorithm, problem, time, algorithms, graph, number, network, graphs, set, paper 2162

Topic 4: data, information, users, user, social, model, paper, different, based, study 1832

Topic 5: learning, model, method, models, neural, proposed, methods, data, using, performance 2586

The topic modeling analysis for Computer Science reveals distinct research areas and methodological approaches within the field. Topic 1, with keywords like "systems", "design", "performance", and "software", appears to focus on systems and software engineering (3,321 documents). This topic emphasizes the practical aspects of computer science, particularly in system design and implementation.

Topic 2 (2,779 documents) centers on theoretical and mathematical aspects of computer science, as evidenced by terms like "model", "problem", "linear", and "method". The presence of "channel" suggests applications in communication systems, while the combination with modeling terms indicates a focus on mathematical modeling and problem-solving approaches.

Topic 3, comprising 2,162 documents, focuses on algorithms and graph theory, featuring terms like "algorithm", "algorithms", "graph", and "graphs". This topic represents core computer science research in algorithmic design and analysis, particularly in graph-based problems and network algorithms.

Topic 4 emerges as a distinct topic (1,832 documents) concentrating on social computing and user interaction, with keywords including "data", "information", "users", and "social". This suggests a significant research focus on human-computer interaction, social media analysis, and user behavior studies.

Topic 5 (2,586 documents) clearly represents machine learning and neural networks, with terms like "learning", "neural", "models", and "method". This topic reflects the growing importance of artificial intelligence and machine learning in modern computer science research, with emphasis on neural network architectures and learning methodologies.

The distribution of documents across these topics shows a balanced research landscape in computer science, with a slight emphasis on systems and software engineering (Topic 1). The significant presence of machine learning (Topic 5) and theoretical computer science (Topic 2) reflects the field's current trends. The relatively smaller representation in social computing (Topic 4) suggests this might be a more specialized research area. This distribution effectively captures the diverse nature of modern computer science, spanning from theoretical foundations to practical applications and emerging technologies.

#### Quantitative Biology
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: protein, model, proteins, cell, structure, energy, using, dynamics, based, folding 386

Topic 2: model, dna, results, different, neurons, using, method, properties, data, models 289

Topic 3: data, models, brain, model, networks, analysis, methods, used, network, learning 479

Topic 4: model, species, population, networks, network, gene, dynamics, evolution, cell, growth 525

Topic 5: cancer, tree, data, trees, genes, gene, species, using, results, used 182

The topic modeling analysis for Quantitative Biology reveals distinct research areas and methodological approaches within the field. Topic 1, with keywords like "protein", "proteins", "cell", and "structure", focuses on molecular and cellular biology (386 documents). This topic emphasizes protein structure, dynamics, and folding, suggesting a strong focus on molecular biophysics and structural biology.

Topic 2 (289 documents) represents a diverse research area combining molecular biology and neuroscience, as evidenced by terms like "dna", "neurons", and "properties". The presence of modeling-related terms suggests this topic encompasses computational approaches to studying biological systems at both molecular and cellular levels.

Topic 3, comprising 479 documents, concentrates on computational neuroscience and network analysis, with keywords including "brain", "networks", and "learning". This topic reflects the growing importance of data-driven approaches and machine learning in understanding neural systems and brain function.

Topic 4 emerges as the largest topic (525 documents) focusing on ecological and evolutionary biology, with terms like "species", "population", "evolution", and "growth". The presence of "network" and "dynamics" suggests an emphasis on studying complex biological systems and their interactions at population and ecosystem levels.

Topic 5 (182 documents) appears to focus on genomics and phylogenetics, particularly in cancer research, as indicated by keywords like "cancer", "tree", "genes", and "species". This topic represents the intersection of molecular biology with evolutionary analysis, particularly in the context of disease studies.

The distribution of documents across these topics shows a balanced research landscape in quantitative biology, with slightly higher emphasis on ecological/evolutionary biology (Topic 4) and computational neuroscience (Topic 3). The relatively smaller representation in cancer genomics (Topic 5) suggests this might be a more specialized research area. This distribution reflects the modern state of quantitative biology, where computational and mathematical approaches are applied across different scales of biological organization, from molecular to ecosystem levels.

#### Economics

Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: effects, effect, electricity, demand, economic, economy, different, treatment, study, energy 46

Topic 2: equilibrium, model, agents, choice, economic, using, paper, network, models, project 50

Topic 3: model, data, paper, results, economic, countries, study, policy, models, performance 68

Topic 4: model, data, models, paper, using, treatment, time, methods, distribution, test 104

Topic 5: economic, growth, network, power, knowledge, gdp, global, complexity, central, quantum 40

The topic modeling analysis for Economics reveals distinct research areas and methodological approaches within the field. Topic 1 (46 documents) focuses on empirical economic analysis, particularly in energy and utility sectors, with keywords like "electricity", "demand", and "energy". The presence of "effects" and "treatment" suggests an emphasis on causal analysis and impact evaluation studies.

Topic 2 (50 documents) represents theoretical economic research, centered on equilibrium modeling and agent-based approaches. Keywords like "equilibrium", "agents", and "choice" indicate a focus on microeconomic theory and decision-making models, with network effects also being a significant consideration.

Topic 3, comprising 68 documents, appears to focus on macroeconomic policy and cross-country analysis. The combination of terms like "countries", "policy", and "economic" suggests research examining economic policies and their performance across different nations, supported by empirical data analysis.

Topic 4 emerges as the largest topic (104 documents) concentrating on econometric methods and statistical analysis. Keywords including "model", "data", "treatment", and "distribution" indicate a strong focus on quantitative research methods and statistical testing in economic analysis.

Topic 5 (40 documents) represents research on economic growth and complex systems, with terms like "growth", "network", "complexity", and "gdp" suggesting studies of economic development and interconnected economic systems. The unexpected presence of "quantum" might indicate interdisciplinary research or methodological crossover from physics.

The distribution of documents across these topics shows a clear emphasis on quantitative methods and empirical analysis (Topic 4), followed by policy research (Topic 3). The relatively smaller representation in growth and complexity studies (Topic 5) suggests this might be a more specialized research area. This distribution reflects the modern state of economics research, balancing theoretical frameworks with empirical analysis and policy applications.

#### Statistics

Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: data, model, models, using, approach, bayesian, results, based, used, methods 280

Topic 2: algorithm, data, models, methods, bayesian, distribution, method, carlo, monte, sampling 326

Topic 3: methods, model, models, kernel, approach, analysis, method, learning, used, statistical 93

Topic 4: data, statistical, model, models, analysis, statistics, methods, inference, time, spatial 200

Topic 5: data, model, method, proposed, estimation, methods, regression, paper, approach, models 442

The topic modeling analysis for Statistics reveals several distinct research areas and methodological approaches within the field. Topic 1 (280 documents) represents a broad focus on statistical modeling with a Bayesian emphasis, as indicated by keywords like "data", "model", and "bayesian". This topic suggests research combining theoretical frameworks with practical applications, using various methodological approaches.

Topic 2 (326 documents) concentrates on computational statistics and sampling methods, particularly Monte Carlo techniques. Keywords like "algorithm", "carlo", "monte", and "sampling" indicate a strong focus on computational approaches to statistical problems, with Bayesian methods also playing a significant role.

Topic 3 (93 documents) appears focused on advanced statistical learning methods, particularly kernel-based approaches. The presence of terms like "kernel", "learning", and "statistical" suggests research at the intersection of statistics and machine learning, emphasizing methodological developments.

Topic 4 (200 documents) represents traditional statistical analysis and inference, with a particular focus on temporal and spatial applications. Keywords including "statistical", "inference", "time", and "spatial" indicate research involving various types of statistical analysis across different domains.

Topic 5 emerges as the largest topic (442 documents) focusing on statistical methodology and estimation techniques. Terms like "estimation", "regression", and "proposed" suggest an emphasis on developing and applying statistical methods, particularly in the context of regression analysis and model estimation.

The distribution of documents across these topics shows a strong emphasis on methodological development (Topic 5) and computational approaches (Topic 2). The relatively smaller representation in kernel-based methods (Topic 3) suggests this might be a more specialized research area. This distribution reflects the modern state of statistics, balancing theoretical developments with computational methods and practical applications.

#### Quantitative Finance
Analyzing 'summary' column
Top 10 words for each of the 5 topics:
Topic 1: market, financial, stock, data, price, trading, model, models, time, learning 253

Topic 2: model, volatility, price, models, pricing, stochastic, option, risk, process, method 271

Topic 3: market, model, risk, portfolio, financial, problem, time, prices, strategies, equilibrium 125

Topic 4: risk, trade, countries, economic, analysis, time, market, network, firm, financial 79

Topic 5: model, risk, financial, distribution, data, time, market, wealth, models, variables 93

The topic modeling analysis for Quantitative Finance reveals distinct research areas within the field. Topic 1 (253 documents) focuses on market analysis and trading, with keywords like "market", "stock", "price", and "trading" indicating research on stock markets and trading behavior. The presence of "learning" suggests the application of machine learning techniques to financial market analysis.

Topic 2 (271 documents) concentrates on financial modeling, particularly in derivatives and risk analysis. Keywords such as "volatility", "pricing", "stochastic", and "option" point to research in option pricing models and stochastic processes, representing the mathematical foundations of quantitative finance.

Topic 3 (125 documents) represents research in portfolio management and market equilibrium. Terms like "portfolio", "equilibrium", and "strategies" suggest studies focusing on portfolio optimization, market equilibrium models, and investment strategies, bridging theoretical finance with practical applications.

Topic 4 (79 documents) appears to focus on broader economic and market risk analysis. The combination of terms like "trade", "countries", "economic", and "network" indicates research examining international trade, market networks, and firm-level analysis, with an emphasis on risk assessment.

Topic 5 (93 documents) emphasizes financial modeling and risk analysis with a focus on statistical approaches. Keywords including "distribution", "variables", and "wealth" suggest research involving statistical modeling of financial phenomena and wealth distribution, combining quantitative methods with financial applications.

The distribution of documents across these topics shows a primary focus on financial modeling (Topic 2) and market analysis (Topic 1), which together account for the majority of the documents. The smaller representation in international trade and network analysis (Topic 4) suggests this might be a more specialized research area. This distribution reflects the field's emphasis on mathematical modeling and market analysis, while also maintaining coverage of broader economic and strategic considerations.


### Named Entity Recognition (NER)
The Named Entity Recognition (NER) analysis across different scientific fields reveals distinct patterns in how various types of entities are used in research papers. This analysis helps understand the characteristic ways that different disciplines reference and discuss entities like numbers, organizations, people, and concepts. By examining the most common named entities in each field, we can gain insights into the writing styles, methodological approaches, and key focus areas that define different academic disciplines.

The following sections present detailed NER analyses for Physics and Mathematics, highlighting both the commonalities and unique characteristics in how these fields employ different types of entities in their research communications. This analysis provides an interesting lens through which to understand the linguistic and conceptual frameworks that shape scientific discourse in these fields.


#### Physics
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 7192
first (ORDINAL): 3544
one (CARDINAL): 3437
three (CARDINAL): 2042
second (ORDINAL): 1467
zero (CARDINAL): 932
four (CARDINAL): 825
linear (ORG): 753
third (ORDINAL): 372
five (CARDINAL): 284

The Named Entity Recognition analysis for the Physics corpus reveals interesting patterns in the types of entities mentioned in physics research papers. The most frequent entities are predominantly numerical indicators, with cardinal numbers and ordinal numbers dominating the top entities.

Cardinal numbers feature prominently, with "two" being the most frequent (7,192 occurrences), followed by "one" (3,437 occurrences), "three" (2,042 occurrences), "zero" (932 occurrences), "four" (825 occurrences), and "five" (284 occurrences). This high frequency of cardinal numbers reflects the quantitative nature of physics research, where numerical values and quantities play a crucial role in describing phenomena, measurements, and results.

Ordinal numbers also appear frequently, with "first" (3,544 occurrences), "second" (1,467 occurrences), and "third" (372 occurrences) being common. These ordinal numbers likely indicate sequential processes, ordered relationships, or prioritization in physics research methodologies and findings.

Interestingly, "linear" appears as the only organizational entity (ORG) in the top 10, with 753 occurrences. This could reflect the importance of linear systems, linear algebra, or linear relationships in physics research, though its classification as an organizational entity might warrant further investigation.

The dominance of numerical entities in the physics corpus aligns with the field's mathematical and quantitative nature. The relative scarcity of other entity types (such as persons, locations, or dates) in the top entities suggests that physics research papers tend to focus more on abstract concepts and numerical relationships rather than specific people, places, or temporal references.

#### Mathematics
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 2703
one (CARDINAL): 1595
first (ORDINAL): 1480
second (ORDINAL): 861
linear (ORG): 656
three (CARDINAL): 613
zero (CARDINAL): 585
abelian (NORP): 375
four (CARDINAL): 206
third (ORDINAL): 154

The Named Entity Recognition analysis for the Mathematics corpus shows similar patterns to Physics, with numerical entities dominating the most frequent occurrences, though with some notable differences in distribution and the presence of field-specific terms.

Cardinal numbers remain prominent, with "two" being the most frequent (2,703 occurrences), followed by "one" (1,595 occurrences), "three" (613 occurrences), "zero" (585 occurrences), and "four" (206 occurrences). While the pattern of cardinal numbers is similar to the Physics corpus, the frequencies are notably lower, possibly reflecting a smaller corpus size or different writing patterns in mathematical research.

Ordinal numbers also feature significantly, with "first" (1,480 occurrences), "second" (861 occurrences), and "third" (154 occurrences) appearing frequently. These ordinal numbers likely indicate sequential proofs, theorem statements, or ordered mathematical relationships, which are fundamental to mathematical writing.

Two interesting entities distinguish the Mathematics corpus from Physics. "Linear" appears as an organizational entity (656 occurrences), similar to the Physics corpus, highlighting the importance of linear concepts across both fields. Uniquely, "Abelian" appears as a NORP (Nationality or Religious or Political group) entity with 375 occurrences. While its classification as NORP might be technically correct due to its capitalization (being named after mathematician Niels Henrik Abel), its frequent appearance reflects the importance of Abelian groups and related concepts in mathematics.

The overall entity distribution in Mathematics shows a strong focus on numerical and mathematical-specific terms, with fewer general entities. This aligns with the abstract and theoretical nature of mathematical research, where concepts and relationships often take precedence over physical or empirical references.

#### Electrical Engineering and Systems Science
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 358
first (ORDINAL): 221
one (CARDINAL): 168
three (CARDINAL): 114
second (ORDINAL): 84
linear (ORG): 82
cnn (ORG): 43
irs (ORG): 39
recent years (DATE): 33
four (CARDINAL): 25

The Named Entity Recognition analysis for the Electrical Engineering and Systems Science corpus reveals some notable differences from the Physics and Mathematics corpora, while maintaining certain common patterns in the types of entities identified.

Cardinal numbers continue to be prominent, though with lower absolute frequencies due to the smaller corpus size. "Two" leads with 358 occurrences, followed by "one" (168 occurrences), "three" (114 occurrences), and "four" (25 occurrences). While this maintains the pattern seen in other fields, the relative proportions are similar, indicating consistent usage of numerical quantifiers across scientific disciplines.

Ordinal numbers also maintain their significance, with "first" (221 occurrences) and "second" (84 occurrences) appearing frequently. These likely serve similar functions as in other fields, marking sequence and priority in technical processes and methodological steps.

Uniquely to this corpus, several organizational entities (ORG) appear prominently. "Linear" continues its presence (82 occurrences) as seen in other fields, but is joined by "CNN" (43 occurrences) and "IRS" (39 occurrences). The appearance of "CNN" likely refers to Convolutional Neural Networks, reflecting the field's engagement with modern machine learning techniques. "IRS" could refer to various technical terms in the field (such as Infrared Systems or Internal Reference System), though this would require further context for confirmation.

A temporal entity appears in the top entities with "recent years" (33 occurrences), which is unique among the three analyzed corpora. This might indicate a greater emphasis on current developments and technological progress in electrical engineering and systems science, reflecting the field's rapid evolution and practical applications.

The entity distribution in this corpus suggests a field that combines mathematical precision (through numerical entities) with specific technical terminology and a stronger connection to contemporary developments. The presence of machine learning-related terms (CNN) particularly highlights the field's integration of modern computational approaches.

#### Computer Science
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 3358
first (ORDINAL): 2073
one (CARDINAL): 1936
three (CARDINAL): 1087
second (ORDINAL): 761
linear (ORG): 533
four (CARDINAL): 397
cnn (ORG): 251
recent years (DATE): 233
five (CARDINAL): 192

The Named Entity Recognition analysis of the Computer Science corpus reveals patterns that reflect both the mathematical foundations and modern technological focus of the field.

Cardinal numbers dominate the most frequent entities, with "two" leading at 3,358 occurrences, followed by "one" (1,936 occurrences), "three" (1,087 occurrences), "four" (397 occurrences), and "five" (192 occurrences). This extensive use of cardinal numbers suggests a strong quantitative aspect in computer science research, possibly relating to algorithm complexity, system components, or experimental results.

Ordinal numbers appear prominently, with "first" (2,073 occurrences) and "second" (761 occurrences) being particularly frequent. These likely indicate sequential steps in algorithms, priority ordering, or comparative analyses, which are fundamental to computer science methodology.

Organizational entities show interesting patterns. "Linear" appears frequently (533 occurrences), consistent with other technical fields, likely referring to linear algorithms, complexity, or mathematical relationships. "CNN" (251 occurrences) indicates the significant presence of Convolutional Neural Networks in computer science research, reflecting the field's strong engagement with machine learning and artificial intelligence.

The temporal entity "recent years" (233 occurrences) suggests an emphasis on current developments and technological progress, appropriate for a rapidly evolving field. This temporal reference might indicate discussions of advances in technology, emerging research trends, or comparative analyses with previous approaches.

The entity distribution in the Computer Science corpus reveals a field that heavily employs quantitative descriptions while maintaining strong connections to contemporary technological developments, particularly in machine learning. The frequencies are generally higher than in other fields, possibly indicating a larger corpus size or more detailed technical descriptions in computer science papers.

#### Quantitative Biology
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 589
one (CARDINAL): 295
first (ORDINAL): 236
three (CARDINAL): 173
second (ORDINAL): 105
four (CARDINAL): 60
linear (ORG): 52
five (CARDINAL): 36
six (CARDINAL): 31
ec (ORG): 22

The Named Entity Recognition analysis of the Quantitative Biology corpus reveals patterns that reflect the field's quantitative nature while showing some distinct characteristics from other disciplines.

Cardinal numbers are the most prevalent entities, with "two" leading at 589 occurrences, followed by "one" (295 occurrences), "three" (173 occurrences), "four" (60 occurrences), "five" (36 occurrences), and "six" (31 occurrences). The frequency of cardinal numbers, while lower than in Computer Science, indicates the importance of numerical precision in quantitative biological research, possibly relating to experimental groups, sample sizes, or biological measurements.

Ordinal numbers also feature prominently, with "first" (236 occurrences) and "second" (105 occurrences) appearing frequently. These likely indicate sequence ordering in biological processes, experimental procedures, or priority relationships in research findings.

Two organizational entities appear in the most frequent entities: "linear" (52 occurrences) and "EC" (22 occurrences). The presence of "linear" suggests the use of linear models or relationships in biological systems, though at a lower frequency than in other technical fields. "EC" likely refers to Enzyme Commission numbers, a numerical classification scheme for enzymes, reflecting the field's connection to biochemistry and molecular biology.

Notably, the overall frequencies of entities in Quantitative Biology are lower compared to Computer Science and Electrical Engineering, which might indicate either a smaller corpus size or less reliance on numerical descriptions. The entity distribution suggests a field that combines quantitative analysis with biological systems, though with less emphasis on contemporary technological terms compared to other technical fields.

#### Economics
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 95
one (CARDINAL): 60
first (ORDINAL): 54
second (ORDINAL): 28
three (CARDINAL): 25
china (GPE): 13
european (NORP): 13
india (GPE): 11
monthly (DATE): 9
linear (ORG): 9

The Named Entity Recognition analysis of the Economics corpus reveals distinct patterns that reflect the field's focus on economic systems, geographical regions, and quantitative analysis, though with notably lower overall frequencies compared to the previously analyzed fields.

Cardinal numbers remain the most frequent entities, with "two" (95 occurrences), "one" (60 occurrences), and "three" (25 occurrences) leading the cardinal numbers. While following the pattern seen in other fields, the significantly lower frequencies suggest either a smaller corpus size or less reliance on numerical descriptions in economic research.

Ordinal numbers maintain their importance, with "first" (54 occurrences) and "second" (28 occurrences) appearing frequently. These likely indicate sequential analyses, priority rankings, or ordered economic phenomena, though again at lower frequencies than in other fields.

A distinctive feature of the Economics corpus is the prominent presence of geographical and demographic entities. "China" (13 occurrences) and "India" (11 occurrences) appear as significant geographical entities (GPE), while "European" (13 occurrences) appears as a demographic or national/regional descriptor (NORP). This suggests a strong focus on international economic analysis and regional economic studies.

Temporal references appear through "monthly" (9 occurrences as DATE), indicating the importance of time-series analysis and periodic economic measurements in the field. The organizational entity "linear" (9 occurrences) suggests the use of linear models or relationships in economic analysis, though at a much lower frequency than in other technical fields.

The entity distribution in Economics reveals a field that combines quantitative analysis with strong geographical and temporal components, reflecting its focus on studying economic systems across different regions and time periods. The lower overall frequencies compared to other fields might indicate different corpus characteristics or writing styles in economic research.


#### Statistics
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 445
one (CARDINAL): 202
first (ORDINAL): 158
three (CARDINAL): 116
linear (ORG): 97
second (ORDINAL): 89
four (CARDINAL): 43
abc (ORG): 31
lasso (PERSON): 22
recent years (DATE): 21

The Named Entity Recognition analysis of the Statistics corpus reveals patterns that highlight the field's strong focus on quantitative methods and mathematical concepts, with notably higher frequencies compared to Economics but lower than Computer Science.

Cardinal numbers dominate the most frequent entities, with "two" (445 occurrences), "one" (202 occurrences), "three" (116 occurrences), and "four" (43 occurrences) appearing frequently. These high frequencies reflect the fundamental role of numerical analysis and quantitative methods in statistical research.

Ordinal numbers also feature prominently, with "first" (158 occurrences) and "second" (89 occurrences) appearing frequently. These likely indicate sequence ordering in statistical procedures, methodological steps, or priority relationships in research findings.

Several organizational entities appear in the most frequent entities: "linear" (97 occurrences) and "abc" (31 occurrences). The high frequency of "linear" suggests the extensive use of linear models and relationships in statistical analysis, while "abc" might refer to specific statistical methods or algorithms.

The presence of "lasso" (22 occurrences) as a PERSON entity is interesting, though this likely represents the LASSO (Least Absolute Shrinkage and Selection Operator) statistical method being misclassified as a person name. This highlights both the prominence of regularization methods in statistics and the occasional challenges in entity classification.

Temporal references appear through "recent years" (21 occurrences as DATE), indicating discussions of current trends and developments in statistical research. This suggests a field that actively reflects on its recent progress and evolving methodologies.

The entity distribution in Statistics reveals a field heavily focused on quantitative methods and mathematical concepts, with strong emphasis on numerical relationships and methodological approaches. The relatively high frequencies of mathematical and methodological terms, compared to fields like Economics, reflect the technical and analytical nature of statistical research.

#### Quantitative Finance
Analyzing 'summary' column:

Most common named entities:
two (CARDINAL): 167
one (CARDINAL): 130
first (ORDINAL): 124
three (CARDINAL): 65
european (NORP): 56
second (ORDINAL): 47
daily (DATE): 41
linear (ORG): 28
zero (CARDINAL): 28
american (NORP): 26

The Named Entity Recognition analysis of the Quantitative Finance corpus reveals patterns that highlight the field's intersection of financial markets, mathematical methods, and geographical considerations, with frequencies generally lower than both Statistics and Computer Science.

Cardinal numbers feature prominently among the most frequent entities, with "two" (167 occurrences), "one" (130 occurrences), "three" (65 occurrences), and "zero" (28 occurrences) appearing frequently. These frequencies reflect the importance of numerical analysis in quantitative finance, though at lower levels than in pure Statistics.

Ordinal numbers also appear frequently, with "first" (124 occurrences) and "second" (47 occurrences) suggesting the common use of sequential analysis or prioritization in financial methodologies and results presentation.

Geographic and cultural entities feature notably in this field, with "european" (56 occurrences) and "american" (26 occurrences) as NORP (Nationality or Religious or Political group) entities appearing among the most frequent. This highlights the importance of different market regions and financial systems in quantitative finance research.

The temporal dimension is represented through "daily" (41 occurrences as DATE), indicating the significance of daily market data and time-series analysis in financial research. This frequency suggests regular time-interval analysis is central to quantitative finance methodologies.

The organizational entity "linear" (28 occurrences) appears less frequently than in Statistics or Economics, though still notably, suggesting the use of linear models in financial analysis, albeit potentially with less emphasis than in other quantitative fields.

The entity distribution in Quantitative Finance reveals a field that combines mathematical and quantitative methods with strong geographical considerations, reflecting its focus on analyzing financial markets across different regions. The relatively lower frequencies compared to Statistics suggest potentially different corpus characteristics or a broader distribution of entity types in financial research.

### Sentiment Analysis

Average Sentiment in SUmmary for each category:
- Computer Science: 0.087
- Quantitative Biology: 0.073
- Physics: 0.093
- Mathematics: 0.071
- Electrical Engineering and Systems Science: 0.077
- Economics: 0.081
- Statistics: 0.070
- Quantitative Finance: 0.076

The sentiment analysis reveals subtle but notable variations in emotional tone across different scientific disciplines. Computer Science shows the highest average sentiment score (0.087), followed closely by Physics (0.093), suggesting these fields tend to use slightly more positive language in their research summaries. This could reflect the optimistic nature of technological advancement and physical discoveries, or the tendency to emphasize positive outcomes and improvements in these fields.

Mathematics and Statistics demonstrate the lowest sentiment scores (0.071 and 0.070 respectively), indicating a more neutral tone in their research communications. This aligns with the traditionally objective and formal nature of mathematical discourse, where emotional language is typically minimized in favor of precise, technical expression.

The applied fields - Electrical Engineering and Systems Science (0.077) and Quantitative Finance (0.076) - show moderate sentiment scores, falling near the middle of the range. This might reflect a balance between technical objectivity and practical applications, where positive outcomes and real-world impacts are discussed alongside methodological details.

Economics (0.081) and Quantitative Biology (0.073) present interesting contrasts, with Economics showing relatively higher positivity in its language. This could stem from discussions of positive economic outcomes, growth, or improvements in economic conditions, while Quantitative Biology maintains a more neutral tone typical of life sciences research.

Overall, the sentiment scores across all disciplines remain relatively close to neutral (ranging from 0.070 to 0.093), which is characteristic of academic writing. The small variations, while subtle, may reflect underlying differences in how different fields communicate their research findings and the balance they strike between objective reporting and highlighting positive outcomes or advances in their respective domains.

## Feature Engineering

Feature engineering plays a crucial role in transforming raw text data into structured numerical features that machine learning models can effectively process. In this study, we implemented a comprehensive feature engineering pipeline that combines traditional NLP techniques with modern deep learning approaches to capture both semantic and statistical characteristics of the scientific papers. The following sections detail the various feature extraction and transformation techniques applied:

### Tokenise and Lemmatise using BERT
The tokenization and lemmatization process was implemented using BERT (Bidirectional Encoder Representations from Transformers), leveraging its advanced contextual understanding of text. This approach offers several advantages over traditional lemmatization methods:

First, BERT's bidirectional nature allows it to consider the full context when processing each word, leading to more accurate lemmatization that accounts for word sense and usage. The model processes text through its transformer architecture, which helps maintain semantic relationships while reducing words to their base forms.

The implementation handles text processing in batches to optimize computational efficiency, with automatic memory management for CUDA-enabled systems. This batch processing approach allows for efficient processing of large text corpora while maintaining consistent quality. The system automatically adapts to available computational resources, utilizing GPU acceleration when available and gracefully falling back to CPU processing when necessary.

Special attention was paid to maintaining text integrity during processing. The system preserves important linguistic features while removing unnecessary tokens and standardizing text representation. This preprocessing step was applied to all text fields (titles, summaries, comments, and author lists) to ensure consistent treatment across the dataset.

The lemmatization process helps reduce vocabulary size and standardize word forms, making subsequent analysis more reliable. This is particularly important for scientific text, where technical terms and their variants need to be properly normalized while preserving their semantic meaning.


### Vectorise using BERT
Text vectorization was implemented using BERT embeddings to capture rich semantic representations of the scientific papers. The vectorization process transforms text into high-dimensional numerical vectors that encode contextual meaning and relationships between words.

The implementation uses the 'bert-base-uncased' model to generate embeddings for each text field (titles, summaries, comments, and author lists). BERT's transformer architecture processes text bidirectionally, allowing it to capture complex contextual relationships and nuances in scientific writing. The model generates 768-dimensional vectors for each text input, providing a dense representation of semantic content.

The vectorization process includes several optimizations for handling large datasets. Batch processing is implemented to efficiently handle large volumes of text, while automatic memory management enables GPU acceleration when available. The system gracefully falls back to CPU processing when needed, ensuring reliable processing regardless of hardware constraints. Additionally, truncation and padding mechanisms are employed to properly handle variable-length inputs, maintaining consistent vector dimensions across all samples.

Special attention was paid to maintaining consistent vector representations across different text fields. The process handles missing or malformed text appropriately, ensuring robust feature generation even with imperfect input data. The resulting embeddings capture both local syntactic patterns and broader semantic relationships in the scientific text.

These BERT embeddings provide a rich foundation for downstream machine learning tasks, encoding complex relationships between scientific concepts, methodologies, and findings. The high-dimensional nature of these vectors allows the capture of subtle variations in meaning that are particularly important in distinguishing between different scientific disciplines.


### Word Count
Word count analysis was performed on all text fields (titles, summaries, comments, and author lists) to quantify text length and verbosity. The implementation processes text in efficient batches of 1000 samples, splitting text on whitespace to count individual words.

For each text field, a new feature column is created containing the raw word count. This provides a basic but important metric of text length that can help distinguish between different types of scientific papers. For example, theoretical papers may tend toward longer, more detailed summaries compared to experimental papers.

The word counting process incorporates several robustness features to ensure reliable processing. The system automatically handles non-string inputs by converting them to strings before processing, and properly manages empty or malformed text to prevent errors. To maintain consistency, the counting methodology is standardized across all text fields in the dataset. Additionally, the implementation optimizes performance through batch processing, allowing efficient handling of large volumes of text data.

While simple compared to semantic analysis, word counts provide valuable signals about paper structure and writing style. They can reveal patterns in how different scientific disciplines structure their abstracts and summaries, potentially helping distinguish between fields that favor concise versus detailed descriptions.

The word count features complement the more sophisticated semantic features by providing explicit quantification of text length. This can be particularly useful when combined with complexity metrics and semantic embeddings to build a complete picture of paper characteristics.


### Named Entity Recognition (NER)
Named Entity Recognition was implemented using spaCy's pre-trained "en_core_web_sm" model to identify and quantify different types of entities in the text. The implementation processes text in batches of 100 samples for optimal performance, analyzing all text fields including titles, summaries, comments, and author lists.

For each text field, the system identifies and counts entities across multiple categories including person names, organizations, locations, dates, and other domain-specific entities. The NER process creates separate count features for each entity type, allowing for fine-grained analysis of the content focus. This batch processing approach ensures efficient handling of large text volumes while maintaining consistent entity recognition quality.

The implementation includes robust error handling and type conversion, automatically converting all text to strings before processing. Entity counts are aggregated per document and stored in dedicated columns following a standardized naming convention (e.g., 'title_ner_PERSON_count', 'summary_ner_ORG_count'). This structured approach enables detailed analysis of how different scientific disciplines use various types of named entities in their papers.

The entity recognition provides valuable insights into the domain-specific terminology and focus areas of different scientific fields. For example, it can reveal patterns in how frequently papers reference specific organizations, locations, or key figures in their field. These entity patterns serve as important features for distinguishing between different scientific disciplines, complementing the semantic and statistical features extracted through other methods.

### Sentiment Analysis 
Sentiment analysis was performed using TextBlob to calculate polarity scores ranging from -1 (negative) to 1 (positive) for all text fields in the dataset. The implementation processes text in batches of 1000 samples for efficient computation, analyzing titles, summaries, comments, and author lists.

For each text field, the system calculates a sentiment polarity score that captures the overall emotional tone of the text. The process automatically handles text preprocessing by converting all inputs to strings before analysis. The sentiment scores are stored in dedicated columns following a standardized naming convention (e.g., 'title_sentiment', 'summary_sentiment').

While scientific papers generally aim for neutral and objective language, subtle variations in sentiment can provide valuable signals for classification. For example, certain fields may tend toward more positive language when describing results, while others maintain stricter neutrality. The sentiment features complement the semantic and statistical features by capturing these subtle emotional undertones in scientific writing.

The implementation includes robust error handling and parallel processing optimizations through the TOKENIZERS_PARALLELISM environment variable. This ensures reliable and efficient sentiment analysis even with large volumes of text data. The resulting sentiment scores provide an additional dimension for distinguishing between different scientific disciplines based on their characteristic writing styles and emotional expression patterns.

### Text Complexity Metrics
Text complexity analysis was performed using the Automated Readability Index (ARI) to quantify the sophistication level of all text fields in the dataset. The implementation processes text in batches of 1000 samples, analyzing titles, summaries, comments, and author lists.

For each text field, the system calculates an ARI score based on character count, word count, and sentence count using the formula: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43. The scores are bounded between 1 and 14, with higher scores indicating more complex text. The implementation includes robust handling of edge cases, such as texts with no sentence-ending punctuation, and automatically converts all inputs to strings before analysis.

The ARI scores provide valuable insights into the linguistic complexity patterns across different scientific disciplines. For example, certain fields may consistently use more complex language in their titles or summaries compared to others. The implementation creates separate complexity features for each text field (e.g., 'title_ari', 'summary_ari'), enabling detailed analysis of how writing complexity varies across different parts of scientific papers.

These complexity metrics complement the semantic and statistical features by providing explicit quantification of text sophistication. When combined with other features like word counts and entity recognition, they help build a comprehensive picture of the writing styles characteristic of different scientific fields.

### Feature Normalization
Feature normalization was implemented using scikit-learn's MinMaxScaler to transform all numerical features to a consistent [0,1] range. The normalization process excludes the categorical target variable ('category') and split designation columns to preserve their original values. This scaling ensures that all numerical features contribute proportionally to model training, preventing features with larger absolute values from dominating the learning process.

The implementation automatically identifies numerical columns by their data type (int64 or float64) and applies the scaler transformation. The MinMaxScaler preserves zero values and the shape of the original distribution while bounding all values between 0 and 1. This normalization is particularly important for our feature set which combines dense BERT embeddings (768 dimensions per text field) with scalar metrics like word counts, entity counts, sentiment scores, and readability indices that exist on very different scales.

The final feature set provides a rich representation of each paper, combining normalized semantic embeddings that capture complex meaning with interpretable metrics that quantify specific textual characteristics. The careful normalization process ensures that all features contribute meaningfully to the classification task while maintaining their relative relationships. The scaler object is preserved to enable consistent transformation of new data during model deployment.

## Experimentation (1st Run)
The initial experimental phase evaluated seven different model architectures on the original dataset to establish baseline performance metrics and identify promising approaches. The models ranged from simple linear classifiers to complex deep neural networks, allowing us to assess the relationship between model complexity and classification performance.

The experiments were conducted using a standardized training pipeline with consistent hyperparameters across models where applicable. All models were trained on the same feature set, including the BERT embeddings, word counts, entity counts, sentiment scores, and complexity metrics described in previous sections. The training process utilized early stopping with a patience of 5 epochs to prevent overfitting, and model checkpointing to save the best performing weights based on validation loss.

Each model was evaluated using stratified 5-fold cross-validation to ensure robust performance assessment across different data splits. The evaluation metrics include accuracy, precision, recall, and F1-score, calculated both as weighted averages (accounting for class imbalance) and macro averages (treating all classes equally). This comprehensive evaluation framework allows for detailed comparison of model performance across different aspects of the classification task.

The following table provides a comparative summary of the performance of all seven models evaluated in this study. The metrics include accuracy, and weighted averages of precision, recall, and F1-score.

| Model                                      | Accuracy | Precision  | Recall    | F1-Score  | Precision | Recall    | F1-Score  | 
|                                            |          | (weighted) | (weighted)| (weighted)| (macro)   | (macro)   | (macro)   |
|--------------------------------------------|----------|------------|-----------|-----------|-----------|-----------|-----------|
| $M_0$: Logistic Regression                 | 0.69     | 0.81       | 0.69      | 0.73      | 0.47      | 0.64      | 0.49      |
| $M_1$: Shallow Artificial Neural Network   | 0.70     | 0.82       | 0.70      | 0.73      | 0.48      | 0.67      | 0.50      |
| $M_2$: Deep Artificial Neural Network      | 0.56     | 0.73       | 0.56      | 0.56      | 0.37      | 0.56      | 0.33      |
| $M_3$: Recurrent Neural Network (RNN)      | 0.66     | 0.81       | 0.66      | 0.70      | 0.46      | 0.63      | 0.46      |
| $M_4$: Convolutional Neural Network (CNN)  | 0.67     | 0.80       | 0.67      | 0.71      | 0.45      | 0.63      | 0.47      |
| $M_5$: Autoencoder Neural Network          | 0.59     | 0.76       | 0.59      | 0.60      | 0.38      | 0.54      | 0.35      |
| $M_6$: Residual Neural Network (ResNet)    | 0.66     | 0.80       | 0.66      | 0.69      | 0.45      | 0.63      | 0.45      |

The experimental results reveal several key patterns and insights:

1. Model Performance Overview:
The shallow models (Logistic Regression and Shallow ANN) demonstrated surprisingly strong performance compared to their deeper counterparts. The Shallow Artificial Neural Network achieved the highest accuracy at 0.70, followed closely by Logistic Regression at 0.69. This suggests that the classification task may not require deep architectural complexity to capture the underlying patterns.

2. Deep Architecture Performance:
Interestingly, the Deep Artificial Neural Network ($M_2$) showed the poorest performance across all metrics, with an accuracy of only 0.56. This unexpected result might indicate potential overfitting or difficulties in training the deeper architecture with the given dataset size and feature distribution. The other deep architectures (RNN, CNN, ResNet) performed moderately better but still couldn't surpass the simpler models.

3. Precision-Recall Trade-off:
All models showed higher weighted precision compared to their recall scores, suggesting a tendency to be more conservative in their predictions. The gap between precision and recall is particularly noticeable in the Logistic Regression (0.81 vs 0.69) and Shallow ANN (0.82 vs 0.70), indicating these models might be better at avoiding false positives at the cost of missing some positive cases.

4. Macro vs Weighted Metrics:
The substantial difference between macro and weighted metrics (e.g., macro F1-scores around 0.45-0.50 vs weighted F1-scores around 0.70-0.73) indicates significant class imbalance in the dataset. This suggests that models perform better on more prevalent classes but struggle with minority classes.

5. Model Stability:
The CNN, RNN, and ResNet showed very similar performance patterns (accuracies between 0.66-0.67), suggesting that the sequential or spatial features these architectures are designed to capture may not provide significant advantages for this particular classification task.

### Model Selection for Next Iteration
Based on the results from the first experimental run, we selected the top 3 performing models for further evaluation with a balanced dataset.

The Shallow Artificial Neural Network ($M_1$) emerged as the best overall performer with 0.70 accuracy, achieving the highest weighted precision (0.82) and strong F1-score (0.73). This model demonstrated an excellent balance between architectural complexity and performance metrics, making it a prime candidate for further optimization.

Logistic Regression ($M_0$) proved to be a surprisingly strong contender, achieving 0.69 accuracy with strong weighted precision (0.81) and F1-score (0.73). As the simplest model in our evaluation, its competitive performance suggests that linear decision boundaries might be sufficient for significant portions of our classification task.

The Convolutional Neural Network ($M_4$) showed moderate but promising performance with 0.67 accuracy, good weighted precision (0.80), and F1-score (0.71). While both RNN and ResNet achieved similar metrics, we selected the CNN for further evaluation due to its slightly better F1-score, generally faster training time, and lower computational requirements.

The selection criteria for these models encompassed multiple factors including overall performance metrics (accuracy, precision, recall, F1-score), computational efficiency, model simplicity, interpretability, and potential for improvement with balanced data. By choosing a mix of simple (Logistic Regression), moderate (Shallow ANN), and complex (CNN) architectures, we aimed to provide a comprehensive evaluation framework for the balanced dataset phase of our experiment.

## Dataset Balancing and Preparation for Second Experimental Phase
After analyzing the results from the first experimental phase, we identified class imbalance as a significant factor affecting model performance. To address this, we performed comprehensive dataset balancing using the preprocessing and feature engineering pipelines developed in our initial phase.

The data preprocessing pipeline included several key steps. First, we performed text cleaning and normalization, followed by lowercasing of all text fields. We then removed special characters and numbers from the text, standardized the text encoding to UTF-8, and performed tokenization of all text fields.

Following the preprocessing steps, we applied our feature engineering pipeline to extract meaningful features from the data. We generated BERT embeddings for the title, summary, comment, and author fields. We also calculated word count statistics and Named Entity Recognition (NER) counts. Additionally, we performed sentiment analysis to generate sentiment scores and computed text complexity metrics using ARI scores.

These preprocessing and feature engineering steps were carefully maintained from the first experimental phase to ensure consistency and comparability of results. The key difference in this phase was the implementation of balanced sampling to address the class distribution issues identified earlier.

The balanced dataset consisted of 21,152 total observations, which were split into 13,264 training samples, 5,768 validation samples, and 2,120 test samples. Each category contained exactly 2,644 observations to ensure balanced representation across classes.

To ensure a fair comparison between models, we carefully balanced the dataset using random undersampling. This technique was chosen over oversampling methods to avoid potential overfitting that could arise from synthetic data generation. The balanced dataset contains 21,152 total observations, with exactly 2,644 samples per category, ensuring equal representation across all classes.

The dataset was split into training (62.7%), validation (27.3%), and test (10%) sets, maintaining the balanced class distribution across all splits. This resulted in 13,264 training samples, 5,768 validation samples, and 2,120 test samples. The split ratios were chosen to provide sufficient data for model training while retaining a substantial validation set for model selection and hyperparameter tuning.

The preprocessing pipeline remained consistent with the original dataset to maintain comparability of results. This included standardization of numerical features, encoding of categorical variables, and handling of missing values. Feature engineering steps were also kept identical to ensure that any performance improvements could be attributed to the balanced data rather than changes in the feature space.

The balanced dataset preparation phase was crucial for addressing the class imbalance issues identified in the first experimental run. By equalizing the class distributions, we aimed to reduce bias towards majority classes and improve model performance on minority classes. This balancing also made macro and weighted metrics more directly comparable, allowing for better assessment of model performance across all classes. Additionally, the balanced dataset provided a more reliable evaluation of each model's true discriminative capabilities by ensuring all classes were equally represented in the training data.

This balanced dataset served as the foundation for our second experimental run, allowing us to evaluate whether the selected models could achieve better performance when trained on equally represented classes.


## Experimentation (2nd Run)
In the second experimental run, we evaluated the three selected models (Logistic Regression, Shallow ANN, and CNN) on the balanced dataset. This phase aimed to assess whether addressing the class imbalance would lead to improved model performance and more consistent results across classes.

Each model was trained using identical preprocessing and feature engineering pipelines from the first run, with the key difference being the balanced training data. We maintained consistent hyperparameter settings from the first experimental phase to isolate the effects of data balancing on model performance.

The balanced dataset provided equal representation across all classes, allowing us to better evaluate each model's true discriminative capabilities without the bias introduced by class imbalance. This setup also enabled more meaningful comparisons between macro and weighted metrics, as the balanced class distribution meant these metrics should theoretically converge.

The following table provides a comparative summary of the performance of all seven models evaluated in this study. The metrics include accuracy, and weighted averages of precision, recall, and F1-score.

| Model                                      | Accuracy | Precision  | Recall    | F1-Score  | Precision | Recall    | F1-Score  | 
|                                            |          | (weighted) | (weighted)| (weighted)| (macro)   | (macro)   | (macro)   |
|--------------------------------------------|----------|------------|-----------|-----------|-----------|-----------|-----------|
| $M_7$: Logistic Regression                 | 0.71     | 0.71       | 0.71      | 0.69      | 0.71      | 0.71      | 0.69      |
| $M_8$: Shallow Artificial Neural Network   | 0.76     | 0.76       | 0.76      | 0.76      | 0.76      | 0.76      | 0.76      |
| $M_9$: Convolutional Neural Network (CNN)  | 0.74     | 0.74       | 0.74      | 0.74      | 0.74      | 0.74      | 0.74      |

The experimental results reveal several key patterns and insights:

The second experimental run with balanced data revealed significant improvements across all models compared to the first run. Most notably, the Shallow Artificial Neural Network ($M_8$) demonstrated exceptional performance with 0.76 accuracy across all metrics (weighted and macro), showing a substantial improvement from its previous 0.70 accuracy. This consistent performance across both weighted and macro metrics indicates that the model performs equally well across all classes, validating the effectiveness of our data balancing approach.

The Convolutional Neural Network ($M_9$) also showed marked improvement, achieving 0.74 accuracy (up from 0.67), with consistent performance across all metrics. This improvement suggests that the CNN's ability to capture spatial patterns in the data was previously hindered by the class imbalance, and the balanced dataset allowed it to better learn discriminative features for all classes.

Logistic Regression ($M_7$) maintained its strong performance with 0.71 accuracy (slightly higher than its previous 0.69), demonstrating remarkable resilience and consistency across both imbalanced and balanced datasets. The model's weighted and macro metrics are nearly identical (around 0.71 for precision and recall, 0.69 for F1-score), indicating uniform performance across all classes. This suggests that the linear decision boundaries it creates are equally effective for all categories in the balanced scenario.

A particularly interesting observation is the convergence of weighted and macro metrics for all models in this balanced dataset scenario. This convergence confirms that our data balancing strategy successfully eliminated the bias towards majority classes present in the first run. The consistent performance across both metric types indicates that all models are now making equally reliable predictions across all classes, rather than achieving higher performance on majority classes at the expense of minority classes.


## Experimentation (3rd Run)
Building upon the insights gained from the second experimental run, we conducted a third phase focused on optimizing the neural network architecture through extensive hyperparameter tuning. This phase aimed to identify the optimal model configuration that could further improve upon the strong performance achieved with the balanced dataset.

The hyperparameter search focused on two critical aspects of neural network design: the hidden layer dimensions and learning rates. These parameters were chosen for optimization as they significantly impact both the model's capacity to learn complex patterns and its training dynamics. By conducting a comprehensive grid search across these parameters, we aimed to find the configuration that would maximize the model's discriminative power while maintaining stable training.

This experimental phase maintained the same balanced dataset and preprocessing pipeline from the second run to ensure valid comparisons. The key difference was the systematic exploration of the hyperparameter space to find the optimal model architecture.

For the third experimental run, we conducted an extensive hyperparameter search to optimize the neural network architecture. We explored a wide range of hidden layer dimensions, from very small (2 neurons) to very large (2,056 neurons), allowing us to understand how the model's capacity affects its performance. The hidden dimensions tested were: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1,024, and 2,056 neurons.

Additionally, we investigated the impact of different learning rates, spanning seven orders of magnitude from 1e-7 to 1e-1. This broad range enabled us to find the sweet spot between convergence speed and stability. The learning rates tested were: 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, and 0.0000001.

The hyperparameter search was performed using validation loss as the primary metric for model selection. Each combination of hidden dimension and learning rate was evaluated, resulting in 77 different model configurations. The models were trained using the same balanced dataset from the second experimental run to maintain consistency and comparability.

The hyperparameter optimization process revealed several key insights about the neural network architecture. The best performing model utilized a hidden dimension of 256 neurons with a learning rate of 0.001, achieving a validation loss of 1.5293. This configuration represents a balance between model complexity and training stability.

The results show that larger hidden dimensions (1024, 2056) generally performed worse, especially with higher learning rates (0.1, 0.01), often resulting in validation losses above 2.0. This suggests that overly complex architectures may be prone to overfitting on this particular dataset. Similarly, very small hidden dimensions (2, 4, 8) struggled to capture the underlying patterns in the data, particularly with lower learning rates.

The learning rate proved to be a crucial factor in model performance. Very high learning rates (0.1) consistently led to poor validation losses across different hidden dimensions, while very low learning rates (1e-07) resulted in slow convergence and suboptimal performance. The optimal learning rate of 0.001 provided the right balance for effective training.

Evaluation results
| Model                                      | Accuracy | Precision  | Recall    | F1-Score  | Precision | Recall    | F1-Score  | 
|                                            |          | (weighted) | (weighted)| (weighted)| (macro)   | (macro)   | (macro)   |
|--------------------------------------------|----------|------------|-----------|-----------|-----------|-----------|-----------|
| $M_8$: Shallow Artificial Neural Network   | 0.76     | 0.76       | 0.76      | 0.76      | 0.76      | 0.76      | 0.76      |
| $M_10$: Hyperparameter Optimised ANN       | 0.77     | 0.77       | 0.77      | 0.77      | 0.77      | 0.77      | 0.77      |

The evaluation results reveal several interesting patterns when comparing the baseline shallow neural network (M₈) with its hyperparameter-optimized counterpart (M₁₀). Most notably, the optimized model demonstrates consistent improvement across all evaluation metrics, albeit with modest gains.

The hyperparameter-optimized model achieves a 77% accuracy, representing a one percentage point improvement over the baseline model's 76%. This pattern of improvement is mirrored across all metrics, with both weighted and macro-averaged precision, recall, and F1-scores showing similar one percentage point gains. The consistency of this improvement across different metrics suggests that the optimization process led to genuine, albeit incremental, enhancement in model performance rather than just improvements in specific areas.

An interesting observation is the identical values between weighted and macro-averaged metrics for both models. This suggests a relatively balanced performance across different classes, as significant class imbalances would typically result in disparities between weighted and macro-averaged metrics. This balance is maintained even after hyperparameter optimization, indicating that the tuning process did not introduce bias toward any particular class.

While the improvements are modest, they demonstrate the value of hyperparameter optimization in fine-tuning model performance. The consistent nature of these improvements across all metrics suggests that the optimized model is more robust and reliable than its baseline counterpart, even if the gains are not dramatic.

## Key Findings

The analysis and experimentation conducted in this study revealed several key findings.

The optimization of the model architecture revealed that a hidden dimension of 256 neurons provided the best performance, effectively balancing model complexity and effectiveness. Larger architectures with 1024 or more neurons showed diminishing returns and potential overfitting issues, while very small architectures with fewer than 8 neurons lacked sufficient capacity to properly model the relationships in the data.

The learning rate proved to be a highly sensitive parameter in the training process. An optimal learning rate of 0.001 was crucial for model performance. Higher learning rates around 0.1 consistently led to unstable training across all tested architectures. Conversely, very low learning rates at 1e-07 resulted in slow convergence and suboptimal performance outcomes.

The hyperparameter-optimized model demonstrated consistent improvements across all evaluation metrics. The accuracy increased from 76% to 77% compared to the baseline model. Both weighted and macro-averaged metrics showed uniform improvements, indicating balanced performance across all classes in the dataset.

The class balance analysis revealed identical weighted and macro-averaged metrics, suggesting well-balanced performance across different classes. This balance was successfully maintained throughout the optimization process, demonstrating robust and unbiased model behavior.

While the improvements from optimization were modest in magnitude, they were notably consistent across all evaluation metrics. The optimization process resulted in a more reliable and robust model without introducing any class-specific biases. These results clearly demonstrate the value of systematic hyperparameter tuning, even when the resulting gains are incremental in nature.

These findings highlight the importance of careful model architecture design and hyperparameter selection in neural network development, while also demonstrating that even modest improvements through optimization can lead to more robust and reliable models.

## Future Work

Several promising directions for future work emerge from this study. First, exploring more sophisticated neural network architectures, such as deep neural networks with multiple hidden layers or architectures incorporating residual connections, could potentially capture more complex patterns in the data. This could help overcome the current model's performance ceiling and achieve more substantial improvements over the baseline.

The investigation of alternative optimization techniques, such as adaptive learning rate methods like Adam or RMSprop, could provide better training dynamics compared to the current approach. Additionally, implementing techniques like batch normalization or dropout could enhance model regularization and potentially improve generalization performance.

Another valuable direction would be to conduct a more extensive feature engineering process. While the current model works with the existing feature set, developing domain-specific features or applying advanced feature selection methods might uncover more informative patterns in the data. This could include exploring feature interactions or incorporating domain knowledge to create more meaningful representations.

Expanding the hyperparameter search space could also yield valuable insights. While this study focused on hidden dimensions and learning rates, other parameters such as batch size, activation functions, and optimization algorithms could be included in the optimization process. A more comprehensive grid search or the implementation of advanced hyperparameter optimization techniques like Bayesian optimization could potentially discover better model configurations.

Finally, investigating the model's behavior on different subsets of the data or specific edge cases could provide insights into its limitations and guide future improvements. This could include analyzing misclassified examples in detail or evaluating the model's performance on particularly challenging instances. Such analysis could inform targeted improvements to the model architecture or training process.

# Acknowledgement

I would like to express my sincere gratitude to Professor Jin Cheon Na for his invaluable guidance and support throughout this research project. His expertise and insightful feedback have significantly contributed to shaping and improving this work.
