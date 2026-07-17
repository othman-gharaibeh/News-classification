# News-classification
# AG News Classification using Word2Vec and LSTM

This repository contains a complete NLP pipeline for classifying news articles into four categories using a Deep Learning approach. The project leverages custom Word2Vec embeddings and a Long Short-Term Memory (LSTM) network to achieve high accuracy.

##  Project Overview
The goal of this project is to categorize news headlines and descriptions from the **AG News Dataset** into one of four topics:
1.  **World**
2.  **Sports**
3.  **Business**
4.  **Sci/Tech**

##  Methodology

### 1. Data Preprocessing
- **Text Cleaning:** Lowercasing, removal of URLs, numbers, and special characters.
- **Tokenization & Lemmatization:** Using NLTK for part-of-speech tagging and lemmatization to reduce words to their base forms.
- **Stopwords:** Removal of common English stopwords to focus on meaningful terms.

### 2. Feature Engineering (Word2Vec)
Instead of using generic pre-trained embeddings, we trained a custom **Word2Vec** model specifically on the AG News corpus. This allows the model to capture the specific context and semantic relationships found in news reporting.
- **Vector Size:** 100 dimensions
- **Window:** 5

### 3. Model Architecture (LSTM)
An LSTM network was built using TensorFlow/Keras to handle the sequential nature of the text:
- **Embedding Layer:** Initialized with the custom Word2Vec weights.
- **Spatial Dropout:** To prevent overfitting.
- **LSTM Layer:** 64 units with dropout and recurrent dropout.
- **Output Layer:** Softmax activation for 4-class classification.

##  Results
- **Final Test Accuracy:** ~90.82%
- **Key Observation:** The model performs exceptionally well on **Sports** news. The most common confusion occurs between **Business** and **Sci/Tech** due to the overlapping nature of corporate technology news.

##  Files in this Repo
- `word2vec_ag_news.model`: The trained Gensim Word2Vec model.
- `lstm_classifier.keras`: The final trained LSTM model file.
- `notebook.ipynb`: The complete code for cleaning, training, and evaluation.

##  Requirements
- Python 3.x
- TensorFlow / Keras
- Gensim
- NLTK
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
