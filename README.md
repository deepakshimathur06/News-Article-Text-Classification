# News Article Text Classification

This project focuses on classifying news articles into categories like Politics, Sports, Business, Entertainment, and Tech using machine learning and deep learning techniques.

## Problem Statement
Given a corpus of raw text articles, predict their categories accurately based on their textual content.

## Approach
- Preprocessing: Stemming, Tokenization, Stopword removal, Bigrams, Keyword extraction (YAKE)
- Feature Engineering: 
  - CountVectorizer (Bag of Words)
  - TF-IDF
  - GloVe Embeddings
  - BERT Embeddings
- Modeling:
  - Neural Networks (MLPClassifier, TensorFlow/Keras model with 2 hidden layers, 128 neurons each)
- Evaluation:
  - 5-Fold Cross-Validation
  - Learning Rate Tuning
  - Optimizer Comparison (SGD, RMSprop, Adam)

## Results
- Achieved average accuracy of 97.5% on validation.
- Best performance using BERT Embeddings + Adam Optimizer.

## Tools & Libraries
- Python, TensorFlow, Keras, Scikit-Learn, NLTK, Gensim, Transformers

## Project Structure
- `notebooks/`: Code for preprocessing, feature extraction, modeling.
- `data/`: Raw and processed datasets.
- `requirements.txt`: Required Python packages.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook in `notebooks/`
3. Run all cells to preprocess, train, and generate predictions.

## Author
Deepakshi Mathur
