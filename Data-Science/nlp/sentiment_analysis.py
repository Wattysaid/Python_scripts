"""
Sentiment Analysis
-----------------
Functions for sentiment analysis and text classification.
"""

import pandas as pd
import numpy as np
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def analyze_sentiment_textblob(texts):
    """Analyze sentiment using TextBlob."""
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    for text in texts:
        blob = TextBlob(text)
        
        # Polarity: -1 (negative) to 1 (positive)
        # Subjectivity: 0 (objective) to 1 (subjective)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        results.append({
            'text': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment,
            'confidence': abs(polarity)
        })
    
    # Summary statistics
    polarities = [r['polarity'] for r in results]
    subjectivities = [r['subjectivity'] for r in results]
    sentiments = [r['sentiment'] for r in results]
    
    summary = {
        'total_texts': len(results),
        'sentiment_distribution': Counter(sentiments),
        'avg_polarity': np.mean(polarities),
        'avg_subjectivity': np.mean(subjectivities),
        'polarity_std': np.std(polarities),
        'subjectivity_std': np.std(subjectivities)
    }
    
    return {
        'individual_results': results,
        'summary_statistics': summary
    }

def train_sentiment_classifier(texts, labels, test_size=0.2, random_state=42, method='logistic'):
    """Train a custom sentiment classifier."""
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state)
    
    # Train model
    if method == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif method == 'naive_bayes':
        model = MultinomialNB()
    else:
        raise ValueError("Method must be 'logistic' or 'naive_bayes'")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance (for logistic regression)
    feature_importance = None
    if method == 'logistic' and hasattr(model, 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        if len(model.classes_) == 2:  # Binary classification
            importance_scores = model.coef_[0]
        else:  # Multi-class
            importance_scores = np.mean(np.abs(model.coef_), axis=0)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', key=abs, ascending=False).head(20)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': feature_importance,
        'test_predictions': y_pred,
        'test_actual': y_test
    }

def extract_sentiment_keywords(texts, sentiments, top_n=20):
    """Extract keywords associated with different sentiments."""
    # Combine texts by sentiment
    sentiment_texts = {}
    for text, sentiment in zip(texts, sentiments):
        if sentiment not in sentiment_texts:
            sentiment_texts[sentiment] = []
        sentiment_texts[sentiment].append(text)
    
    # Extract keywords for each sentiment
    sentiment_keywords = {}
    
    for sentiment, text_list in sentiment_texts.items():
        # Combine all texts for this sentiment
        combined_text = ' '.join(text_list)
        
        # Use TF-IDF to find important words
        vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top keywords
            top_indices = np.argsort(tfidf_scores)[-top_n:]
            keywords = [
                {
                    'keyword': feature_names[i],
                    'tfidf_score': tfidf_scores[i]
                }
                for i in reversed(top_indices)
            ]
            
            sentiment_keywords[sentiment] = keywords
        except:
            sentiment_keywords[sentiment] = []
    
    return sentiment_keywords

def analyze_emotion_intensity(texts, emotion_words=None):
    """Analyze emotional intensity using predefined emotion words."""
    if emotion_words is None:
        emotion_words = {
            'joy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'glad', 'elated'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'outraged'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous', 'frightened'],
            'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'grief', 'sorrow', 'melancholy'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'bewildered'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'nauseated', 'sickened']
        }
    
    results = []
    
    for text in texts:
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, words in emotion_words.items():
            score = sum(text_lower.count(word) for word in words)
            emotion_scores[emotion] = score
        
        # Normalize by text length
        text_length = len(text.split())
        normalized_scores = {emotion: score/text_length if text_length > 0 else 0 
                           for emotion, score in emotion_scores.items()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get) if any(emotion_scores.values()) else 'neutral'
        
        results.append({
            'text': text,
            'emotion_scores': emotion_scores,
            'normalized_scores': normalized_scores,
            'dominant_emotion': dominant_emotion,
            'total_emotional_words': sum(emotion_scores.values())
        })
    
    # Summary statistics
    all_emotions = list(emotion_words.keys())
    emotion_totals = {emotion: sum(r['emotion_scores'][emotion] for r in results) for emotion in all_emotions}
    dominant_emotions = [r['dominant_emotion'] for r in results]
    
    summary = {
        'total_texts': len(results),
        'emotion_distribution': Counter(dominant_emotions),
        'emotion_totals': emotion_totals,
        'avg_emotional_intensity': np.mean([r['total_emotional_words'] for r in results])
    }
    
    return {
        'individual_results': results,
        'summary_statistics': summary,
        'emotion_categories': list(emotion_words.keys())
    }