# Libraries
import re
import string
import spacy
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import Counter
import warnings
import numpy as np
nltk.download("vader_lexicon")
nltk.download('stopwords')
warnings.filterwarnings('ignore')

# Model
class PredictReview:
    
    def vectorize(self,train_data,test_data):
        tfidf = TfidfVectorizer()
        train = tfidf.fit_transform(train_data.values.astype('U'))
        test = tfidf.transform(test_data.values.astype('U'))
        return train,test,tfidf
    
    def split(self,data,train_size=0.1,shuffle=101):
        input_data = data['review']
        output_data = data['label_num']
        train_data, test_data, train_output, test_output = train_test_split(input_data, output_data, test_size=train_size, random_state=shuffle)
        return train_data, test_data, train_output, test_output
    
    def evaluate_model(self, y_true, y_pred, model_name="Logistic Regression"):
        """
        Comprehensive model evaluation with multiple metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Print evaluation results
        print(f"\n{model_name} Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def base(self,data):
        log_reg = LogisticRegression()
        data = self.prepare_data_for_train(data)
        train_data, test_data, train_output, test_output = self.split(data)
        train,test,tfidf = self.vectorize(train_data,test_data)
        log_reg.fit(train,train_output)
        pred = log_reg.predict(test)
        
        # Evaluate model performance
        metrics = self.evaluate_model(test_output, pred)
        
        return log_reg, tfidf, metrics
   
    def prepare_data_for_train(self,input_data):
        nlp = spacy.load('en_core_web_sm')
        stopword = nltk.corpus.stopwords.words('english')
        empty_list  = []
        for text in input_data.review:
            text = text.lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = re.sub(r'[^\w\s]', '',str(text))            
            text=re.split("\W+",text)                          
            text=[word for word in text if word not in stopword]
            text = ' '.join(text)       
            empty_list.append(text)
        input_data['review'] = empty_list
        return input_data
        
    def test_sample(self,text,tfidf,base_model):
        text = self.clean_df(text)
        text_sample = tfidf.transform([text])
        pred = base_model.predict(text_sample)
        prob = base_model.predict_proba(text_sample)
        confidence = max(prob[0])
        
        if pred[0] == 1:
            return f'Positive 🙂 (Confidence: {confidence:.3f})'
        else:
            return f'Negative 🙁 (Confidence: {confidence:.3f})'
        
    def clean_df(self,text):
        text = text.lower()
        nlp = spacy.load('en_core_web_sm')
        stopword = nltk.corpus.stopwords.words('english')
        text = re.sub(r'[^\w\s]', '',str(text))            
        text=re.split("\W+",text)                          
        text=[word for word in text if word not in stopword]
        text = ' '.join(text)                              
        return text
