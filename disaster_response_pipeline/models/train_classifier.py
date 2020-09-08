import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
import nltk 
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pickle
from sklearn.metrics import classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disasterResponse", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, y, list(y.columns.values)

def tokenize(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # normalized
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    # split on space    
    words = word_tokenize(text)
    # remove stop words    
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatize 
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    return lemmed

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__min_df': [1, 2],
        'clf__estimator__max_features': [0.25]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)
    return cv

# Get results and add them to a dataframe.
def get_classification_report(y_test, y_pred):
    """
    To get F1 score,precision for each category
    Param:
    y_test - actual y values 
    y_pred - predicted y values
    """
    for ind,cat in enumerate(y_test.keys()): 
        print("Classification report for {}".format(cat))
        print(classification_report(y_test.iloc[:,ind], y_pred[:,ind]))
        
def evaluate_model(model, X_test, Y_test, category_names):
    y_test_pred = model.predict(X_test)
    print(get_classification_report(Y_test, y_test_pred))    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
