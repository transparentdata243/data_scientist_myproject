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
    '''
    INPUT:
    database_filepath: the database file to be read 

    OUTPUT:
    X - panda series with messages to be trained
    y - dataframe with category columns and corresponding values
    list(y.columns.values) - a list contains the column names

    Reads the database and loads X, y, list(y.columns.values) accordingly
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disasterResponse", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, y, list(y.columns.values)

def tokenize(text):
    '''
    INPUT:
    text - a string that contains message

    OUTPUT:
    lemmed - tokenized string

    This API does the following to the input string and return the processed
    result
    - normalization
    - puncutation removal
    - split on space
    - remove stop words based on nltk dict
    - lemmatization
    '''
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
    '''
    INPUT:
    None

    OUTPUT:
    cv - a machine learning model

    Creates a pipeline with CountVectorizer, TfidfTranformer, and classisifier
    Use GridSearchCV to tune pipeline parameters
    '''
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

def get_classification_report(y_test, y_pred):
    """
    INPUT:
    y_test - actual y values 
    y_pred - predicted y values

    OUTPUT:
    None

    Prints F1 score, precision for each category
    """
    for ind,cat in enumerate(y_test.keys()): 
        print("Classification report for {}".format(cat))
        print(classification_report(y_test.iloc[:,ind], y_pred[:,ind]))
        
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - machine learning model to be evaluated 
    X_test - Test input data
    Y_test - Test output data
    category_names - classification categories

    OUTPUT:
    NONE

    Prints classification results as a evaluation for the training model using
    test data
    '''
    y_test_pred = model.predict(X_test)
    print(get_classification_report(Y_test, y_test_pred))    


def save_model(model, model_filepath):
    '''
    INPUT:
    model - ML model to be saved
    model_filepath - ML model file path

    OUTPUT:
    None

    Saves the ML model to a pickle file
    '''
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
