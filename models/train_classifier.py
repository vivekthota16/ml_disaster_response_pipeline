"""
Python file to train classifier (ML pipeline)
"""
import sys
import dill as pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, f1_score, classification_report
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """
    function to load data from sqlite database
    :param str database_filepath: location of db file
    :return X(features), Y(labels), list of column names of Y
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('etlpipeline', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns.values


def tokenize(text):
    """
    function to tokenize text (perform word tokenization, stemming and lemmatization using nltk)
    :parm str text
    :return words_lemmed: lemmatized words
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    stemmed = [PorterStemmer().stem(tok) for tok in tokens]
    lemmatizer = WordNetLemmatizer()
    words_lemmed = [lemmatizer.lemmatize(tok)
                    for tok in stemmed if tok not in stop_words]
    return words_lemmed

# pylint: disable='line-too-long'


def build_model():
    """
    function to initialize pipeline for training model
    :return GridSearchCV object gcv: object with pipeline and parameters
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(n_jobs=-1)))])
    # Taking too long
    parameters = {'vect__max_df': (0.75, 1.0),
                  #   'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [50, 100, 200]}

    #parameters = {'clf__estimator__n_estimators':[50,100,200]}
    gcv = GridSearchCV(pipeline, param_grid=parameters, verbose=7)
    return gcv


def evaluate_model(model, x_test, y_test, category_names):
    """
    function to evaluate model performance
    :param object model: model to perform prediction
    :param dataframe x_test: test data
    :param dataframe y_test: test labels data
    :param list category_names: list of category_names
    """
    y_pred = model.predict(x_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    print(classification_report(y_test, y_pred_df, category_names))


def save_model(model, model_filepath):
    """
    function to save model in provided file path as .pkl file
    :param object model: model to save 
    :param str model_filepath: location to save the model 
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    main function
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
