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
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('etlpipeline', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns.values


def tokenize(text):
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


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_df_improved = pd.DataFrame(y_pred, columns=category_names)
    for each in y_pred_df_improved.columns:
        print(f"Classification Report for the columns: {each}")
        print(classification_report(y_pred_df_improved[each], Y_test[each]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
