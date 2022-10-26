"""
Python file to perform ETL operations (ETL Pipeline)
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load data from files to pandas dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    raw_df = messages.merge(categories, how='inner', on=['id'])
    return raw_df


def clean_data(raw_df):
    categories = raw_df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(
            lambda x: x.split("-")[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    categories.related[categories.related == 2] = 0
    raw_df.drop(['categories'], axis=1, inplace=True)
    cleaned_df = pd.concat([raw_df, categories], axis=1)
    # drop duplicates
    cleaned_df.drop_duplicates(keep=False, inplace=True)
    return cleaned_df


def save_data(cleaned_df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    cleaned_df.to_sql('etlpipeline', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        raw_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        cleaned_df = clean_data(raw_df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(cleaned_df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
