import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # 3. Split `categories` into separate category columns.
    #- Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
    # - Use the first row of categories dataframe to create column names for the categories data.
    # - Rename columns of `categories` with new column names. 
    # 
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

    # 4. Convert category values to just numbers 0 or 1.
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    # You can perform normal string actions on Pandas Series, like indexing, by including .str after the Series. You may need to first convert the Series to be of type string, which you can do with astype(str)    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Confirm values in each column of the categories dataframe are, in fact, only 0's and 1's.
    #for column in categories:
    #    if len(np.unique(categories[column])) != 2:
    #        print(column, np.unique(categories[column]))
    
    # drop child_alone since all values are the same
    categories.drop('child_alone', axis=1)

    # 5. Replace categories column in df with new category columns.
    # Drop the categories column from the df dataframe since it is no longer needed.
    # Concatenate df and categories data frames. 
    
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)    
    # drop duplicates
    df.drop_duplicates(inplace=True)    
    return df
    
    
def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()