import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, num_samples=10000):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess_data(df):
    # Select relevant columns
    df = df[['id', 'title', 'abstract', 'categories']]
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Combine title and abstract for full-text search
    df['full_text'] = df['title'] + ' ' + df['abstract']
    
    return df

def split_data(df):
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

def main():
    # Load and preprocess data
    df = load_data('data/arxiv-metadata-oai-snapshot.json')
    df = preprocess_data(df)
    
    # Split data
    train_df, test_df = split_data(df)
    
    # Save processed data
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)

if __name__ == "__main__":
    main()