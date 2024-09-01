import pandas as pd

# Preprocess the first dataset (twitter_training.csv)
try:
    twitter_df = pd.read_csv('/Users/nikhilmalkari/Documents/PhD/Main Papers/Hierarchical Transfer Learning for Sentiment Analysis and Text Classification in Low-Resource Domains/Hierarchical-Transfer-Learning-Code/Data/twitter_training.csv', header=None, encoding='ISO-8859-1')
    # Assign column names
    twitter_df.columns = ['es', 'ex', 'sentiment', 'review']  # Adjust based on actual columns
    # Keep only the necessary columns: review and sentiment
    twitter_df = twitter_df[['review', 'sentiment']]
except Exception as e:
    print(f"Error processing twitter_training.csv: {e}")

# Preprocess the second dataset (IMDB Dataset.csv)
try:
    imdb_df = pd.read_csv('/Users/nikhilmalkari/Documents/PhD/Main Papers/Hierarchical Transfer Learning for Sentiment Analysis and Text Classification in Low-Resource Domains/Hierarchical-Transfer-Learning-Code/Data/IMDB Dataset.csv', encoding='ISO-8859-1')
    # Reorder columns: first column as review and second as sentiment
    imdb_df = imdb_df[['review', 'sentiment']]
except Exception as e:
    print(f"Error processing IMDB Dataset.csv: {e}")

# Preprocess the third dataset (all-data.csv)
try:
    all_data_df = pd.read_csv('/Users/nikhilmalkari/Documents/PhD/Main Papers/Hierarchical Transfer Learning for Sentiment Analysis and Text Classification in Low-Resource Domains/Hierarchical-Transfer-Learning-Code/Data/all-data.csv', header=None, encoding='ISO-8859-1')
    # Assign column names
    all_data_df.columns = ['sentiment','review']  # Adjust based on actual columns
    # Keep only the necessary columns: review and sentiment
    all_data_df = all_data_df[['review', 'sentiment']]
except Exception as e:
    print(f"Error processing all-data.csv: {e}")

# Normalize the sentiment labels (e.g., convert to lowercase)
twitter_df['sentiment'] = twitter_df['sentiment'].str.lower()
imdb_df['sentiment'] = imdb_df['sentiment'].str.lower()
all_data_df['sentiment'] = all_data_df['sentiment'].str.lower()

# Save the preprocessed datasets
twitter_df.to_csv('twitter_preprocessed.csv', index=False)
imdb_df.to_csv('imdb_preprocessed.csv', index=False)
all_data_df.to_csv('all_data_preprocessed.csv', index=False)
