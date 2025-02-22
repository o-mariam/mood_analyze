import numpy as np
import pandas as pd
import re
import spacy

# class TextPreprocessor(dataset):

data=pd.read_csv("training.csv")
text=data['text']
label=data['label']

# def remove_special_char(text):

#     return text.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', ' ', str(x)).strip() and str(x).lower())


# data["clean_text"] = remove_special_char(text)

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase and process text
    
    # Remove stopwords and punctuation, keep only lemmatized words
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(clean_tokens)  # Reconstruct cleaned sentence

# Apply the preprocessing function to the dataset
data['clean_text'] = data['text'].apply(preprocess_text)

# Display results
print(data[['text', 'clean_text']].head)