import numpy as np
import pandas as pd
import re
import spacy
from spellchecker import SpellChecker
# class TextPreprocessor(dataset):

data=pd.read_csv("training.csv")
text=data['text']
label=data['label']

# def remove_special_char(text):

#     return text.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', ' ', str(x)).strip() and str(x).lower())


# data["clean_text"] = remove_special_char(text)

nlp = spacy.load("en_core_web_sm")
spell=SpellChecker()
def expand_contractions(text):
    contractions = {
        "I'm": "I am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "can't": "cannot",
        "shouldn't": "should not",
        "couldn't": "could not",
        "wouldn't": "would not"
    }
    
    for contraction, replacement in contractions.items():
        text = re.sub(r"\b" + contraction + r"\b", replacement, text, flags=re.IGNORECASE)
    
    return text

def spell_correction(text):
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word is None:  # Handle None values
            corrected_word = word  # Use the original word if correction fails
        corrected_words.append(corrected_word)
    return " ".join(corrected_words)




def preprocess_text(text):


    text = expand_contractions(text)  
    text = spell_correction(text)

    doc = nlp(text.lower())  
    
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return " ".join(clean_tokens)  


data['clean_text'] = data['text'].apply(preprocess_text)
print(data[['text', 'clean_text']])
