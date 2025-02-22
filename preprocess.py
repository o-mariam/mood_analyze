import numpy as np
import pandas as pd
import re


# class TextPreprocessor(dataset):

data=pd.read_csv("training.csv")
text=data['text']
label=data['label']

def remove_special_char(text):

    return text.apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', ' ', str(x)).strip() and str(x).lower())


data["clean_text"] = remove_special_char(text)

