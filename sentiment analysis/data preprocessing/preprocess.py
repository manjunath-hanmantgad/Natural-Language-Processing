# import libraries

import pandas as pd
import numpy as np
import nltk
import string
import fasttext
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizerplt.xticks(rotation=70)
pd.options.mode.chained_assignment = None
pd.set_option('display.max_colwidth', 100)
%matplotlib inline


# import data

with open('indeed_scrape.csv') as f:
    df = pd.read_csv(f)
f.close()

# drop null values 

df.drop('Unnamed: 0', axis=1, inplace=True)

# check for missing values 

for col in df.columns:
    print(col, df[col].isnull().sum())

# display the data 

rws = df.loc[:, ['rating', 'rating_description']]

# apply contractions using lambda functions

rws['no_contract'] = rws['rating_description'].apply(lambda x: [contractions.fix(word) for word in x.split()])
rws.head()

rws['rating_description_str'] = [' '.join(map(str, l)) for l in rws['no_contract']]
rws.head()

# using pre trained model 

pretrained_model = "lid.176.bin" 
model = fasttext.load_model(pretrained_model)langs = []
for sent in rws['rating_description_str']:
    lang = model.predict(sent)[0]
    langs.append(str(lang)[11:13])rws['langs'] = langs

# tokenization 

rws['tokenized'] = rws['rating_description_str'].apply(word_tokenize)
rws.head()

# converting to lowercase 

rws['lower'] = rws['tokenized'].apply(lambda x: [word.lower() for word in x])
rws.head()

# remove pucntuations 

punc = string.punctuation
rws['no_punc'] = rws['lower'].apply(lambda x: [word for word in x if word not in punc])
rws.head()


# remove stopwords 

stop_words = set(stopwords.words('english'))
rws['stopwords_removed'] = rws['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
rws.head()

# applying lemmatization 

rws['pos_tags'] = rws['stopwords_removed'].apply(nltk.tag.pos_tag)
rws.head()

# apply word lemmatizer 

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUNrws['wordnet_pos'] = rws['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
rws.head()

wnl = WordNetLemmatizer()
rws['lemmatized'] = rws['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
rws.head()

# save this preprocessed file to csv 

rws.to_csv('indeed_scrape_clean.csv')