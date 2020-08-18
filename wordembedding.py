import pandas as pd # for dataframe 
import numpy as np 
import pickle # for object serialization 

""" load the model """

word_embeddings = pickle.load( open("word_embeddings_subset.p" , "rb"))
# here word_embeddings is a dictionary 
# so key is word itself and value is its corresponding vector represntaion

len(word_embeddings) # number of words.

# getthing word embedding for the words. 

nationVector = word_embeddings['nation'] # this will give vector represnetaion 

print(type(nationVector))

print(nationVector) # this will be a ndarray 

""" each word is stored as numpy array """

# get the vector for each word.

def vec(w):
    return word_embeddings[w]

# word embeddings are mutlidimensional arrays.

""" make plot of word embeddings , the arrow represntaions gives us 
to vizualize the vector alignment """

import matplotlib.pyplot as plt 

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']

# convert each word to its vector represnation 

word_vec_rep = np.array([vec(word) for word in words])

# create custom size image 
fig , ax = plt.subplots(figsize = (10,10))

# selct the column for x axis and Y axis.

col1 = 3 # X axis
col2 = 2  # y axis

# print arrow for each word.

for word in word_vec_rep:
    ax.arrow(0,0, word[col1], word[col2], head_width=0.005 , head_length=0.005,
    fc='r', ec='r', width = 1e-5)

# plot dot for each word 

ax.scatter(word_vec_rep[:,col1],word_vec_rep[:,col2]);

# add the label for each word over its dot in scatter plot

for i in range(0, len(words)):
    ax.annotate(words[i], (word_vec_rep[i,col1], word_vec_rep[i, col2]))

plt.show()



# WORD distance 

""" plot the relative meaning words and check the distance  """

words = ['sad', 'happy', 'town', 'village']

# convert each word in its vector represnation 

word_vec_rep2 = np.array([vec(word) for word in words]) 

fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image


col1 = 3 # Select the column for the x axis
col2 = 2 # Select the column for the y axis

# print arrow for each word.

for word in word_vec_rep2:
    ax.arrow(0,0, word[col1], word[col2], head_width=0.005 , head_length=0.005,
    fc='r', ec='r', width = 1e-5)

# print the vector difference among the listed words.

village = vec('village')
town = vec('town')
diff = town - village
ax.arrow(village[col1], village[col2], diff[col1], diff[col2], fc='b', ec='b', width = 1e-5)

# print the vector difference between village and town 

sad = vec('sad')
happy = vec('happy')
diff = happy - sad
ax.arrow(sad[col1], sad[col2], diff[col1], diff[col2], fc='b', ec='b', width = 1e-5)

# plot a dot for each word.

ax.scatter(word_vec_rep2[:, col1], word_vec_rep2[:, col2]);

# now add the word label over each dot in scatter plot 

for i in range(0, len(words)):
    ax.annotate(words[i], word_vec_rep2[i, col1], word_vec_rep2[i, col2])

plt.show()


# predicting capitals of countries using word embeddings.

capital = vec('India') - vec('Delhi')
country = vec('Minsk') + capital 

print(country[0:10]) # print 10 values of vector 

diff = country - vec('Belarus')
print(diff[0:10])


""" If the word embedding works as expected, the most similar word must be 'Belarus'. 
Let us define a function that helps us to do it. 
We will store our word embedding as a DataFrame, which facilitate the lookup operations based on the numerical vectors. """ 


# create a dataframe of dictionary embedding for algebraic operations 

keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])

embedding = pd.DataFrame(data=data , index=keys)

# now define a function to find closest word to vector 

def find_closest_word(v, k = 1):
    diff = embedding.values - v # this calculates vector differences from each word to the input word vector.
    """ get the norm of each difference vector i.e the sqaured euclidean distance from each word to the input word vector """
    delta = np.sum(diff * diff , axis=1)

    # find the index of minimum distance in the array 
    i = np.argmin(delta)
    # now return row name for this item 
    return embedding.iloc[i].name

# now print the rows of embedding as dataframe 

embedding.head(50)


# now find the name that corresponds to the numerical country i.e Belarus 

find_closest_word(country)


# now try to predict other countries 

find_closest_word(vec('Spain') - vec('Madrid') + vec('Minsk'))

# this will give the correct output as ' Belarus '

print(find_closest_word(vec('Berlin') + capital))
print(find_closest_word(vec('Cairo') + capital))

# output should be Germany and Egypt.


# represent a sentence as vector.

""" A whole sentence can be represented as a vector by summing all the word vectors that conform to the sentence """ 

doc = "Spain petroleum city king"

vdoc = [vec(x) for x in doc.split(" ")]
doc2vec = np.sum(vdoc , axis=0)

print(doc2vec)

print(find_closest_word(doc2vec))