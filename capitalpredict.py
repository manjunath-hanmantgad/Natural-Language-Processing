import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_vectors

data = pd.read_csv('capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
data.head(5)

import nltk
from gensim.models import KeyedVectors


embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
f = open('capitals.txt', 'r').read()
set_words = set(nltk.word_tokenize(f))
select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
for w in select_words:
    set_words.add(w)

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


# Testing your function
word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )

# load the words as python dictionary 

word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
len(word_embeddings)

# dimension of the word ?

print("dimension: {}".format(word_embeddings['Spain'].shape[0])

# predicting the realtionship among words.

# using cosine similarity 

""" A and B represent the word vectors and A_i or B_i represent index i of that vector. 
Note that if A and B are identical, you will get cos(\theta) = 1.

Otherwise, if they are the total opposite, meaning, A= -B, then you would get $cos(\theta) = -1$.
If you get cos(\theta) =0, that means that they are orthogonal (or perpendicular).
Numbers between 0 and 1 indicate a similarity score.
Numbers between -1-0 indicate a dissimilarity score.  """

def cosine_similarity(A,B):

    dot = np.dot(A,B)
    norm_A = np.sqrt(np.dot(A,A))
    norm_B = np.sqrt(np.dot(B,B))
    cos = dot / (norm_A * norm_B) # this is formula for cosine similarity.

    return cos 

# now try out words 

man = word_embeddings['man']
woman = word_embeddings['woman']

cosine_similarity(man,woman)

# calculate the Eucliedean distance 

"""  the more the words are similar the less will be Euclidean distance i.e close to 0"""

# implement and check euclidean distance 

def euclidean(A,B):
    d = np.linalg.norm(A-B)
    return d

# test function 

euclidean(man,woman) 


# Finding capital of the country.

# same as : King - Man + Woman = Queen

def get_capital(city1, country1, city2, embeddings):

    # store the city1, country 1, and city 2 in a set called group
    group = set((city1, country1 , city2))

    # get embeddings of city 1
    city1_emb = word_embeddings[city1]

    # get embedding of country 1
    country1_emb = word_embeddings[country1]

    # get embedding of city 2
    city2_emb = word_embeddings[city2]

    # get embedding of country 2
    """ this will be combination of embeddings of city1 , country 1 , city 2"""
    vec = country1_emb - city1_emb + city2_emb # same as king - man + woman = queen 

    # initialize similairty

    similarity = -1

    # initialize country to empty string 
    country = ''

    # loop through all words in the embeddings dict.

    for word in embeddings.key():

        # check if word is not already in the group ( check for test data and training data separation)

        if word not in group:
            # then get word_embedding
            word_emb = word_embeddings[word]

            # calculate cosine similarity as previously.
            # between embedding of country 2 and word in embedding dict.
            cur_similarity = cosine_similarity(vec,word_emb)

            # if cosine similarity is more similar than priveous best similar value then
            if cur_similarity > similarity:
                
                # update similarity to new which is better than old similairty
                similairty = cur_similarity

                # store the country as tuple which will have country and its similarity 

                country = (word,similairty)

    return country

# now test the function 

get_capital('Athens', 'Greece', 'Cairo', word_embeddings)

# check the output here 

# now check the model accuracy 

""" see that if predicted country is same as country in data set then go on to next country and loop this """

def get_accuracy(word_embeddings, data):
    num_correct = 0 

    # loop through the dataframe

    for i, row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']

        # now predict country 

        predicted_country2, _ = get_capital(city1,country1,city2,word_embeddings)

        # if the predicted country2 is the same as the actual country2

        if predicted_country2 == country2:
            # increment the counter and go onto next 
            num_correct += 1

    # get number of rows in dataframe i.e length of dataframe 

    m = len(data)        

    # calculate teh accuracy 

    accuracy = num_correct/m 

    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuraccy is {accuracyL.2f}")


# Using PCA 

""" 
we are working in a 300-dimensional space in this case. Although from a computational perspective we were able to perform a good job, it is impossible to visualize results in such high dimensional spaces.
You can think of PCA as a method that projects our vectors in a space of reduced dimension,
while keeping the maximum information about the original vectors in their reduced counterparts. 
In this case, by maximum infomation we mean that the Euclidean distance between the original vectors and their projected siblings is minimal.
Hence vectors that were originally close in the embeddings dictionary, will produce lower dimensional vectors that are still close to each other.

"""

""" 
 The steps to compute PCA are as follows:

- Mean normalize the data
- Compute the covariance matrix of your data ($\Sigma$).
- Compute the eigenvectors and the eigenvalues of your covariance matrix
- Multiply the first K eigenvectors by your normalized data

"""

def compute_pca(X, n_components=2):
    X_demeaned = X - np.mean(X, axis=0)
    print('X_demeaned.shape: ',X_demeaned.shape)

#calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False

# calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')

# sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)
# reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

# sort the eigen values by idx_sorted_decreasing

    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
 # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]

# select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:,0:n_components]
  # transform the data by multiplying the transpose of the eigenvectors 
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(eigen_vecs_subset.transpose(),X_demeaned.transpose()).transpose()

    return X_reduced

# testing the function 

np.random.seed(1)
X = np.random.rand(3, 10)
#print(X)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)



words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)


# PLOT it !!

result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()