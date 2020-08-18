import numpy as np 

# make a list and arrays in numpy 

mylist = [1,2,3,4]
myarray = np.array([1,2,3,4]) # notice the brackets here for array.

# to see and check the data type of the list and array.

print(mylist)
print(type(mylist))

print(myarray)
print(type(myarray))


# '+' operator:

""" in numpy + operator means element wise addition 
but + operator in python means list concateneation  """

# usage of + operator 

print(myarray + myarray)
print(mylist + mylist)

# '*' operator 

""" in numpy * operator will give scale our vector i.e if its 1,2,3 then *3 
will give us 3,6,9 i.e it is scaling the vector 

in python * operator will concatenate the list with number of times equal to the number"""

# usage of * operator 

print(myarray * 3)
print(mylist * 3)


# matrix creation in numpy 

""" use the np.array(). Matrix is array of an array. 
the resulting matrix will always be an array and never a list """

mymatrix1 = np.array([myarray,myarray,myarray]) # initialize matrix with array 

mymatrix2 = np.array([mylist,mylist,mylist]) # initialize matrix with list

mymatrix3 = np.array([myarray,[1,2,3,4] , myarray]) # initialize matrix using array and list.

# now print them all

print(mymatrix1)
print(mymatrix2)
print(mymatrix3)


# scaling and translating matrices.

# making addition and multiplication 

""" multiply the matrix by 2 and add 1 to it """

mymatrix = np.array([[1,2],[2,3],[3,4]]) # defined 2x2 matrix.

result = mymatrix * 2 + 1 # this will do what I want.

# to add and subtract 2 matrices 

# create 2 matrices 

mymatrix4 = np.array([[1,2],[2,3],[3,4]])
mymatrix5 = np.array([[1,2],[2,3],[3,4]])

""" now make addition and subtraction of matrices """

result1 = mymatrix4 + mymatrix5
print(result1)

result2 = mymatrix4 - mymatrix5
print(result2)


# transpose of matrix i.e switching of matrix over its Rows and columns
# after transpose rows = columns and columns = rows i.e their elements.

# T denotes transpose.

mymatrix6 = np.array([[1,2],[2,3],[3,4]])
print(mymatrix6)

# now print the transpose of matrix 

print(mymatrix6.T) # notice the T here. 

""" transpose matrix will NOT affect 1d array. but if you define an array within an array then
you can get the transpose. 

matrix = np.array([1,2,3]) >> this will not produce transpose
but matrix = np.array([[1,2,3]]) >> this will produce a transpose.
"""

# DOT product 

# only use np.dot and nothing else.

myarray1 = np.array([1,2,3])
myarray2 = np.array([4,5,6])

result3 = np.dot(myarray1, myarray2) # this will give dot product of myarra1 and myarray2
print(result3)


# axis parameter 

""" 
axis=0 means perform something on each column
axis=1 means perform something on each row

"""

