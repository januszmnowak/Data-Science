import numpy as np

"""Assume that the fuction is f(v)= ð£â´ - 5ð£Â² - 3ð£. We want to find minimum of the function
It has a global minimum in ð£ â 1.7 and a local minimum in ð£ â â1.42."""

vector=5 #vector (rozwiazanie) startowy
learn_rate=0.01
n_iter=100
tolerance=0.000001


#gradient funkcji to wartosc pochodnej funkcji dla danego punktu
def gradient(v):
    return 4*v**3-10*v-3

for i in range(n_iter):
    diff=-learn_rate*gradient(vector) #obliczam krok jako iloczyn learn_rate i gradientu ze znakiem minus
    if np.all(np.abs(diff)<=tolerance):
        break #jeÅ¼eli krok jest mniejszy od tolernacji to zatrzymuje algorytm
    vector+=diff
    print (i, vector)

