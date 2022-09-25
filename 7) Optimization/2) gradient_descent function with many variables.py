import numpy as np

"""Assume that the fuction is f(v1,v2)= 𝑣₁² + 𝑣₂⁴. We want to find minimum of the function"""


vector=np.array([1.0, 1.0]) #vector startowy
learn_rate=0.01
n_iter=10000
tolerance=0.000001


#gradient funkcji to wartosc pochodnej funkcji dla danego punktu
def gradient(v):
    return 2*v[0]+4*v[1]**3

for i in range(n_iter):
    diff=-learn_rate*gradient(vector) #obliczam krok jako iloczyn learn_rate i gradientu ze znakiem minus
    if np.all(np.abs(diff)<=tolerance):
        break #jeżeli krok jest mniejszy od tolernacji to zatrzymuje algorytm
    vector+=diff
    print (i, vector)

