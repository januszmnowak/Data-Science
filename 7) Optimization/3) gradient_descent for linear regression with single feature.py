import numpy as np

"""Assume we have regression: π(π±) = πβ + πβπ₯
We need to find the values of weights πβ, πβ, that minimize the sum of squared residuals SSR = Ξ£α΅’(π¦α΅’ β π(π±α΅’))Β² or the cost function πΆ = SSR / (2π), which is mathematically more convenient than SSR"""

"""We need calculus to find the gradient of the cost function πΆ = Ξ£α΅’(π¦α΅’ β πβ β πβπ₯α΅’)Β² / (2π). Since you have two decision variables, πβ and πβ, the gradient βπΆ is a vector with two components:
βπΆ/βπβ = (1/π) Ξ£α΅’(πβ + πβπ₯α΅’ β π¦α΅’) = mean(πβ + πβπ₯α΅’ β π¦α΅’) = mean(residual)
βπΆ/βπβ = (1/π) Ξ£α΅’(πβ + πβπ₯α΅’ β π¦α΅’) π₯α΅’ = mean((πβ + πβπ₯α΅’ β π¦α΅’) π₯α΅’) = mean(residual*x)
rozwiazanie to vector=[b0,b1]"""

#input data and starting vector
x = np.array([5, 15, 25, 35, 45, 55])
y = np.array([5, 20, 14, 32, 22, 38])
vector=np.array([0.5, 0.5]) #vector startowy

#gradient descent parameters
learn_rate=0.001
n_iter=100000
tolerance=0.00001

def gradient(x,y,vector):
    residual=(vector[0]+vector[1]*x)-y
    return residual.mean(), (residual * x).mean()

for i in range(n_iter):
    diff=-learn_rate*np.array(gradient(x,y,vector)) #obliczam krok jako iloczyn learn_rate i gradientu ze znakiem minus
    if np.all(np.abs(diff)<=tolerance):
        break #jeΕΌeli krok jest mniejszy od tolernacji to zatrzymuje algorytm
    vector+=diff
    print (i, vector)
