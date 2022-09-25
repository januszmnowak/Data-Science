import numpy as np

"""Assume we have regression: 𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥
We need to find the values of weights 𝑏₀, 𝑏₁, that minimize the sum of squared residuals SSR = Σᵢ(𝑦ᵢ − 𝑓(𝐱ᵢ))² or the cost function 𝐶 = SSR / (2𝑛), which is mathematically more convenient than SSR"""

"""We need calculus to find the gradient of the cost function 𝐶 = Σᵢ(𝑦ᵢ − 𝑏₀ − 𝑏₁𝑥ᵢ)² / (2𝑛). Since you have two decision variables, 𝑏₀ and 𝑏₁, the gradient ∇𝐶 is a vector with two components:
∂𝐶/∂𝑏₀ = (1/𝑛) Σᵢ(𝑏₀ + 𝑏₁𝑥ᵢ − 𝑦ᵢ) = mean(𝑏₀ + 𝑏₁𝑥ᵢ − 𝑦ᵢ) = mean(residual)
∂𝐶/∂𝑏₁ = (1/𝑛) Σᵢ(𝑏₀ + 𝑏₁𝑥ᵢ − 𝑦ᵢ) 𝑥ᵢ = mean((𝑏₀ + 𝑏₁𝑥ᵢ − 𝑦ᵢ) 𝑥ᵢ) = mean(residual*x)
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
        break #jeżeli krok jest mniejszy od tolernacji to zatrzymuje algorytm
    vector+=diff
    print (i, vector)
