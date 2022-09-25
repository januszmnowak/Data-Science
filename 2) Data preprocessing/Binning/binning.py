import numpy as np
import pandas as pd

#binning into defined bins
ages=[20,22,25,27,21,23,37,31,61,45,41,32]

bins=[18,25,35,60,100]

cats=pd.cut(ages, bins, right=False)

print("Cats: \n",cats)
print("Codes: \n",cats.codes)
print("Categories: \n",cats.categories)
print("Counts: \n",pd.value_counts(cats))

#binning into quantiles
data=np.random.randn(10)

cats=pd.qcut(data, q=4, precision=2) #4=quartiles, 10=decicles, etc.

print("Cats: \n",cats)
print("Codes: \n",cats.codes)
print("Categories: \n",cats.categories)
print("Counts: \n",pd.value_counts(cats))