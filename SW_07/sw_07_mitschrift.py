from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from scipy.stats import norm


methodeA = Series([5.9, 3.4, 6.6, 6.3, 4.2, 2.0, 6.0, 4.8, 4.2, 2.1, 8.7, 4.4, 5.1, 2.7, 8.5, 5.8, 4.9, 5.3, 5.5, 7.9])

print(methodeA.mean())
print(methodeA.std())

# np.random.seed(seed=42)

# for i in range(5):
#     methodeA_sim1 = Series(np.round(norm.rvs(size=6, loc=80, scale=0.02),2))
#     print(methodeA_sim1)

print(1-norm.cdf(x=5)) # Probability that X is greater than 5 --> equivalent to 1-norm.cdf(x=80+5*0.02, loc=80, scale=0.02) --> when defining loc an sacle seperately 

