import pandas as pd 
import csv 
import os 
import numpy as np 

from algml import lucas_test, fermat_test, is_prime

path = os.path.join("data", "train", "primes.csv")

def create_prime_df(path):
    """
    """

    primes = []
    with open(path, 'r') as primes_csv:
        reader = csv.reader(primes_csv)
        for prime in reader:
            primes = prime
    nums = [n for n in range(1, len(primes) + 1)]

    prime_df = pd.DataFrame({
        "num": nums,
        "prime": primes
    })
    return prime_df

def mean_gap(num):
    import math
    num = int(num)
    return math.log2(num)

primes_df = create_prime_df(path)
primes_df["mean_gap"] = primes_df["num"].apply(lambda x: mean_gap(x))
primes_df["lag_1_prime"] = primes_df["prime"].shift(1).fillna(0).astype('int')
primes_df["prime"] = primes_df["prime"].astype('int')
primes_df["lag_sum"] = primes_df["lag_1_prime"] + primes_df["mean_gap"]

# Step 1, let's predict the next prime given size of gap, and last prime. We'll use different kernels for experiment
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(primes_df[["num", "mean_gap", "lag_1_prime", "lag_sum"]].values, primes_df["prime"].values, test_size=0.33, random_state=42)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
gpr = GaussianProcessRegressor(alpha=1e-10,                                 
                     copy_X_train=True,                             
                     kernel = Matern() + 1*RBF(1),                            
                     n_restarts_optimizer=10,                        
                     normalize_y=False
).fit(X_train, y_train)

accuracy = gpr.score(X_test, y_test)
print(accuracy)

print(gpr.predict([[200, 0, 0, 0]]))

# Not a bad score! We need to add a search functionality since these are floats we'll try to use regression
# to predict upper and lowre bound and use Miller rabin test to search witin that range

# Next, let's build a classifier 

# FInally, lets use Bayeasian techniques to try compute miller coefficient for fun

# Ok and here for heart of project we create implementation of MERPS search on Databricks!! Distributed prime search
# Enjoy.

# read in hyperopt values
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

# define the values to search over for n_estimators
search_space = hp.normal('a', 1.24055470525201424067, 0.001)

import math 
# define the function we want to minimise
def objective(a):
    score = sum([1 if fermat_test(math.floor(a ** (2 ** n))) else 0 for n in range(2, 10)])
    return {'loss': -score, 'status': STATUS_OK}

best_params = fmin(
  fn=objective,
  space=search_space,
  max_evals=20)


  # stochastic backtracking approach Sample from generative GPT model and use backtracking on digits to reach next prime
