import math
import random

def factor(number):
    """
    Factor number
    """
    factors=[]
    for i in range(1,num+1):
        if number %i == 0:
            factors.append(i)
    return factors 

def is_prime(number):
    # if number is equal to or less than 1, return False
    if number <= 1:
        return False

    for x in range(2, number):
        # if number is divisble by x, return False
        if not number % x:
            return False
    return True

def is_odd(number):
    return number % 2 + 1 == 0

def generate_primes(n):
  """
  Generator over infinite set of prime numbers
  """
  count = 1
  primes = [] 
  odds = []
  while True:
    if count >= n:
      break
    if is_prime(count):
      primes.append(count)
      count += 1
  return primes

def is_prime(n, k=5): # miller-rabin
    from random import randint
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29]:
        if n % p == 0: return n == p
    s, d = 0, n-1
    while d % 2 == 0:
        s, d = s+1, d/2
    for i in range(k):
        x = pow(randint(2, n-1), d, n)
        if x == 1 or x == n-1: continue
        for r in range(1, s):
            x = (x * x) % n
            if x == 1: return False
            if x == n-1: break
        else: return False
    return True

def legendre(arr, p):
    e = (p - 1) // 2
    results = [pow(a, e, p) for a in arr]
    return [(r-p if r > 1 else r) for r in results]

def gcd(x, y):
    while(y):
       x, y = y, x % y
    return abs(x)

def fermat_test(n):
    from random import randint
    a = randint(2, n)
    g = gcd(a, n)
    while g != 1:
        a = randint(2, n)
        g = gcd(a, n)
    return pow(a, n-1, n) == 1
 
 # Function to generate prime factors of n
def primeFactors(n, factors):
     
    # If 2 is a factor
    if (n % 2 == 0):
        factors.append(2)
         
    while (n % 2 == 0):
        n = n // 2
         
    # If prime > 2 is factor
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if (n % i == 0):
            factors.append(i)
             
        while (n % i == 0):
            n = n // i
             
    if (n > 2):
        factors.append(n)
         
    return factors
     
# This function produces power modulo
# some number. It can be optimized to
# using
def power(n, r, q):
     
    total = n
     
    for i in range(1, r):
        total = (total * n) % q
         
    return total
  
def lucas_test(n):
  
    # Base cases
    if (n == 1):
        return False 
    if (n == 2):
        return True
    if (n % 2 == 0):
        return False
          
    # Generating and storing factors
    # of n-1
    factors = []
     
    factors = primeFactors(n - 1, factors)
  
    # Array for random generator. This array
    # is to ensure one number is generated
    # only once
    rand = [i + 2 for i in range(n - 3)]
          
    # Shuffle random array to produce randomness
    random.shuffle(rand)
  
    # Now one by one perform Lucas Primality
    # Test on random numbers generated.
    for i in range(n - 2):
        a = rand[i]
         
        if (power(a, n - 1, n) != 1):
            return False 
  
        # This is to check if every factor
        # of n-1 satisfy the condition
        flag = True
         
        for k in range(len(factors)):
             
            # If a^((n-1)/q) equal 1
            if (power(a, (n - 1) // factors[k], n) == 1):
                flag = False
                break
  
        # If all condition satisfy
        if (flag):
            return True 
     
    return False 

