# import math

# def is_prime(n): 
#     if n == 2: return True 
#     if n == 0 or n == 1 or n % 2 == 0:
#         return False 
#     for i in range(3, int(math.sqrt(n))+1, 2): 
#         if n % i == 0: 
#             return False 
#     return True

# op = int(input()) 
# if op >= 2: 
#     print(2, end="|") 
#     for i in range(3, op+1, 2): 
#         if is_prime(i): 
#             print(i, end="|")
def euler_sieve(n):
    is_prime = [True] * (n+1)
    primes = []
    for i in range(2, n+1):
        if is_prime[i]:
            primes.append(i)
        for j in range(len(primes)):
            if i * primes[j] > n:
                break
            is_prime[i*primes[j]] = False
            if i % primes[j] == 0:
                break
    return primes

primes = euler_sieve(20000)
for p in primes:
    print(p, end="|")