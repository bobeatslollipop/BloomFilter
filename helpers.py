import random
import math

# Generating hash functions via index.
# See https://stackoverflow.com/questions/2255604/hash-functions-family-generator-in-python#:~:text=The%20universal%20hash%20family%20is,drawn%20randomly%20from%20set%20H%20.
def universal_hashing():
    def rand_prime():
        while True:
            p = random.randrange(2 ** 32, 2 ** 34, 2)
            if all(p % n != 0 for n in range(3, int((p ** 0.5) + 1), 2)):
                return p

    m = 2 ** 32 - 1
    p = rand_prime()
    a = random.randint(0, p)
    if a % 2 == 0:
        a += 1
    b = random.randint(0, p)

    def h(x):
        return ((a * x + b) % p) % m

    return h

def round(x: float) -> int:
    return int(x+0.5)