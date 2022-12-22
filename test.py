# Testing pynauty stuff?

import time
import pynauty
import copy

g1 = pynauty.Graph(5)
g1.connect_vertex(1, [2, 3])
g1.connect_vertex(2, [3])
g1.connect_vertex(3, [])

#g2 = pynauty.Graph(5)
#g2.connect_vertex(1, [2, 3])
#g2.connect_vertex(2, [1, 3])
#g2.connect_vertex(3, [1, 2])

g2 = pynauty.Graph(5)
g2.connect_vertex(2, [3, 4])
g2.connect_vertex(3, [4])
g2.connect_vertex(4, [])

#if pynauty.certificate(g1) == pynauty.certificate(g2):
#    print("they are the same")

#print(hash(pynauty.certificate(g1)))
#print(hash(pynauty.certificate(g2)))

#print(g2.adjecency_dict)
#print(g2.adjacency_dict)

#if 2 in g2.adjacency_dict:
#    print("TRUE")

#print(sorted(g2.adjacency_dict))

g3 = g2.copy()
g3.connect_vertex(1, [2, 3])
print(g2)
print(g3)





def f(x):
    return(x*10205)

from multiprocessing import Pool, cpu_count
p = Pool(processes = cpu_count())
print(f"Core Count: {cpu_count()}")

l = range(0, 100000)

start = time.time()
mapped1 = p.map(f, l)
print(f"Total time is {time.time() - start:.9f}")

start = time.time()
mapped2 = map(f, l)
print(f"Total time is {time.time() - start:.9f}")

print(list(mapped1)[0:5])
print(list(mapped2)[0:5])
