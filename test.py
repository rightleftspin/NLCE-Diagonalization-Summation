# Testing pynauty stuff?

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
