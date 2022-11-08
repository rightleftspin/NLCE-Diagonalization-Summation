import time
import math

import pynauty
import copy


# Three Global Variables to be decided upon
clusterSize = int(input("Cluster Size? "))
leftAdjecencyList = [1, 0, clusterSize, -clusterSize]
rightAdjecencyList = [0, -1, clusterSize, -clusterSize]
totalAdjecencyList = [1, -1, clusterSize, -clusterSize]
###########################################

# Other Global Variables, do not touch
leftEdge = set(range(0, clusterSize ** 2, clusterSize))
rightEdge = set(range(clusterSize - 1, clusterSize ** 2, clusterSize))
graphDict = {}
isoHashSet = set()
symHashSet = set()
######################################

# Lattice Generation, no need to optimize
def getAdjPoints(node):
    global leftAdjecencyList
    global rightAdjecencyList
    global totalAdjecencyList

    global leftEdge
    global rightEdge

    addAdjNode = lambda adj: adj + node

    if node in leftEdge:
        return(list(map(addAdjNode, leftAdjecencyList)))
    elif node in rightEdge:
        return(list(map(addAdjNode, rightAdjecencyList)))
    else:
        return(list(map(addAdjNode, totalAdjecencyList)))

def nodeNeighbors(node, graph):
    neighbors = set()
    adjPoints = getAdjPoints(node)
    for adj in adjPoints:
        if (adj != node) and (adj in graph):
            neighbors.add(adj)

    return(neighbors)

def generateLatticeList(size):
    graph = {}
    vertexList = range(0, size**2)
    for vertex in vertexList:
        graph[vertex] = getAdjPoints(vertex)
    return(graph)

def generateLatticeSet(size):
    graph = {}
    vertexList = range(0, size**2)
    for vertex in vertexList:
        graph[vertex] = set()
        for adj in getAdjPoints(vertex):
            if (adj in vertexList) and (adj != vertex):
                graph[vertex].add(adj)
    return(graph)

listSquareGraph = generateLatticeList(clusterSize)
setSquareGraph = generateLatticeSet(clusterSize)
#########################################

def symmetricHashFunction(graph):
    global listSquareGraph
    vertexTypeGraph = ()
    orderedGraph = sorted(graph)
    for node in orderedGraph:
        nodeNumber = 0
        adjPoints = listSquareGraph[node]
        for index, adjNode in enumerate(adjPoints):
            if (adjNode != node) and (adjNode in graph):
                nodeNumber += 2 ** index

        vertexTypeGraph += (nodeNumber, )
    return(hash(vertexTypeGraph))

def isomorphicHashFunction(graph):
    global listSquareGraph
    graphList = list(graph)
    graphObj = pynauty.Graph(len(graphList))
    for index, node in enumerate(graphList):
        adjPoints = listSquareGraph[node]
        for adjNode in adjPoints:
            if (adjNode != node) and (adjNode in graph):
                graphObj.connect_vertex(index, graphList.index(adjNode)) 

    return(hash(pynauty.certificate(graphObj)))

def addGraph(graph):
    # This function takes a graph and adds it to the graphDictionary if it doesn't already exist
    # Start by calling the global variables to pull from global state
    global graphDict
    global isoHashSet
    global symHashSet

    # Take the isomorphic hash of the graph, ideally from nauty, but we will see
    isoHash = isomorphicHashFunction(graph)
    # If the graph is not isomorphic to an existing graph, then add it to the graph dictionary
    if isoHash not in isoHashSet:

        # Add the isomorphic and symmetric  hashes to the set of existing isomorphic and symmetric hashes
        isoHashSet.add(isoHash)
        symHashSet.add(symmetricHashFunction(graph))
        # Add the graph to the dictionary
        graphDict[isoHash] = [graph, 1]
    else:
        pass
        # If it is already isomorphic to something, it might be symmetric to something as well
        # compute the symmetric hash
        symHash = symmetricHashFunction(graph)
        # If it is symmetric to an existing graph in the hash set, ignore it
        if symHash not in symHashSet:
            # If it is a new graph, increase the multiplicity counter for its graph dictionary entry
            symHashSet.add(symHash)
            graphDict[isoHash][1] += 1

    return(None)

def vSimple(graph, subgraph, neighbors, guardingSet, size):
    # This is a recursive algorithm that takes a graph and breaks it up into all possible subgraphs
    # of a specific order labelled by the size

    # Start by checking to see if your subgraph is already the proper size
    if len(subgraph) == size:
        # If it is, try adding it to the graph dictionary
        addGraph(subgraph)
        return(True)

    hasIntLeaf = False
    # Loop over all the neighbors for the current node
    for neighbor in neighbors:
        # create a subgraph by adding the neighbor
        newSubgraph = subgraph | {neighbor}
        # Add all the neighbors of the new node that we just added to the new subgraph and
        # take out any neighbors in the guarding set and that are already in the subgraph
        addNeighbors = graph[neighbor].difference(newSubgraph).difference(guardingSet)
        # Create a new set of neighbors that is a combination of the old set of neighbors
        # and the new set of neighbors from the new node
        newNeighbors = neighbors.difference({neighbor}) | addNeighbors
        # Recursively call this algorithm again with the new subgraph and new neighbor set
        if vSimple(graph, newSubgraph, newNeighbors, guardingSet, size):
            hasIntLeaf = True
        else:
            break
        # Update the guarding set to include the neighbor
        guardingSet = guardingSet | {neighbor}

        # If the guarding set ever gets too big, break out of this loop
        if (len(graph) - len(guardingSet)) < size:
            break

    return(hasIntLeaf)

def enumerateGraph(graph, size, startingVertices):
    # This is a simple wrapper function for the recursive subgraph generator, vSimple
    # Initialize with an empty guarding set
    guardingSet = set()

    for vertex in startingVertices:
        # Initialize with the neighbors of the starting vertex
        neighbors = graph[vertex]
        # start running vSimple
        vSimple(graph, {vertex}, neighbors.difference(guardingSet), guardingSet, size)
        # add vertex to guardingSet
        guardingSet.add(vertex)

    return()


startingPoint = clusterSize//2 + ((clusterSize//2) * clusterSize)


start = time.time()
enumerateGraph(setSquareGraph, clusterSize, {startingPoint})
print(f"Total time is {time.time() - start}")
print(len(graphDict))

total = 0
for item in graphDict:
    total += graphDict[item][1]

print(total)
