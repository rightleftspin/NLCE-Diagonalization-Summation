import time
import pynauty
import math

# Gobal Variables that need to be defined #
clusterSize = 4
adjecencyList = [ 1, -1, (2 * clusterSize) + 1, -((2 * clusterSize) + 1) ]
graphDict = {}
isoHashSet = set()
symHashSet = set()
###########################################

def symmetricHashFunction(graph):
    # This function takes a graph as an input and returns a hash that corresponds
    # uniquely to the shape of the graph. Namely, it converts it to minimum 
    # vertex type numbering and hashses the minimum vertex type tuple
    global adjecencyList
    vertexTypeGraph = ()
    # Graph needs to be ordered so that the vertex type tuple is truly
    # unique for each graph
    orderedGraph = sorted(graph)
    # Determine the vertex number for each node in the ordered graph
    for node in orderedGraph:
        nodeNumber = 0
        # For the given node, test to see if there are any nodes adjecent to it
        # in the given directions and label it according to the adjecencies
        for index, adjNode in enumerate(adjecencyList):
            if (node + adjNode) in graph:
                nodeNumber += 2 ** index

        vertexTypeGraph += (nodeNumber, )
    # Return a hash of vertex type labelling tuple 
    return(hash(vertexTypeGraph))

def isomorphicHashFunction(graph):
    ######### NEEDS MORE OPTIMIZATION ############
    global adjecencyList
    graphList = list(graph)
    graphObj = pynauty.Graph(len(graphList))
    for index, node in enumerate(graphList):
        for adjNode in adjecencyList:
            checkNode = node + adjNode
            if checkNode in graph:
                graphObj.connect_vertex(index, graphList.index(checkNode)) 

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

def generateLattice(size):
    global adjecencyList
    paddingSize = (2 * size) + 1
    vertices = set(range(paddingSize ** 2))
    graph = {}
    for vertex in vertices:
        graph[vertex] = set()
        for adj in adjecencyList:
            edge = vertex + adj
            if edge in vertices:
                graph[vertex].add(edge)

    centerPoint = paddingSize//2 + ((paddingSize//2) * paddingSize)
    return(graph, centerPoint)

graph, centerPoint = generateLattice(clusterSize)

start = time.time()
enumerateGraph(graph, clusterSize, {centerPoint})
print(f"{time.time() - start:0.6f}s")
print(len(graphDict))
