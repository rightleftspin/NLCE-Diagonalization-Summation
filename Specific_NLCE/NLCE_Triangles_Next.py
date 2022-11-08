import time
import math

import pynauty

# Three Global Variables to be decided upon
clusterSize = int(input("Cluster Size? "))
actualSize = clusterSize
clusterSize = clusterSize * 2
leftAdjecencyList = [1, 0, clusterSize, -clusterSize, clusterSize + 1, 0, (2 * clusterSize + 1), 0, clusterSize + 2, 0, 0, -clusterSize + 1]
rightAdjecencyList = [0, -1, clusterSize, -clusterSize, 0, -(clusterSize + 1), 0, clusterSize - 1, 0, -(clusterSize + 2), -(2 * clusterSize + 1), 0]
totalAdjecencyList = [1, -1, clusterSize, -clusterSize, clusterSize + 1, -(clusterSize + 1), (2 * clusterSize + 1), clusterSize - 1, clusterSize + 2, -(clusterSize + 2), -(2 * clusterSize + 1), -clusterSize + 1]
##########################################

# Other Global Variables, do not touch
leftEdge = set(range(0, clusterSize ** 2, clusterSize))
rightEdge = set(range(clusterSize - 1, clusterSize ** 2, clusterSize))
clusterDict = {}
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

listLattice = generateLatticeList(clusterSize)
setLattice = generateLatticeSet(clusterSize)
startingPoint = clusterSize//2 + ((clusterSize//2) * clusterSize)
#########################################

def symmetricHashFunction(graph):
    global listLattice

    # Initialize the vertex type graph tuple
    vertexTypeGraph = ()
    # Sort the graph so that the graph is oriented along strips of the x axis
    orderedGraph = sorted(graph)
    for vertex in orderedGraph:
        vertexNumber = 0
        # Get all the possible points near the vertex
        adjVertices = listLattice[vertex]
        for index, adjVertex in enumerate(adjVertices):
            # Check if the adjecent vertex is actually in the graph
            if (adjVertex != vertex) and (adjVertex in graph):
                # Add to the vertex number based on the position in the vertex list
                vertexNumber += 2 ** index

        # add the vertex type to the vertex tuple
        vertexTypeGraph += (vertexNumber, )
    # return unique symmetric hash
    return(hash(vertexTypeGraph))

def isomorphicHashFunction(graph):
    global setLattice
    # Turn graph into a list so that each number is associated with a specifc index
    graphList = list(graph)
    # Create a pynauty graph object corresponding to the graph
    graphNauty = pynauty.Graph(len(graph))
    for index, vertex in enumerate(graphList):
        # Get adjecent vertices to vertex
        adjVertices = setLattice[vertex]
        for adjVertex in adjVertices:
            # Check if the adjecent vertex is actually in the graph
            if adjVertex in graph:
                # connect vertex to each vertex that it can be connected to
                graphNauty.connect_vertex(index, graphList.index(adjVertex))

    # Return the hash of the pynauty certificate of the graph
    return(hash(pynauty.certificate(graphNauty)))

def addSubgraph(isoHash, subgraph):
    global clusterDict
    # Get the isomorphic hash of the subgraph
    isoHashSubgraph = isomorphicHashFunction(subgraph)
    # Check if the subgraph is already counted for
    if isoHashSubgraph in clusterDict[isoHash][2]:
        # if it is, increase its multiplicity
        clusterDict[isoHash][2][isoHashSubgraph][1] += 1
    else:
        # if not, add it to the subgraph
        clusterDict[isoHash][2][isoHashSubgraph] = [subgraph, 1]

    return(None)

def clusterConnection(cluster):
    # This function takes a cluster and returns its corresponding cluster
    # dictionary, meaning the vertices as keys and the edges as values
    global setLattice
    # Initialize an empty graph dictionary
    clusterGraph = {}
    for vertex in cluster:
        # Initialize the vertex with an empty set
        clusterGraph[vertex] = set()
        # Loop over all the possible adjecent verticies
        for adjVertex in setLattice[vertex]:
            # Check if the vertex is in the cluster
            if adjVertex in cluster:
                # add to the graph as an edge if it is
                clusterGraph[vertex].add(adjVertex)

    return(clusterGraph)

def addCluster(cluster):
    # This function takes a cluster and adds it to the cluster dictionary if it doesn't already exist
    global clusterDict
    global isoHashSet
    global symHashSet

    # Take the isomorphic hash of the cluster
    isoHash = isomorphicHashFunction(cluster)
    # Check if the cluster is isomorphic to an already cataloged cluster
    if isoHash not in isoHashSet:
        # Add the isomorphic and symmetric hashes to the set of existing isomorphic and symmetric hashes
        isoHashSet.add(isoHash)
        symHashSet.add(symmetricHashFunction(cluster))
        # Add the cluster to the dictionary along with an initial multiplicity of 1 and an empty
        # subgraph dictionary
        clusterDict[isoHash] = [cluster, 1, {}]

        # Add subgraphs of the cluster to the cluster dictionary
        addSubgraphsForCluster = lambda subgraph: addSubgraph(isoHash, subgraph)
        # Add edges to the cluster
        clusterGraph = clusterConnection(cluster)
        for size in range(2, len(cluster)):
            # add all clusters of given size ranging from 2 up till the size of the cluster - 1
            enumerateGraph(clusterGraph, size, cluster, addSubgraphsForCluster)
    else:
        # If it is already isomorphic to something, it might be symmetric to something as well
        # compute the symmetric hash
        symHash = symmetricHashFunction(cluster)
        # If it is symmetric to an existing graph in the hash set, ignore it
        if symHash not in symHashSet:
            # If it is a new graph, increase the multiplicity counter for its graph dictionary entry
            symHashSet.add(symHash)
            clusterDict[isoHash][1] += 1

    return(None)

def vSimple(graph, subgraph, neighbors, guardingSet, size, graphFunc):
    # This is a recursive algorithm that takes a graph and breaks it up into all possible subgraphs
    # of a specific order labelled by the size

    # Start by checking to see if your subgraph is already the proper size
    if len(subgraph) == size:
        # If it is, try adding it to the graph dictionary
        graphFunc(subgraph)    
        return(True)

    hasIntLeaf = False
    # Loop over all the neighbors for the current node
    neighborIterator = neighbors.copy()
    guardingSetClone = guardingSet.copy()

    while (len(neighborIterator) > 0):
        # create a subgraph by adding the neighbor
        neighbor = neighborIterator.pop()
        newSubgraph = subgraph | {neighbor}
        # Add all the neighbors of the new node that we just added to the new subgraph and
        # take out any neighbors in the guarding set and that are already in the subgraph
        addNeighbors = graph[neighbor].difference(subgraph).difference(guardingSetClone)
        # Create a new set of neighbors that is a combination of the old set of neighbors
        # and the new set of neighbors from the new node
        newNeighbors = neighborIterator | addNeighbors
        # Recursively call this algorithm again with the new subgraph and new neighbor set
        end = vSimple(graph, newSubgraph, newNeighbors, guardingSetClone, size, graphFunc)
        if end:
            hasIntLeaf = True
        else:
            return(hasIntLeaf)
        # Update the guarding set to include the neighbor
        guardingSetClone.add(neighbor)

        # If the guarding set ever gets too big, break out of this loop
        if (len(graph) - len(guardingSetClone)) < size:
            return(hasIntLeaf)

    return(hasIntLeaf)

def enumerateGraph(graph, size, startingVertices, graphFunc):
    # This is a simple wrapper function for the recursive subgraph generator, vSimple
    # Initialize with an empty guarding set
    guardingSet = set()

    for vertex in startingVertices:
        # Initialize with the neighbors of the starting vertex
        neighbors = graph[vertex]
        # start running vSimple
        vSimple(graph, {vertex}, neighbors.difference(guardingSet), guardingSet.copy(), size, graphFunc)
        #print(guardingSet)
        # add vertex to guardingSet
        guardingSet = guardingSet | {vertex}

    return(None)
        
start = time.time()
enumerateGraph(setLattice, actualSize, {startingPoint}, addCluster)
print(f"Total time to enumerate graph is {time.time() - start}")

# Metrics to look at to make sure everything makes sense
totalGraphs = 0
totalBrokenDown = 0
for isoHash in clusterDict:
    totalGraphs += clusterDict[isoHash][1]
    totalBrokenDown += len(clusterDict[isoHash][2])


print(f"Isomorphically Distinct: {len(clusterDict)}")
print(f"Total Graphs: {totalGraphs}")
print(f"Total Graphs x Isomorphic Subclusters: {totalBrokenDown}")
########################################################

#for isoHash in clusterDict:
#    print(clusterDict[isoHash][0])
#    print("-----------------------------")
#    for subgraph in clusterDict[isoHash][2]:
#        print(clusterDict[isoHash][2][subgraph][0])
#        print(clusterDict[isoHash][2][subgraph][1])
#    print("-----------------------------")
