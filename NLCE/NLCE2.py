import time
import pynauty

# Gobal Variables that need to be defined #
sqrAdj = [(1, 0), (-1, 0), (0, 1), (0, -1)]
graphDict = {}
isoHashSet = set()
symHashSet = set()
###########################################

def symmetricHashFunction(graph):
    # This function takes a graph as an input and returns a hash that corresponds
    # uniquely to the shape of the graph. Namely, it converts it to minimum 
    # vertex type numbering and hashses the minimum vertex type tuple
    global sqrAdj
    vertexTypeGraph = ()
    # Graph needs to be ordered so that the vertex type tuple is truly
    # unique for each graph
    orderedGraph = sorted(graph)
    # Determine the vertex number for each node in the ordered graph
    for node in orderedGraph:
        nodeNumber = 0
        # For the given node, test to see if there are any nodes adjecent to it
        # in the given directions and label it according to the adjecencies
        for index, adjNode in enumerate(sqrAdj):
            if (node[0] + adjNode[0], node[1] + adjNode[1]) in graph:
                nodeNumber += 2 ** index

        vertexTypeGraph += (nodeNumber, )
    # Return a hash of vertex type labelling tuple 
    return(hash(vertexTypeGraph))

def isomorphicHashFunction(graph):
    ######### NEEDS MORE OPTIMIZATION ############
    global sqrAdj
    graphList = list(graph)
    graphObj = pynauty.Graph(len(graph))
    for index, node in enumerate(graphList):
        for adjNode in sqrAdj:
            checkNode = (node[0] + adjNode[0], node[1] + adjNode[1])
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

def enumerateGraph(graph, size, startingVertex):
    # This is a simple wrapper function for the recursive subgraph generator, vSimple
    # Initialize with an empty guarding set
    guardingSet = set()
    # Initialize with the neighbors of the starting vertex
    neighbors = graph[startingVertex]
    # start running vSimple
    vSimple(graph, {startingVertex}, neighbors, guardingSet, size)


def generateSquareLattice(size):
    global sqrAdj
    testVert = set()
    testGraph = {}
    for x in range(size):
        for y in range(size):
            testVert.add((x, y))
            testGraph[(x, y)] = set()
    
    for vert in testVert:
        for adj in sqrAdj:
            edge = (vert[0] + adj[0], vert[1] + adj[1])
            if edge in testVert:
                testGraph[vert].add(edge)

    return(testGraph)

testGraph = generateSquareLattice(11)

start = time.time()
enumerateGraph(testGraph, 6, (5, 5))
print(time.time() - start)
print(len(graphDict))
