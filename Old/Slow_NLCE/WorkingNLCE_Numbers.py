import time
import pynauty
import math
# Gobal Variables that need to be defined #
clusterSize = 3 
coordAdjList = [ (1, 0), (-1, 0), (0, 1), (0, -1) ]
graphDict = {}
isoHashSet = set()
symHashSet = set()
subgraphEnum = {}
subgraphIsoHashSet = set()
###########################################

def convert(coordinate):
    # Converts coordinates from tuple representation to a single number
    global clusterSize
    # take the first coordinate and add it directly
    x = coordinate[0]
    # take the second coordinate and add it a row above
    y = coordinate[1] * clusterSize
    return(x + y)

def convertBack(number):
    # Converts coordinates from number representation to tuple
    global clusterSize
    # find the first coordinate by modding out the number with the cluster size
    x = number % clusterSize
    # find the second coordinate by inverting the y coordinate equation
    y = number // clusterSize
    return((x, y))

# We create the adjecency list here because the rest of the code needs it
# and it needs the convert function above
adjecencyList = list(map(convert, coordAdjList))

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

        vertexTypeGraph += (nodeNumber,)
    # Return a hash of vertex type labelling tuple 
    return(hash(vertexTypeGraph))

def isomorphicHashFunction(graph):
    #### Needs Optimization ####
    # Turns graph into pynauty graph object in order to get the pynauty certificate
    # that uniquely identifies an isomorphically unique graph
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
    global subgraphEnum

    # Take the isomorphic hash of the graph, ideally from nauty, but we will see
    isoHash = isomorphicHashFunction(graph)
    # If the graph is not isomorphic to an existing graph, then add it to the graph dictionary
    if isoHash not in isoHashSet:
        
        # Add the isomorphic and symmetric  hashes to the set of existing isomorphic and symmetric hashes
        isoHashSet.add(isoHash)
        symHashSet.add(symmetricHashFunction(graph))
        # Add the graph to the dictionary
        enumerateSubgraph(graph)
        graphDict[isoHash] = [graph, 1, subgraphEnum]
        subgraphEnum = {}
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

def generateLattice(size):
    # This function will generate a square shaped lattice of given size
    global adjecencyList

    # Create empty vertex sets and coordinate graph dictionary
    vertices = set()
    coordGraph = {}

    # Loop through and intialize all the coordinates into the vertex set
    for x in range(size):
        for y in range(size):
            # Add vertex to verticies set
            vertices.add((x, y))

    # Take all the vertices and add their edges according to the adjecency list
    for vertex in vertices:
        # Initialize vertex with empty set
        coordGraph[vertex] = set()
        # loop through possible adjecencies
        for adj in coordAdjList:
            # Create adjecent edge
            edge = (vertex[0] + adj[0], vertex[1] + adj[1])
            # See if adjecent edge is an actual vertex to be able to connect to
            if edge in vertices:
                # if it is, add it
                coordGraph[vertex].add(edge)

    # Now we need to convert this graph to the numbering system
    # Intialize empty number graph
    numGraph = {}
    # Loop through every vertex in the coordinate graph
    for vertex in coordGraph:
        # First convert the vertex
        numVertex = convert(vertex)
        # Then initialize the converted vertex with an empty set
        numGraph[numVertex] = set()
        # Now loop through all the edges for the vertex
        for edge in coordGraph[vertex]:
            # Convert all the edges as well and add them as edges to this new graph
            numGraph[numVertex].add(convert(edge))

    centralVertex = convert((size//2, size//2))

    return(numGraph, centralVertex)

def enumerateGraph(size):
    # This is a wrapper function that generates a lattice of a certain size
    # and enumerates all of its subgraphs
    # Creates a graph with a certain size
    graph, startingVertex = generateLattice(size)
    print(graph)
    print(startingVertex)
    # Chooses center starting point
    # startingVertex = ((size//2) + (((size**2)//2) - 1))
    # Initialize with an empty guarding set
    guardingSet = set()
    # Initialize with the neighbors of the starting vertex
    neighbors = graph[startingVertex]
    # start running vSimple
    vSimple(graph, {startingVertex}, neighbors, guardingSet, size)

    return()

def addSubgraph(subgraph):
    global subgraphEnum
    global subgraphIsoHashList




def vSimple_subgraph(graph, subgraph, neighbors, guardingSet, size):
    # This is a recursive algorithm that takes a graph and breaks it up into all possible subgraphs
    # of a specific order labelled by the size

    # Start by checking to see if your subgraph is already the proper size
    if len(subgraph) == size:
        # If it is, try adding it to the graph dictionary
        addSubgraph(subgraph)
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

def enumerateSubgraph(subgraph):
    # Takes in a subgraph/subcluster and returns a dictionary keyed by isomorphic hashes
    # this dictionary is of the form { Hash: (subcluster: multiplicity)}
    global coordAdjList
    guardingSet = set()

    coordVertices = set(map(convertBack, subgraph))
    newSubgraph = {}
    for vertex in coordVertices:
        newSubgraph[vertex] = set()
        for adj in coordAdjList:
            edge = (vertex[0] + adj[0], vertex[1] + adj[1])
            if edge in coordVertices:
                newSubgraph[vertex].add(edge)

    numGraph = {}
    # Loop through every vertex in the coordinate graph
    for vertex in newSubgraph:
        # First convert the vertex
        numVertex = convert(vertex)
        # Then initialize the converted vertex with an empty set
        numGraph[numVertex] = set()
        # Now loop through all the edges for the vertex
        for edge in newSubgraph[vertex]:
            # Convert all the edges as well and add them as edges to this new graph
            numGraph[numVertex].add(convert(edge))


    for size in range(1, len(subgraph)):
        guardingSet = {}
        for vertex in numGraph:
            vSimple_subgraph(numGraph, {vertex}, numGraph[vertex].difference(guardingSet), guardingSet, size)
            guardingSet.add(vertex)

    return()



    



        



start = time.time()

enumerateGraph(clusterSize)

print(f"{time.time() - start:0.3f}s")

print(len(graphDict))
print(graphDict)
