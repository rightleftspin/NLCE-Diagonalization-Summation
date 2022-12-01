import matplotlib.pyplot as plt
import numpy as np
import math

# Define some global variables
granularity = .01
energy = -4 
temp = 1

muRangeList = list(np.arange(-5, 5, granularity))
nuRangeList = list(np.arange(-5, 5, granularity))
####################################################################

def calculateEnergy(mu,nu,E,temp):
    global granularity
    enArr = [(nu - (2 * mu)), -mu, -mu , 0]
    norm = 0
    enCalc = 0
    for i in enArr:
        norm  = norm + math.exp(-i/temp)
        enCalc = enCalc + (i * math.exp(-i/temp))

    if (abs(E - (enCalc/norm)) < 2):
        return True
    else:
        return False

#####################################################################

def iterateOverMu(muRange1, nuConstant):
    global energy, temp
    tempFunc = lambda var: calculateEnergy(var, nuConstant, energy, temp)
    tryVarRange = list(filter(tempFunc, muRange1))
    if len(tryVarRange) > 0:
        return(sum(tryVarRange)/len(tryVarRange))
    else:
        raise TypeError("What 1?")

muLoop = lambda nu: iterateOverMu(muRangeList, nu)  
muFuncNu = list(map(muLoop, nuRangeList))

plt.plot(nuRangeList, muFuncNu)
plt.savefig("mufuncnu.pdf")
plt.close()

#####################################################################

def iterateOverNu(nuRange1, muConstant):
    global energy, temp
    tempFunc = lambda var: calculateEnergy(muConstant, var, energy, temp)
    tryVarRange = list(filter(tempFunc, nuRange1))
    if len(tryVarRange) > 0:
        return(sum(tryVarRange)/len(tryVarRange))
    else:
        raise TypeError("What?")

nuLoop = lambda mu: iterateOverNu(nuRangeList, mu)  
nuFuncMu = list(map(nuLoop, muRangeList))

plt.plot(nuRangeList, muFuncNu)
plt.savefig("nufuncmu.pdf")


