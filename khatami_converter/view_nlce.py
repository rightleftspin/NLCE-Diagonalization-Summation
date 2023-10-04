import json, sys, time, copy, os, pickle, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def open_dat_file(file_name):
    """
    Takes in a dat file from khatami's fortran code and turns it into a workable dataset
    """
    dat_matrix = np.genfromtxt(file_name)
    
    seperated_by_mu = np.vsplit(dat_matrix, 300)
    t_array = np.copy(seperated_by_mu[0][:, 1])

    return(t_array, seperated_by_mu)


order_9 = "Bare_Obsrvbls_9_site.dat"
temperature_9, data_9 = open_dat_file(order_9)

order_8 = "Bare_Obsrvbls_8_site.dat"
temperature_8, data_8 = open_dat_file(order_8)

order_7 = "Bare_Obsrvbls_7_site.dat"
temperature_7, data_7 = open_dat_file(order_7)

plt.figure()
plt.xlabel("Temperature")
plt.ylabel("entropy")
plt.xscale("log")
plt.xlim([0.1, 7.5])
plt.ylim([0, 2])
print(data_9[20][0, 0])
plt.plot(temperature_9, data_9[20][:, 11], label = "9")
plt.plot(temperature_8, data_8[20][:, 11], label = "8")
plt.plot(temperature_7, data_7[20][:, 11], label = "7")
plt.legend()
plt.savefig("entropy_temp.pdf")
