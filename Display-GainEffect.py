# -*- coding: utf-8 -*-
"""
Created on Tue May 05 13:17:49 2015

@author: c1248317
"""

import dill as pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
cm = plt.get_cmap('gist_rainbow')

path_result = ".\\GainEffect_Results\\flatConnection\\"
plt.figure()
for i, name in enumerate(glob.glob(path_result+"*.p")):    
    print name
    ax = plt.subplot(2,3, i+1 )
    r = pickle.load(open(name, "rb"))
    plt.title("Gain rate %d"%r["gain"])
    print r["data"]["distance"][:]
    map_FR = np.array(r["data"]["last_firingmap"][:])
    print map_FR.shape
    plt.imshow(map_FR, vmin = 0, vmax = 500, aspect="auto")
    ax.set_yticklabels([str(t*6) for t in ax.get_yticks()])
    #plt.plot(np.array(r["data"]["P1"]["pos"][:]))
    plt.show()
    
plt.figure()    
ax = plt.subplot(1, 1, 1 )
gains = []
ax.set_color_cycle([cm(1.*i/6) for i in range(6)])
for i, name in enumerate(glob.glob(path_result+"*.p")):    
    r = pickle.load(open(name, "rb"))
    gains.append("gain rate: "+str(r["gain"]))
    distances =  r["data"]["distance"][:]
    sum_FR = np.array(np.sum(r["data"]["last_firingmap"][:], axis=1))
    plt.plot(distances, sum_FR/60000.0)
    
plt.legend(gains)
plt.show()