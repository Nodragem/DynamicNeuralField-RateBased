# -*- coding: utf-8 -*-
"""
Created on Tue May 05 13:17:49 2015

@author: c1248317
"""

import dill as pickle
import glob, os, sys
import matplotlib.pyplot as plt
import util.util as util
import numpy as np
cm = plt.get_cmap('gist_rainbow')

print os.chdir(os.path.dirname(__file__) )
path_result = ".\\GainEffect_Results\\sigmoid-bigscale\\"
# path_result = ".\\GainEffect_Results\\baseline-shift-bigscale\\"
print glob.glob(path_result+"r*.p")
print util.sort_nicely(glob.glob(path_result+"*.p"))

plt.figure()
for i, name in enumerate(glob.glob(path_result+"*.p")): 
    if i > 5:
        break;
    print name
    ax = plt.subplot(2,3, i+1 )
    r = pickle.load(open(name, "rb"))
    val1 = r["data"]["P1"]["value"][0] # first distance
    val2 = r["data"]["P2"]["value"][0]
    plt.plot(val1, color="red")
    plt.plot(val2, color="blue")
    plt.legend(["P1", "P2"])
plt.show()

plt.figure(figsize=(7.5,6))
for i, name in enumerate(util.sort_nicely(glob.glob(path_result+"r*.p"))):
    if i > 5:
        break;
    print name
    ax = plt.subplot(2,3, i+1 )
    r = pickle.load(open(name, "rb"))
    if "baseline" not in r.keys():
        plt.title("Gain rate %1.3f \n Baseline: %d"%(r["gain"], 100))
    else:
        plt.title("Gain rate %1.3f \n Baseline: %d"%(r["gain"], r["baseline"]))
    #plt.title("Gain rate %1.3f \n Baseline: %d"%(r["gain"], r["baseline"]))

    print r["data"]["distance"][:]
    map_FR = np.array(r["data"]["last_firingmap"][:])
    print map_FR.shape
    plt.imshow(map_FR, vmin = 0, vmax = 600, aspect="auto")
    ax.set_yticklabels([str(t*5) for t in ax.get_yticks()])
    #plt.plot(np.array(r["data"]["P1"]["pos"][:]))
    plt.hlines(140/5, 0, 1000)

    
# plt.figure()
# ax = plt.subplot(1, 1, 1 )
# gains = []
# ax.set_color_cycle([cm(1.*i/6) for i in range(6)])
# for i, name in enumerate(glob.glob(path_result+"*.p")):
#     r = pickle.load(open(name, "rb"))
#     gains.append("gain rate: "+str(r["gain"]))
#     distances =  r["data"]["distance"][:]
#     sum_FR = np.array(np.sum(r["data"]["last_firingmap"][:], axis=1))
#     plt.plot(distances, sum_FR/60000.0)
    
#plt.legend(gains)
plt.tight_layout()
plt.show()