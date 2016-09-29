import matplotlib.pyplot as plt
from matplotlib import animation
from util.Generators import *
import shutil
import util.util as util
import os
import dill as pickle

current_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_folder)
path_result = "./GainEffect_Results/baseline-shift-newnoise/"
shutil.copy2(os.path.realpath(__file__), path_result+"copy-code.py")
simulation_time = 1000 #frames
## Note that we use two stimuli which are not equal in strength
## equal competition doesn't work well when sigmoid gain function + method of Euler for integration.

## --- AIMS ---
# Effect of the gain of neurons on the Global effect?
#distances = np.arange(0, 200, 5)
#distances = np.arange(0, 400, 10)
distances = np.arange(0, 800, 10) # here you will see the bimodal responses
#tested_gain_rate = [100.0, 200.0, 250.0, 300.0, 350.0, 400.0]
#tested_gain_rate = [170.0, 180.0, 190.0, 210.0, 220.0, 230.0]
gain_rate = 0.07 #np.arange(0.005, 0.011, 0.001) #[0.0075]#, 0.07, 0.11]
baseline_offsets = [0, 25, 50, 75, 100, 200] #[100]
result_by_gain_rate = []
SHOW_CONNECTIONS = True

from time import time
for n, baseline_offset in enumerate(baseline_offsets):
    dico_result = {}
    for k, distance in enumerate(distances):
        print "\r Run of distance %d and baseline %d ..."%(distance, baseline_offset),
        t0 = time()
        ## -- Input and recording configurations:
        GC1 = GeneratorConstante(281.25*1.0 + np.random.normal(0, 1, size=x)) ## this input is slightly stronger than GC2, it represent firing rate.
        ## note that with Trappenberg parameters: an exact equality still lead to a winner (miracle of float precision?)
        GC1.color = [1,0,0]
        GC2 = GeneratorConstante(250.0*1.0)
        ## here you choose the position of the input on the map
        position1 = 500 - distance/2 # 400/600: winner-take-all, 450/550: auto-inhibition, 480/520: merging
        s1 = 30.0
        position2 = 500 + distance/2
        s2 = 30.0
        width = 1000
        NF = NeuralField1D(width, 100) ## time constant of 100 just to slow the process down and see it on the animation
        NF.u0 = baseline_offset ##Trappenberg biases the equilibirum state of build-up neurons, we don't
        NF.getProbe(0).position = position1
        NF.getProbe(0).color = [1.0, 0.7, 0.2]
        NF.getProbe(0).name = "P1"
        NF.addProbe("P2", position2).color = [0.2, 0.7, 1.0]
        NF.addProbe("Center", (position2+position1)/2 ).color = [0.7, 0.2, 1.0]

        ## uncomment this line to get noise:
        NF.noise = lambda x: np.random.normal(0, 1, size=x)*300 #*3000 # 1000
        # sigmoid test:
        NF.beta = gain_rate
        NF.firing_offset = 350
        # exponential test:
        # --- Definition of the Gain function:
        # --- Negative Exponential (as a capacitor function)
        # But we can also change the gain function and use an inverse exponential:
        #def exponentialRise(u):
        #    global gain_rate
        #    tau_fr = gain_rate
        #    u_offset = 150
        #    fr = 500.0*(1-np.exp(-(u-u_offset)/tau_fr))
        #    fr[fr<0] = 0
        #    return fr
        # you just need to replace the function in the neural field NF:
        #NF.GainFunction = exponentialRise

        ## -- Definition of the Connections:
        ## It seems that the parameters of Trappenberg was way to strong (but note that we have more neurons in our model)
        coupling = 1/10.
        Sa = 60.0; A = coupling*144.0/200;  ## Sa = 0.6mm while 1 neurons is 0.01
        Sb = 180.0; B = coupling*44.0/160;
        c = coupling*16.0/200;
        #c = coupling*1.0/200;

        # low coupling (not possible, the lowest was already 144.0/200/5)
        #Sa = 2*60.0; A = 144.0/200/5/2;  ## Sa = 0.6mm while 1 neurons is 0.01
        #Sb = 2*180.0; B = 44.0/100/5/2;
        #c = 16.0/100/5/2;

        ## flat inhibition
        #Sa = 2*30.0; A = 144.0/200/5;  ## Sa = 0.6mm while 1 neurons is 0.01
        #Sb = 2*180.0; B = 0;
        #c = 50.0/100/5;
        ## I create the lateral connections and attached them to the neural field
        def DoF(x, Sa, Aa, Sb, Ab, c):
            Ga = Aa*(np.exp(-(x**2)/(2*Sa**2)))
            Gb = Ab*(np.exp(-(x**2)/(2*Sb**2)))
            DoG = Ga - Gb - c
            return DoG

        connections = lambda x: DoF(x, Sa, A, Sb, B, c) ## that my connection function
        NF.createConnectionCurve(connections) ## I give my connection function to my Neural Field

        ## I connect the Contanst Generator to my NeuralField as external input
        #w1 = np.zeros(width); w1[position1-s1/2.0:position1+s1/2.0] = 1.0
        w1 = util.getGaussian(np.arange(width), 1.0, position1, s1)
        W1 = NF.set_external(w1, GC1)
        #w2 = np.zeros(width); w2[position2-s2/2.0:position2+s2/2.0] = 1.0
        w2 = util.getGaussian(np.arange(width), 1.0, position2, s2)
        W2 = NF.set_external(w2, GC2)

        list_obj = [GC1, GC2, NF] ## they need to be updated at each time step

        # I plot the connection function and the gain function:
        if (n == 0) and (k==0) and SHOW_CONNECTIONS:
            plt.figure()
            plt.subplot(211)
            plt.title("Close the window to start the simulation... \n Connections Pattern")
            s = NF.myConnection.shape[0]/2
            plt.plot(NF.myConnection[s-width/2:s+width/2], color = "pink", linewidth=2.0)
            plt.xlabel("neurons")
            plt.hlines(0, 0, width, color="gray", linestyle="--")
            plt.subplot(212)
            plt.plot(NF.GainFunction(np.arange(0,2000)), color = "purple")
            plt.xlabel("Current/Membrane potential")
            plt.ylabel("Firing Rate (Hz)")
            plt.title("Gain Function")
            plt.tight_layout()
            plt.savefig("ConnectionD%dG%1.3f.png"%(distance, gain_rate))
            plt.show(block=True)

        ## I define which probes we want to plot:
        displayed_probes = ["Default", "P1", "P2"]

        # -- Loop section:
        ## this for-loop is the simulation:
        for i in xrange(simulation_time):
            for o in list_obj:
                o.update()

        ## After the simulation we:
        ## --> extract the last firing rate of neurons.
        firing_map = NF.GainFunction(NF.V)
        if k == 0: ## if first distance, we initialaze a dictionnary to save the results.
            dico_result = {"distance": [], "last_firingmap": []}
            for probe in NF.getProbes():
                dico_result[probe.name] = {"pos":[], "time":[], "value":[]}

        ## --> we fill our dictionnary with the results of the simulations:
        dico_result["distance"].append(distance)
        dico_result["last_firingmap"].append(firing_map)
        for probe in NF.getProbes():
            dico_result[probe.name]["pos"].append(probe.position)
            dico_result[probe.name]["time"].append(probe.time_buffer)
            dico_result[probe.name]["value"].append(probe.value_buffer)

        ## --> we display the duration:
        t1 = time()
        #print((t1 - t0))
    ## -- > for each gain, we save a dictionnary with the result for each distances
    pickle.dump({"baseline":baseline_offset,"gain": gain_rate, "data":dico_result}, open( path_result+"resultB%d.p"%baseline_offset, "wb" ) )
    # plt.subplot(211)
    # plt.imshow(np.array(dico_result["last_firingmap"][:]), aspect="auto")
    # plt.subplot(212)
    # for c1, c2 in zip(dico_result["P2"]["value"], dico_result["P1"]["value"]):
    #     plt.plot(np.array(c1)-np.array(c2))
    # plt.show()