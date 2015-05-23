import matplotlib.pyplot as plt
from matplotlib import animation
from util.Generators import *
import os

current_folder = os.path.dirname(os.path.realpath(__file__))
save_folder = current_folder+"\\video\\"
os.chdir(current_folder)

print "Video will be save in: \n", save_folder

RECORD_VIDEO = False
simulation_time = 1000 #frames
## Note that we use two stimuli which are not equal in strength
## equal competition doesn't work well when sigmoid gain function + method of Euler for integration.

## --- Old Comments ---
## for repulsion only: A = 1.0 // GC1 = 1.0
## for repulsion and selection: A = 1.0 // GC1 = 100.0
## for selection only: A = 0.2 // GC1 = 100.0
## --------------------

## -- Input and recording configurations: 
GC1 = GeneratorConstante(251.0*1.0) ## this input is slightly stronger than GC2, it represent firing rate.  
## note that with Trappenberg parameters: an exact equality still lead to a winner (miracle of float precision?)
GC1.color = [1,0,0]
GC2 = GeneratorConstante(250.0*1.0) 
## here you choose the position of the input on the map
position1 = 450 # 400/600: winner-take-all, 450/550: auto-inhibition, 480/520: merging
s1 = 20.0
position2 = 550
s2 = 20.0 
width = 1000
NF = NeuralField1D(width, 100) ## time constant of 100 just to slow the process down and see it on the animation
NF.u0 = 0 ##Trappenberg biases the equilibirum state of build-up neurons, we don't
NF.getProbe(0).position = position1
NF.getProbe(0).color = [1.0, 0.7, 0.2]
NF.getProbe(0).name = "P1"
NF.addProbe("P2", position2).color = [0.2, 0.7, 1.0]

## uncomment this line to get noise:
#NF.noise = lambda x: np.random.uniform(0, 1, x)*20

## --- Definition of the Gain function:
## --- Sigmoid, reasonnable function:
## Here we use the default gain function, which is a sigmoid defined by beta slope and an x_offset:
# NF.beta = 0.02 
# NF.firing_offset = 350 
## \--> this offset allows the model to be stable when no input (quite important, isn't it?'), it can't fire for negative value of u(t), which seems quite reasonnable. However, this parameters was set to 0 in Trappenberg 2001
## Note: with this set of parameters, you need bigger difference between GC1 and GC2 to diverge to the threshold than with the negative exponential function or Trappenberg parameters.

## --- Definition of the Gain function:
## --- Trappenberg parameters:
# NF.beta = 0.07
# NF.firing_offset = 0

# --- Definition of the Gain function:
# --- Negative Exponential (as a capacitor function)
# But we can also change the gain function and use an inverse exponential:
def exponentialRise(u):
    tau_fr = 200.0
    u_offset = 150
    fr = 500*(1-np.exp(-(u-u_offset)/tau_fr))
    fr[fr<0] = 0
    return fr
# you just need to replace the function in the neural field NF:
NF.GainFunction = exponentialRise

## -- Definition of the Connections:
## It seems that the parameters of Trappenberg was way to strong (but note that we have more neurons in our model)
Sa = 2*60.0; A = 144.0/200/5;  ## Sa = 0.6mm while 1 neurons is 0.01
Sb = 2*180.0; B = 44.0/100/5;
c = 16.0/100/5;
## I create the lateral connections and attached them to the neural field
def DoF(x, Sa, Aa, Sb, Ab, c):
    Ga = Aa*(np.exp(-(x**2)/(2*Sa**2)))
    Gb = Ab*(np.exp(-(x**2)/(2*Sb**2)))
    DoG = Ga - Gb - c
    return DoG

connections = lambda x: DoF(x, Sa, A, Sb, B, c) ## that my connection function
NF.createConnectionCurve(connections) ## I give my connection function to my Neural Field

## I connect the Contanst Generator to my NeuralField as external input
w1 = np.zeros(width); w1[position1-s1/2.0:position1+s1/2.0] = 1.0
W1 = NF.set_external(w1, GC1)  
w2 = np.zeros(width); w2[position2-s2/2.0:position2+s2/2.0] = 1.0
W2 = NF.set_external(w2, GC2)

list_obj = [GC1, GC2, NF] ## they need to be updated at each time step


## -- Plotting and Loop section:

# I plot the connection function and the gain function:
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
plt.show()


## I define which probes we want to plot:
displayed_probes = ["Default", "P1", "P2"]

## First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize = (7,11))
print "Hello"

ax1 = fig.add_subplot(411); ax1.set_xlim(0, 500); ax1.set_ylim(-1000, 1000)
ax1.set_title("Recorded Probes")
ax1.set_xlabel("time"); ax1.set_ylabel("Membrane Potential")
list_lines = []
for o in list_obj:
    for probe in o.getProbes():
        list_lines.append(*ax1.plot([], [], color = probe.color, linestyle = o.lns, lw=2))

ax2 = fig.add_subplot(412); ax2.set_xlim(0, width); ax2.set_ylim(-1000, 1000)
ax2.set_title("Membrane Potential over the Map (1D)")
ax2.set_xlabel("neuron positions"); ax2.set_ylabel("membrane potential")
NF_mp, = ax2.plot([], [], color = "red", linestyle = "-", lw=2)
ax2.hlines(0, 0, 2000, color = "gray", linestyle="--")
for probe in NF.getProbes():
   ax2.scatter(probe.position, 0, s=10, color=probe.color)
list_lines.append(NF_mp)

ax3 = fig.add_subplot(413); ax3.set_xlim(0, width); ax3.set_ylim(0, 600)
ax3.set_title("Firing Rate over the Map (1D)")
ax3.set_xlabel("neuron positions"); ax3.set_ylabel("Firing Rate")
NF_firing, = ax3.plot([], [], color = "green", linestyle = "-", lw=2)
list_lines.append(NF_firing)

plt.tight_layout()


## Now the graph is ready, I run the sim and plot the results in real time:
def init():
    global list_lines
    line_index = 0
    for o in list_obj:
        for probe in o.getProbes():              
            if probe.name in displayed_probes:
                list_lines[line_index].set_data([], [])   
            line_index+=1
    print "Initialisation"
    return list_lines
    
## animation function.  This is called sequentially
def animate(i):
    #print "Animate"
    line_index = 0
    global list_obj, list_lines, ax1 ## list_obj is all the objects which need to be updated at each time step
    for o in list_obj:
        o.update()
        for probe in o.getProbes():
            #print probe.name, "at", probe.position
            if probe.name in displayed_probes:
                list_lines[line_index].set_data(probe.time_buffer, probe.value_buffer)
            line_index+=1
        xmin, xmax = ax1.get_xlim()
        if o.t > xmax:
            ax1.set_xlim(xmin, 2*xmax)
            ax1.figure.canvas.draw()
    NF_mp.set_data(xrange(0,width), NF.V)
    NF_firing.set_data(xrange(0,width), NF.GainFunction(NF.V))
    if RECORD_VIDEO:
        return mplfig_to_npimage(fig) 
    else:
        return list_lines
   
from time import time
t0 = time() 
print list_obj

## call the animator.  blit=True means only re-draw the parts that have changed.
if not RECORD_VIDEO:
    anim = animation.FuncAnimation(fig, animate, frames=simulation_time,  interval=5 , blit=True, init_func= init, repeat=False)
else:
    from moviepy.editor import VideoClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    anim = VideoClip(animate, duration = simulation_time/60)
    anim.write_videofile("test2.mp4", fps = 60)

t1 = time()
print((t1 - t0))
plt.show()