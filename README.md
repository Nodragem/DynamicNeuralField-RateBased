# DynamicNeuralField-RateBased

A Small Library to quickly run and test 1D Neural Fields with a population rate model.

## Example Files:
- **NeuralMap-TwoStimCompetition.py:** note that you can see an output example with `test-TwoStim.mp4`.
- **NeuralMap-SizeEffect.py:** note that you can see some output examples with `test-SizeEffect*.mp4`.
- **NeuralMap-GainEffect.py:** note that `Display-GainEffect.py` needs to be run after it to display the results. You may need to create some folders.

Note that the two first examples run only one simulation at a time with a real-time display and can record a video. You need to change the parameters by hand and re-run a simulation to explore their effects. A better implementation would separate display from simulation, which would allows to run several simulations at once and re-use the same display code for all of them. I may improve that later.

Please refer to the third example as a more efficient way to run several simulations which separated data creation (simulation) from data display.

## Library content:

All the useful function are in `./util/Generators.py`.
Generators are objects which generate a value V for each timestep of a simulation thanks to a predefined function f. The V value can be seen as the membrane potential of a neuron - when the generator simulates a neuron or a neural field.

Once you created and set up Generators you can run them as simply as:
``` python
for t in xrange(simulation_time):
    for g in list_generators:
        g.update()
```
More details in the examples are provided to save the result of the simulations.

Here a list of the currently available Generators:

#### NeuralField1D:
*Note that the library contains also some useful comments in the code source.*

It allows to generate an array of membrane potential V thanks to a function f which combines a **Gain Function** and a **Connection Kernel**. Less abstractly, a neural field simulates the firing rate of an array of neurons homogeneously connected together and receiving external input stimulations.

Homogeneous connections means that each neuron in the field is connected to its neighborhood in a similar way. This common connectivity pattern is the **Connection Kernel**, and it usually function of space. See examples to see how to define it.

Neurons influence each other through their connections; however they don't influence each other directly with their membrane potential V. A neuron influence an other neuron only through its spikes (or action potential). Here, the relevant information is assumed to be in the rate of spikes -- the firing rate: the function used to estimate the firing rate from the membrane potential V is defined by the **GainFunction**. See examples to see how to re-define it.

Note that external inputs does not pass through the GainFunction and can be set with `set_external()`.

#### Accumulator:
It allows to simulate a single neural unit (which may represent on or several identical neurons). It gives you back its membrane potential V at each timestep of the simulation according to a function f which takes in account external input, connection inputs and a **GainFunction**.

Note that external inputs DO NOT pass through the **GainFunction** to influence the Accumulator.
Note that connection inputs DO pass through the **GainFunction** to influence the Accumulator.

Connection input can be set with `set_connection()` and it is used to connect two Accumulators together.
External input can be set with `set_external()` and it is used to connect an arbitrary input to an Accumulator.

#### GeneratorEPSP:
Return a V value at each timestep which draws an EPSP curve over time: instantaneous peak followed by a decay of activity. Used as external input.

#### GeneratorGaussienne:
Return a V value at each timestep which draws a Gaussian curve over time. Used as external input.

#### GeneratorConstante:
Return a V value at each timestep which is Constant. Used as external input.
