import numpy as np
from scipy.signal import fftconvolve

class Probe:
    def __init__(self, name, position):
        self.position = position
        self.line = 0
        self.value_buffer = []
        self.time_buffer = []
        self.lns = "-"
        self.color = [0.0, 0.0, 1.0] ## blue?
        self.name = name

    def update(self, time, value):
        self.value_buffer.append(value)
        self.time_buffer.append(time)

class Generator:
    """ Generators are objects which generates a V value for each timestep thanks to function f"""
    def __init__(self):
        self.dt = 1
        self.t = 0.0
        self.end = 2000
        self.V = np.array([0])
        self.probe_list= [Probe("default", 0)]
        self.lns = "-"
        self.color = [0.0, 0.0, 1.0] ## blue?

    def setGenerator(self, frames, dt=1):
        self.dt = dt
        self.end = frames

    def f(self, t):
        return 0

    def addProbe(self, name, position):
        self.probe_list.append(Probe( name, position ) )
        return self.probe_list[-1]

    def getProbe(self, index):
        return self.probe_list[index]

    def getProbes(self):
        return self.probe_list

    def update(self):
        for i in xrange(self.end):
            #print "update"
            self.t += self.dt ## we update our t
            self.V[:] = self.f(self.t) ## we take the value of f() for the time t
            self.update_probes() ## if I have probes I update them (they record V and t in an array)
            return self.t, self.V

    def update_probes(self):
        for p in self.probe_list:
            p.update(self.t, self.V[p.position])


class GeneratorConstante(Generator):
    def __init__(self, constante, name="Const"):
        Generator.__init__(self)
        self.c = constante

    def f(self, t): ## mu = centre gauss en ms, compter 3sigma pour largeur, sim_time = duree de la simulation en ms
        return self.c

class NeuralField1D(Generator):
    """ A Neural Field is a Generator which generate many V values for each time step"""
    def __init__(self, number_of_neurons, tau, name="NF_Default"): # what is the unit of tau ?
        Generator.__init__(self)
        self.n = number_of_neurons
        self.V = np.zeros(self.n)
        self.input = np.zeros(self.n)
        self.tau = tau
        self.beta = 0.07
        self.firing_offset = 0
        self.u0 = 0
        self.dV = 0
        self.noise = lambda x: 0
        self.myConnection = 0
        self.external = [] ## they don't pass by the function A(u)

    def f(self,t): ## estimate the neural activity at time t (called at every time step)
        self.dV = np.zeros(self.n, dtype=np.float64)
        if len(self.external) > 0: ## check if my neural field has external inputs
            for e in self.external: ## add the external input
                ## Note about the external input:
                ## - they are given without passing in the function self.A(), i.e. they are "current input" in membrane potential
                ## - they are Generators, they update themselves their value over time.
                self.dV += e[0] * e[1].V ## the attribute V is updated by the external input
        if isinstance(self.myConnection, np.ndarray): ## check if myConnection is an array
            activity_map = self.GainFunction(self.V) # get the estimate of the firing rate over the map
            ## Note about the np.convolve function:
            ## - np.convolve in full mode return an array of lenght M-N+1 if M is the largest
            ## - self.myConnection is an 2*N-1 array
            ## - lenght(self.input) = lengt(self.V) = self.n = N
            ## OK, so:
            # 1) fftconvolve or np.convolve give the same results
            # 2) the mode does not change the computation method/results,
            #    it changes just the window you take from the results.
            # 3) the order of the matrix do not change anything (see the html-example I did).
            convolution = fftconvolve(self.myConnection, activity_map,  mode="valid") # better
            self.dV += convolution
        self.dV += self.noise(self.n)
        self.dV -= self.u0
        self.dV -= self.V
        self.dV = self.dV/self.tau
        new_V = self.V + self.dt*self.dV
        #self.V[self.V>100] = 100. ## better to have a upper limit
        #self.V[self.V<-50] = -50. ## better to have a downer limit
        return new_V

    def GainFunction(self, u): ## transform the activity in firing rate
        #print u
        #firing_rate = 500.0/(1.0+ np.exp(-0.07*(u-250.0))) ## better than Trappenberg 2001.
        firing_rate = 500.0/(1.0+np.exp(-self.beta*(u-self.firing_offset))) ## like Trappenberg 2001
        #print firing_rate
        #firing_rate[firing_rate < 0] = 0 ## very important !! (we forgot it) .. hmm.. normally it should never be the case!
        return firing_rate

    def createConnectionCurve(self, g):
        """Creates a connection curve based on a given function g """
        x = np.arange(-self.n+1, self.n, 1) ## better, use the "valid" option of the convolution in the update function
        # x = np.arange(-self.n/2, self.n/2, 1) ## closer to Trappenberg, please used "same" in the convolution function in self.update
        self.myConnection = g(x)
        return self.myConnection

    #def changeCurrentToFiringFunction(self, g)

    def set_noise(self, noise):
        self.noise = noise;

    def set_external(self, weight, input):
        """ sets up an external input"""
        self.external.append([weight, input])
        return self.external[-1]

class GeneratorEPSP(Generator):
    def __init__(self, name, start, decay, amp=1.0):
        Generator.__init__(self)
        self.amp = amp
        self.decay = decay
        self.start = start

    def f(self, t):
        if t < self.start: return 0
        if t >= self.start: return self.amp*np.exp(-self.decay*(t-self.start)) #+ A*exp((-(x-mu-100)**2)/(sigma**2))


class GeneratorGaussienne(Generator):
    def __init__(self, name, freq_max, mu, sigma):
        Generator.__init__(self)
        self.A = freq_max
        self.mu = mu
        self.s = sigma

    def f(self, t):
    ## mu = center; s= standard deviation; A= max (of the Gaussian)
        return self.A*np.exp((-(t-self.mu)**2)/(self.s**2)) #+ A*exp((-(x-mu-100)**2)/(sigma**2))

class Accumulator(Generator):
    """ It is like a neuron """
    def __init__(self, name, tau, connection=[], external=[]):
        Generator.__init__(self)
        self.Tr = 50
        self.tau = tau
        self.input = 0
        self.connection = connection ## they pass by the GainFunction(u)
        self.external = external ## they don't pass by the GainFunction A(u)
        self.reached = False

    def GainFunction(self, u):
        return ( 1/(1+ np.exp(-0.07*u)) )

    def f(self, t):
        self.input = 0
        if len(self.connection) > 0:
            for c in self.connection:
                self.input +=  c[0] * self.GainFunction(c[1].V)
        if len(self.external) > 0:
            for e in self.external:
                self.input += e[0] * e[1].V
        self.V += self.dt*(self.input-self.V)/self.tau +0.01*np.random.randn(1)[0]
        if self.V > self.Tr:
            self.V = self.Tr
            self.reached = True
        return self.V

    def set_connection(self, weight, input):
        self.connection.append([weight, input])

    def set_external(self, weight, input):
        self.external.append([weight, input])
