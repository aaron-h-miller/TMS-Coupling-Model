COMMENT
Author: Aaron Miller
Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig Germany
--
Point process which computes membrane current (nonspecific current) as a result of a user provided conductance
timeseries for a voltage gated NMDA synapse. Based on a conductance-based synapse, the user specifies the refersal
potential 'e', the weight of the synapse 'weight' and plays a vector of conductance values in uS into 'g'.
The current is computed as: i = g * weight * (v - e)
Voltage gated NMDA synapse include a scaling factor to the conductance namely 1 / (1 + 0.28 * Mg * exp(-0.062 * v))
where Mg is the concentration of extracellular magnesium, set here to 1 mmol
(Jahr, C. E., & Stevens, C. F. (1990). Voltage dependence of NMDA-activated macroscopic conductances predicted by
single-channel kinetics. The Journal of neuroscience : the official journal of the Society for Neuroscience,
10(9), 3178–3182. https://doi.org/10.1523/JNEUROSCI.10-09-03178.1990).
ENDCOMMENT

NEURON {
	POINT_PROCESS SynCurNMDA
	RANGE e, weight, i
	NONSPECIFIC_CURRENT i

	RANGE g  : (Input conductance values per time step)
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

CONSTANT {
	Mg = 1
}

PARAMETER {
	e = 0	(mV)
	weight = 1 (uS)
	g = 0 (uS)
}

ASSIGNED {
	v (mV)
	i (nA)
}

BREAKPOINT {
	i = weight*g*(v - e) / (1 + 0.28 * Mg * exp(-0.062 * v))
}
