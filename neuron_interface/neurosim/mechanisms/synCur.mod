COMMENT
Author: Aaron Miller
Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig Germany
--
Point process which computes membrane current (nonspecific current) as a result of a user provided conductance
timeseries.
Based on a conductance-based synapse, the user specifies the refersal potential 'e', the weight of the synapse 'weight'
and plays a vector of conductance values in uS into 'g'. The current is computed as:
i = g * weight * (v - e)
ENDCOMMENT

NEURON {
	POINT_PROCESS SynCur
	RANGE e, weight, i
	NONSPECIFIC_CURRENT i

	RANGE g  : (Input conductance values per time step)
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
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
	i = weight*g*(v - e)
}
