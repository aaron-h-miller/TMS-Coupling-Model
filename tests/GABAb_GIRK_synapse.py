import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Kd = 100  # muM^4
    K1 = 0.5  # mM^-1 ms^-1
    K2 = 0.0012  # ms^-1
    K3 = 0.18  # ms^-1
    K4 = 0.034  # ms^-1

    # Square Pulse forcing function for Transmitter concentration
    def force(t, args):
        return np.select(condlist=[t <= 5], choicelist=[0.5], default=0.0)

    # Vector field for 2 coupled ODEs describing
    # R: activated receptor fraction
    # G: micromolar concentration of activated G protein
    def vector_field(t, y, args):
        r, g = y
        k1, k2, k3, k4 = args
        d_r = k1 * force(t, args) * (1 - r) - k2 * r
        d_g = k3 * r - k4 * g
        d_y = np.array([d_r, d_g])
        return d_y

    t_start = 0
    t_stop = 150
    dt0 = 0.025
    y0 = (0.7, 1)
    field_args = (K1, K2, K3, K4)
    time = np.arange(t_start, t_stop + dt0, dt0)
    y_euler = np.zeros(shape=(len(time), 2))

    for t_index in range(1, len(time)):
        dy = vector_field(time[t_index], y0, field_args)
        y_euler[t_index, :] = y_euler[t_index - 1, :] + dy * (time[t_index] - time[t_index - 1])

    G = y_euler[:, 1]
    R = y_euler[:, 0]

    conductance = G**4 / (G**4 + Kd)

    fig, ax = plt.subplots(3, 1, sharex='all')
    ax[0].plot(time, R)
    ax[0].set_title('Fraction Activated Receptor')
    ax[0].set_ylabel('Fraction')
    ax[1].plot(time, G)
    ax[1].set_title('Concentration Activated G Protein')
    ax[1].set_ylabel('Concentration (micromolar)')
    ax[2].plot(time, conductance)
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('uS')
    ax[2].set_title('GABAb Conductance')
    fig.tight_layout()
