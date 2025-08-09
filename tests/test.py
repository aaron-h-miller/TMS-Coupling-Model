import argparse


if __name__ == '__main__':
    import numpy as np
    from neuron import h
    import matplotlib
    # matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    axon = h.Section()
    axon.L = 3000
    axon.nseg = 3000
    axon.insert('hh')
    axon.insert('pas')

    VecAxon = h.Vector()
    VecAxon.record(axon(0.1)._ref_v)

    time = h.Vector()
    time.record(h._ref_t)

    VecT = h.Vector([0, 1000])
    VecStim = h.Vector([50, 0])

    stim = h.IClamp(axon(0))
    stim.dur = 1e9
    # stim.amp = 1e9

    VecStim.play(stim._ref_amp, VecT, 1)
    # h.VecStim.play(stim.amp, h.VecT, 1)
    # h.VecStim.play(stim, VecT, 1)

    h.finitialize(-68)
    h.run(10)

    plt.figure()
    plt.plot(np.array(time), np.array(VecAxon))

    # ###################################### Test argparse #############################################################
    # parser = argparse.ArgumentParser(description='Test input arguments')
    # parser.add_argument('-pc', '--proc_id', type=int, default=None, required=False)
    #
    # args = parser.parse_args()
    # proc_id = args.proc_id
    #
    # print(args)
    # print(proc_id)
    #
    # if proc_id is None:
    #     raise ValueError('Proc id was none')

    # #################################### Test kde scaling ############################################################
    #
    # from axonal_delay_kernel import get_axonal_delay_kernel
    # import matplotlib
    # # matplotlib.use('Qt5Agg')
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import scipy.stats as stats
    #
    # kernel = get_axonal_delay_kernel(
    #     source_layer_name='L23',
    #     theta=30,
    #     gradient=0,
    #     intensity=225,
    #     density=False
    # )
    # t_bin_center, z_bin_center, histogram = kernel.t_bin_centers, kernel.z_bin_layer_centers, kernel.layer_histogram
    #
    # fig = plt.figure()
    # cplot = plt.pcolor(t_bin_center, z_bin_center, histogram, cmap='RdBu_r')
    # plt.colorbar(cplot, label='spike density ($\\mu m^-1$ $ms^-1$)' if kernel.density else 'spike count')

# ############################################# Plotting Ellipse #######################################################
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # Create a color plot with x ranging from -100 to 100
#     x = np.linspace(-100, 100, 100)
#     y = np.linspace(0, 5, 50)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sin(X) * np.cos(Y)
#
#     fig, ax = plt.subplots()
#     cax = ax.pcolormesh(X, Y, Z, shading='auto')
#     cb = fig.colorbar(cax)
#
#     xrange = abs(np.diff(np.array(ax.get_xlim()))[0])
#     yrange = abs(np.diff(np.array(ax.get_ylim()))[0])
#
#     # Define the start position and arrow length in data coordinates
#     start_x = 0  # Starting at x = 0
#     start_y = 2  # Arbitrary y value
#
#     # Arrow length
#     arrow_length_y = 2  # Length in y-axis units
#     arrow_length_x = (arrow_length_y / np.tan(np.deg2rad(30))) * (xrange / yrange) # Length in x-axis units considering the angle
#
#     # Plot the arrow at a 30-degree angle
#     ax.annotate(
#         '', xy=(start_x + arrow_length_x, start_y + arrow_length_y), xytext=(start_x, start_y),
#         arrowprops=dict(facecolor='black', shrink=0, width=2, headwidth=8)
#     )
#
#     # Display the plot
#     plt.show()


