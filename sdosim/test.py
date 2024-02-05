import numpy as np
import matplotlib.pyplot as plt


import matplotlib.collections as mcoll
import matplotlib.path as mpath

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def daisy(r0, tau, phi1, phi2, nosc):

    dt = 0.1
    nSamples = np.round((nosc*tau)/dt)
    t = np.arange(0,nSamples)*dt

    y0 = 0#-11.53
    x0 = 0#192.5

    dx = r0 * np.sin(2*np.pi*t/tau + phi1) * np.cos(2*t/tau + phi2)/np.cos(y0)
    dy = r0 * np.sin(2*np.pi*t/tau + phi1) * np.sin(2*t/tau + phi2)

    x = x0+dx
    y = y0+dy

    print(np.shape(x))

    return x, y



nosc              = 44*8#22*2 # 22 radial oscillations for closed Daisy pattern
map_radius        = 1.0  # arc-minutes
radial_osc_period = 120 # seconds
n_scans           = 1    # split 22*10 oscillations over 5 scans
scanDuration      = nosc*radial_osc_period
phi2              = 2.0*nosc / n_scans
phi1              = 3.14159265*phi2






x, y = daisy(r0=map_radius, tau=radial_osc_period, phi1=phi1, phi2=phi2, nosc=nosc/2)

print(np.shape(x))

z = np.linspace(0, 1, len(x))
#plt.plot(x,y)
colorline(x, y, z, cmap=plt.get_cmap('inferno_r'), linewidth=1)


plt.axis('square')
plt.xlabel(r'$\Delta x$ (deg)')
plt.ylabel(r'$\Delta y$ (deg)')
plt.show()




asdasdasd


for i in np.arange(n_scans):
    phi1_to_use = i*phi1
    phi2_to_use = i*phi2

    x, y = daisy(r0=map_radius, tau=radial_osc_period, phi1=phi1_to_use, phi2=phi2_to_use, nosc=nosc/2)

    print(np.shape(x))

    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('inferno_r'), linewidth=1)

    #plt.plot(x,y,color=f'C{i}')#,alpha=0.2)
    plt.figure(1,figsize=(2,2))
    plt.axis('square')
    print(phi1_to_use, phi2_to_use)
    plt.xlabel(r'$\Delta x$ (deg)')
    plt.ylabel(r'$\Delta y$ (deg)')
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.savefig('daisy.pdf')
    plt.savefig('daisy.png')

plt.show()
