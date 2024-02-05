def generate_scans(self, x0, y0, r0, tau, phi1, phi2, nSamples, dt, mjd0,longitude, latitude):

    t = np.arange(nSamples)*dt

    dx = r0 * np.sin(2*np.pi*t/tau + phi1) * np.cos(2*t/tau + phi2)/np.cos(y0)
    dy = r0 * np.sin(2*np.pi*t/tau + phi1) * np.sin(2*t/tau + phi2)

    x = x0 + dx
    y = y0 + dy
    mjd = mjd0 + t/86400.
    az,el = Coordinates.h2e(x, y, mjd, longitude, latitude)

    return az,el,mjd,x,y
