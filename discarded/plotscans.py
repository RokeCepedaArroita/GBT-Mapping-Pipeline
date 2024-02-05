import numpy as np
import matplotlib.pyplot as plt

# Plot both as commas and as heatmap
plt.figure(1, figsize=(6,6))
plt.plot(l_c, b_c,linewidth=0.5,alpha=0.7)
plt.axis('square')
plt.ylim([central_coords[1]-rad_c, central_coords[1]+rad_c])
plt.xlim([central_coords[0]-rad_c, central_coords[0]+rad_c])
plt.xlabel('Galactic Longitude (deg)')
plt.ylabel('Galactic Latitude (deg)')
plt.title('C-Band Scans')
plt.show()

plt.figure(1, figsize=(6,6))
plt.plot(l_ku, b_ku,linewidth=0.5,alpha=0.5)
plt.axis('square')
plt.ylim([central_coords[1]-rad_ku, central_coords[1]+rad_ku])
plt.xlim([central_coords[0]-rad_ku, central_coords[0]+rad_ku])
plt.xlabel('Galactic Longitude (deg)')
plt.ylabel('Galactic Latitude (deg)')
plt.title('Ku-Band Scans')

plt.show()
