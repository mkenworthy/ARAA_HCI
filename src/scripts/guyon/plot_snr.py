import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from matplotlib.colors import Normalize
fin = 'StarList_20pc.txt'

t = ascii.read(fin,format='commented_header',guess=False)

print(t)



planetlim = 30 # V band magnitude cutoff for planet

# estimated V band magnitude of the planet

planetV=t['Vmag']-2.5*np.log10(t['HZcontrast'])
fig, ax2 = plt.subplots(1,1,figsize=(8,6),)
ax2.hist(planetV,range=(25,35),bins=50)
ax2.set_xlabel('planet magnitude [apparent in V]')
ax2.set_ylabel('N')
plt.draw()


logT = np.log10(t['Teff'])

fig, ax = plt.subplots(1,1,figsize=(8,6),)

ax.set_xlim([0.004,2.0])
ax.set_ylim([1e-12,1e-6])

# # plot x and y limits
# x0: 0.07
# x1: 5
# # y0: 3E-11
# # y1: 8E-4
# ax.set_xlim([0.01,5.0])
# ax.set_ylim([3e-11,8e-4])


# G0V 5930
# G5V 5660
# G9V 5380
# K0V 5270
# K5V 4440
# K9V 3930
# M0V 3850
# M5V 3060
# M9V 

ax.set_xscale('log')
ax.set_yscale('log')
cmap = plt.colormaps['plasma']

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

colors = ["brown", "orange","yellow", "blue"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

size = 200/t['dist']

m = (planetV<faintplanet)
plt.scatter((t['HZseparcsec'])[m],(t['HZcontrast'])[m],
	marker='o', linewidth=0, s=size[m],
	c=logT[m], vmin=3.4,vmax=4.1, cmap=cmap1, alpha=0.9)


fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(3.4, 4.1), cmap=cmap1),
             ax=ax, label="log10 Temperature")


plt.scatter(t['HZseparcsec'][~m],t['HZcontrast'][~m],
	marker='o', linewidth=0, s=size[~m],
	c='gray', alpha=0.4)


plt.draw()
plt.show()
