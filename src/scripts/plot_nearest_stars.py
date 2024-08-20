import numpy as np
from matplotlib import pyplot as plt
import paths
from astropy.io import ascii
import astropy

#import matplotlib as mpl
#mpl.use('MacOSX')


t = ascii.read(str(paths.data/'J_ApJS_271_55/table4.dat'), # weird str() used because PosixPaths() error.. bug in ascii.read
	readme=str(paths.data/'J_ApJS_271_55/ReadMe'),  
	format='cds')

#print(t['OName'],t['Ncomp'],t['Plx'],t['Gmag'],t['Hmag'])
print(len(t))
t = t[(~t['Plx'].mask) + (~t['Gmag'].mask)]

ts = t[(t['Plx']>100.) * (t['Gmag']<15) * (t['NcTR']==1)]

print(len(ts))

fig, (ax) = plt.subplots(1,1,figsize=(8,6))

#ax.scatter(ts['Plx'],ts['Gmag'])


ts=ts[~ts['Gmag'].mask]

teff = ts['Teff']
print(teff)
d = 1000./ts['Plx'] # distance in pc

for r in ts:
	if r['Name']:
		ax.text(1000./r['Plx'],r['Gmag'],r['Name'])
ax.scatter(d,ts['Gmag'])

for r in ts:
	if r['SpTOpt']:
		ax.text(1000./r['Plx'],r['Gmag']+0.6,r['SpTOpt'],color='red')

#ax.text(d,ts['Gmag'],ts['Name'])
ax.set_xlim(0,10)
ax.set_ylim(15,2)
ax.set_xlabel('Distance [pc]')
ax.set_ylabel('Magnitude [G]')

plt.draw()
#plt.show()

plt.savefig(paths.figures / 'nearest_stars.pdf')
