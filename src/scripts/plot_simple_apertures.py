from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths


# import matplotlib as mpl
# mpl.use('MacOSX')

# We make the grids on which the simulation is defined.
pupil_grid = make_pupil_grid(128)

# We want to sample from -16 lam/D to 16 lam/D with 5 pixels per lam/D.
focal_grid = make_focal_grid(5, 16)

prop = FraunhoferPropagator(pupil_grid, focal_grid)

def pupil_focal_planes(obscuration=0, num_spiders=4, spider_width=0, wavelength=1):
    ''' A function that generates a pupil and its corresponding PSF.

    Parameters
    ------------
      obscuration - float
        The obscuration ratio of the telescope aperture in percentage.
      num_spiders - int
        The number of spiders in the aperture.
      spider_width - float
        The width of the spiders in physical units (e.g. meters).
    '''
    ap = make_obstructed_circular_aperture(1, obscuration, num_spiders, spider_width)
    pup = evaluate_supersampled(ap, pupil_grid, 4)

    wf = Wavefront(pup, wavelength)

    psf = prop(wf).power
    psf /= psf.max()
    return pup,psf




fig, axes = plt.subplots(2,4,figsize=(8,4))


ax = np.ndarray.flatten(axes)

pup,psf = pupil_focal_planes(0)
imshow_field(pup, cmap='gray',ax=ax[0])
imshow_field(np.log10(psf), cmap='inferno', vmax=0, vmin=-6,ax=ax[4])

ax[0].set_xlabel('')
ax[0].set_ylabel('y / D')
ax[4].set_xlabel('$\lambda/D$')
ax[4].set_ylabel('$\lambda/D$')

pup,psf = pupil_focal_planes(0.20)
imshow_field(pup, cmap='gray',ax=ax[1])
imshow_field(np.log10(psf), cmap='inferno', vmax=0, vmin=-6,ax=ax[5])

pup,psf = pupil_focal_planes(0,num_spiders=3,spider_width=0.02)
imshow_field(pup, cmap='gray',ax=ax[2])
imshow_field(np.log10(psf), cmap='inferno', vmax=0, vmin=-6,ax=ax[6])

pup,psf = pupil_focal_planes(0.20,num_spiders=4,spider_width=0.04)
imshow_field(pup, cmap='gray',ax=ax[3])
imshow_field(np.log10(psf), cmap='inferno', vmax=0, vmin=-6,ax=ax[7])

for i in (ax[0:4]):
	i.set_facecolor('black')
	i.set_xlim(-0.6,0.6)
	i.set_ylim(-0.6,0.6)

#ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])
ax[5].set_yticks([])
ax[6].set_yticks([])
ax[7].set_yticks([])


plt.draw()
#plt.show()
plt.savefig(paths.figures/'simple_apertures.pdf')
