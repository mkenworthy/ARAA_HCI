from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import paths

mpl.rcParams['figure.dpi'] = 100

pupil_grid = make_pupil_grid(256, 1.5)
focal_grid = make_focal_grid(8, 12)
prop = FraunhoferPropagator(pupil_grid, focal_grid)

aperture = evaluate_supersampled(make_circular_aperture(1),
                                 pupil_grid, 4)

wf = Wavefront(aperture)
img_ref = prop(wf).intensity

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))

imshow_field(np.log10(img_ref / img_ref.max()), 
             ax=ax1,
             vmin=-5, 
             cmap='inferno')

imshow_field(np.log10(img_ref / img_ref.max()), 
             ax=ax2,
             vmin=-5, 
             cmap='viridis')

fig.suptitle("PSF of an aperture")
ax1.set_xlabel("X [$\lambda/D$]")
ax1.set_ylabel("Y [$\lambda/D$]")
ax2.set_xlabel("X [$\lambda/D$]")
# ax2.set_ylabel("Y [$\lambda/D$]")
plt.draw()
plt.savefig(paths.figures / 'simple_psf.pdf')
