from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

Dtel = 39.4
grid = make_pupil_grid(256, 42.0)

aperture_function, segment_functions = make_elt_aperture(return_segments=True)
aperture = evaluate_supersampled(aperture_function, grid, 4)

#Try to read the segments file otherwise create it.
try:
	segments = read_field(str(paths.data/'elt_segments.fits'))
except:
	segments = Field([evaluate_supersampled(seg, grid, 4) for seg in segment_functions], grid)
	write_field(segments, str(paths.data/'elt_segments.fits'))


#segments = Field([evaluate_supersampled(seg, grid, 4) for seg in segment_functions], grid)
	
num_segments = segments.shape[0]

phasing_errors = PhaseApodizer(grid.zeros())
coronagraph = PerfectCoronagraph(aperture)

focal_grid = make_focal_grid(q=5, num_airy=50, spatial_resolution=1/Dtel)
prop = FraunhoferPropagator(grid, focal_grid)

wf = Wavefront(aperture)
wf.total_power = 1

norm = prop(wf).power.max()

abb = SurfaceAberration(grid, 0.1, Dtel)
ff = FourierFilter(grid, make_circular_aperture(2 * np.pi * 60 / Dtel))
abb.surface_sag = abb.surface_sag - np.real(ff.forward(abb.surface_sag + 0j)) 

rms = np.logspace(-1.5, -0.5, 3)

fig, axes =plt.subplots(1,3,figsize=(12,4))

ax = np.ndarray.flatten(axes)
#for i, rms_i in enumerate(rms):

rms_i = 0.32

phasing_errors.phase = segments.T.dot( rms_i * np.random.randn(num_segments) )

corim = prop( coronagraph( phasing_errors(abb(wf) ) ) ).power / norm
psf = prop( phasing_errors(abb(wf) ) ).power / norm

imshow_field(phasing_errors.phase, vmin=-np.pi/2, vmax=np.pi/2,cmap='bwr',ax=ax[0])
ax[0].set_title(f'Phase error r.m.s.={rms_i:4.2f} rad')
plt.colorbar()
imshow_psf(psf, vmax=1, vmin=1e-5,ax=ax[1])
ax[1].set_title('ELT PSF')
ax[2].set_title('Coronagraphic PSF')
imshow_psf(corim, vmax=1e-2, vmin=1e-7,ax=ax[2])

plt.tight_layout()
plt.draw()
plt.savefig(paths.figures/'phasing_errors.pdf')

#plt.show()