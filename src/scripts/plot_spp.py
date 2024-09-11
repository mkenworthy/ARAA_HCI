from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
	
	grid = make_pupil_grid(256, 1.1)
	aperture_func = make_circular_aperture(1)
	aperture = evaluate_supersampled(aperture_func, grid, 4)
	aperture_mask = aperture > 0

	##
	iwa = 2.0
	owa = 10.
	target_contrast = 1e-5

	focal_grid = make_focal_grid(q=5, num_airy=owa+5)
	prop = FraunhoferPropagator(grid, focal_grid)

	# Make the wavefront
	wf = Wavefront(aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()
	
	spp_amp = read_field('./Data/spp_amplitude_example.fits')
	spp = Apodizer(spp_amp)

	wf_foc = prop(spp(wf))
	corim = wf_foc.power / norm

	plt.figure(figsize=(14,8))

	plt.subplot(2,4,1)
	plt.title('No apodizer')
	imshow_field(grid.ones(), grid, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2,4,2)
	plt.title('No focal plane mask')
	imshow_field(focal_grid.ones(), focal_grid, cmap='gray', vmin=0, vmax=1)

	plt.subplot(2,4,5)
	plt.title('PSF before focal plane mask')
	imshow_psf(prop(wf).power / norm, vmax=1, vmin=target_contrast, colorbar=False)
	plt.subplot(2,4,6)
	plt.title('PSF after focal plane mask')
	imshow_psf(prop(wf).power / norm, vmax=1, vmin=target_contrast, colorbar=False)

	plt.subplot(2,4,3)
	plt.title('SPP amplitude pattern')
	imshow_field(spp.apodization, grid, vmin=0, vmax=1, cmap='gray')
	plt.subplot(2,4,4)
	plt.title('Post-coronagraphic stellar PSF')
	imshow_psf(corim, vmax=1, vmin=target_contrast, colorbar=False)

	plt.subplot(2,4,7)
	plt.title('Star light before Lyot stop')
	imshow_field(wf.power / wf.power.max(), vmin=0, vmax=1, cmap='inferno')
	plt.subplot(2,4,8)
	plt.title('Star light after Lyot stop')
	imshow_field(wf.power / wf.power.max(), vmin=0, vmax=1, cmap='inferno')
	
	plt.tight_layout()
	plt.show()