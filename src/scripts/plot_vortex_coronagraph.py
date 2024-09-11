from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
	
	grid = make_pupil_grid(512, 1.1)
	aperture_func = make_circular_aperture(1)
	aperture = evaluate_supersampled(aperture_func, grid, 4)

	lyot_mask = evaluate_supersampled(make_circular_aperture(0.95), grid, 4)
	lyot_stop = Apodizer(lyot_mask)

	focal_grid = make_focal_grid(q=5, num_airy=15)
	prop = FraunhoferPropagator(grid, focal_grid)

	# Make the wavefront
	wf = Wavefront(aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()
	
	pre_apodizer = Apodizer(1.0 * (aperture>0))

	# Make the focal plane mask
	charge = 4
	lyot_coronagraph = VortexCoronagraph(grid, charge)
	target_contrast = 1e-8

	# The intermediate focal grid
	vortex_mask = Field(np.angle(np.exp(1j * charge * focal_grid.as_('polar').theta)), focal_grid)
	focal_plane_mask_low_res = PhaseApodizer(vortex_mask)

	#
	wf_apodized = pre_apodizer(wf)
	
	# First focal plane wavefronts
	wf_pre_fpm = prop(wf_apodized)
	wf_post_fpm = focal_plane_mask_low_res(wf_pre_fpm)

	# Lyot Plane wavefronts
	wf_lyot = lyot_coronagraph(wf_apodized)
	wf_post_lyot = lyot_stop(wf_lyot)

	# Science focal plane wavefronts
	wf_foc = prop(wf_post_lyot)

	corim = wf_foc.power / norm

	plt.figure(figsize=(14,8))

	plt.subplot(2,4,1)
	plt.title('Apodizer')
	imshow_field(grid.ones() * (aperture>0), grid, cmap='gray', vmin=0, vmax=1)
	plt.subplot(2,4,2)
	plt.title('Focal plane mask')
	imshow_field(focal_plane_mask_low_res.phase, focal_grid, cmap='twilight', vmin=-np.pi, vmax=np.pi)

	plt.subplot(2,4,5)
	plt.title('PSF before focal plane mask')
	imshow_psf(wf_pre_fpm.power / norm, vmax=1, vmin=target_contrast, colorbar=False)
	plt.subplot(2,4,6)
	plt.title('PSF after focal plane mask')
	imshow_psf(wf_post_fpm.power / norm + 1e-15, vmax=1, vmin=target_contrast, colorbar=False)

	plt.subplot(2,4,3)
	plt.title('Lyot Stop')
	imshow_field(lyot_stop.apodization, grid, vmin=0, vmax=1, cmap='gray')
	plt.subplot(2,4,4)
	plt.title('Post-coronagraphic stellar PSF')
	imshow_psf(corim, vmax=1, vmin=target_contrast, colorbar=False)

	plt.subplot(2,4,7)
	plt.title('Star light before Lyot stop')
	imshow_field(wf_lyot.power / wf_lyot.power.max(), vmin=0, vmax=1, cmap='inferno')
	plt.subplot(2,4,8)
	plt.title('Star light after Lyot stop')
	imshow_field(wf_post_lyot.power / wf_lyot.power.max(), vmin=0, vmax=1, cmap='inferno')
	
	plt.tight_layout()
	plt.show()