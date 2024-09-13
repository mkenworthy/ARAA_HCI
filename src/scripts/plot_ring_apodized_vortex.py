from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

if __name__ == "__main__":
	# Make the grids
	grid = make_pupil_grid(512, 1.1)
	focal_grid = make_focal_grid(q=5, num_airy=15)
	prop = FraunhoferPropagator(grid, focal_grid)

	charge = 2

	#
	clear_aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)
	clear_lyot_stop = Apodizer(evaluate_supersampled(make_circular_aperture(0.95), grid, 4))

	central_obscuration_ratio = 0.3
	obscured_aperture = evaluate_supersampled(make_obstructed_circular_aperture(1, central_obscuration_ratio), grid, 4)
	obscured_lyot_stop = Apodizer(evaluate_supersampled(make_obstructed_circular_aperture(0.95, 1.05**2 * central_obscuration_ratio), grid, 4))
	
	ring_apodizer_func, lyot_stop_func = make_ravc_masks(central_obscuration_ratio, charge=charge, pupil_diameter=1, lyot_undersize=0.05)
	ring_apodizer = Apodizer(evaluate_supersampled(ring_apodizer_func, grid, 1))
	lyot_stop = Apodizer(evaluate_supersampled(lyot_stop_func, grid, 1))

	vortex = VortexCoronagraph(grid, charge)

	# First simulate the clear aperture
	wf = Wavefront(clear_aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()
	
	wf_lyot = vortex(wf)
	wf_foc = prop(clear_lyot_stop(wf_lyot))
	clear_corim = wf_foc.power / norm

	# First simulate the obstructed aperture without modification
	wf = Wavefront(obscured_aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()
	
	wf_obs_lyot = vortex(wf)
	wf_foc = prop(obscured_lyot_stop(wf_obs_lyot))

	obstructed_corim = wf_foc.power / norm

	# Simulate the ring apodized vortex
	wf_ravc_lyot = vortex(ring_apodizer(wf))
	wf_foc = prop(lyot_stop(wf_ravc_lyot))

	ravc_corim = wf_foc.power / norm

	images = [[clear_aperture, grid.ones(), wf_lyot.power, clear_lyot_stop.apodization, clear_corim],
	[obscured_aperture, grid.ones(), wf_obs_lyot.power, obscured_lyot_stop.apodization, obstructed_corim],
	[obscured_aperture, ring_apodizer.apodization, wf_ravc_lyot.power, lyot_stop.apodization, ravc_corim]]

	for k in range(3):
		plt.subplot(3, 5, 5 * k + 1)
		imshow_field(images[k][0])

		plt.subplot(3, 5, 5 * k + 2)
		imshow_field(images[k][1], vmin=0, vmax=1, cmap='gray')
		
		plt.subplot(3, 5, 5 * k + 3)
		imshow_field(images[k][2])

		plt.subplot(3, 5, 5 * k + 4)
		imshow_field(images[k][3])

		plt.subplot(3, 5, 5 * k + 5)
		imshow_psf(images[k][4], vmax=1, vmin=1e-8)


	plt.show()