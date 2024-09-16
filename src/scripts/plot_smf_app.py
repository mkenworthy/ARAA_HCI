from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":	
	# Set up the grids
	grid = make_pupil_grid(256, 1.1)
	aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)

	focal_grid = make_focal_grid(q=15, num_airy=6)
	prop = FraunhoferPropagator(grid, focal_grid)
	
	# Setup the coronagraph
	app_phase = read_field('smf_app_phase_example.fits')
	app = PhaseApodizer(app_phase)

	separation = 2.5
	mfd = 1.4
	stellar_smf = SingleModeFiberInjection(focal_grid, make_gaussian_fiber_mode(mfd), position=np.array([separation, 0.0]))
	photometric_aperture = evaluate_supersampled(make_circular_aperture(mfd, center=np.array([separation, 0.0])), focal_grid, 4)

	num_waves = 151
	bandwidth = 0.2
	wavelengths = 1 + np.linspace(-bandwidth/2, bandwidth/2, num_waves)

	# Make the wavefront
	norm = 0
	for wave in wavelengths:
		wf = Wavefront(aperture, wave)
		wf.total_power = 1
		norm += prop(wf).power.max()
	
	# Simulate the optical systems
	corim = 0
	eta_star = 0
	eta_wave = []
	photo_star = []
	for wave in wavelengths:
		wf = Wavefront(aperture, wave)
		wf.total_power = 1
	
		wf_app = app(wf)
		wf_foc = prop(wf_app)
		wf_smf = stellar_smf.backward(stellar_smf(wf_foc))
		
		mono_im = wf_foc.power / norm
		corim += mono_im
		
		photo_star.append(np.sum(mono_im * photometric_aperture))
		eta_wave.append(wf_smf.total_power)

	# Plot the results

	plt.figure(figsize=(15, 3.3))
	plt.subplot(1,4,1)
	imshow_field(app.phase, grid, vmin=-np.pi, vmax=np.pi, cmap='twilight')
	
	plt.subplot(1,4,2)
	imshow_field(wf_foc.phase, vmin=-np.pi, vmax=np.pi, cmap='twilight')
	circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
	plt.gca().add_patch(circ)
	plt.xlim([separation - 2.5, separation + 2.5])
	plt.ylim([-2.5, 2.5])

	plt.subplot(1,4,3)
	imshow_psf(corim, vmax=1, vmin=1e-6, colorbar=False)
	circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
	plt.gca().add_patch(circ)
	plt.xlim([separation - 2.5, separation + 2.5])
	plt.ylim([-2.5, 2.5])
	
	plt.subplot(1,4,4)
	plt.plot(wavelengths, eta_wave, 'C0', lw=2.5, label='SMF power')
	plt.plot(wavelengths, photo_star, 'C3', lw=2.5, label='MMF power')
	plt.ylim([1e-10, 1e-3])
	plt.legend()
	plt.yscale('log')

	plt.tight_layout()
	plt.show()