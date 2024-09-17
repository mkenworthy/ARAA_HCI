from hcipy import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
	
	grid = make_pupil_grid(512, 1.1)
	aperture = evaluate_supersampled(make_circular_aperture(1), grid, 4)

	focal_grid = make_focal_grid(q=15, num_airy=15)
	prop = FraunhoferPropagator(grid, focal_grid)

	# Make the wavefront
	wf = Wavefront(aperture)
	wf.total_power = 1
	norm = prop(wf).power.max()

	zmodes = make_zernike_basis(10, 1, grid)
	text = [
		'rad rms astigmatism'# \n no defocus',
		'rad rms astigmatism'# \n no defocus',
		'rad rms astigmatism'# \n no defocus',
		'rad rms astigmatism'# \n with defocus',
		'rad rms astigmatism'# \n with defocus',
		'rad rms astigmatism'# \n with defocus',
	]
	options = ['No', 'With']
	plt.figure(figsize=(10,6))
	for di, focus_amp in enumerate([0, 1]):
		for ai, amp in enumerate([-1, 0, 1]):
			k = 3 * di + ai

			# 1 radian of focus
			focus_diversity = PhaseApodizer(focus_amp * zmodes[3])
			astigmatism = PhaseApodizer(amp * zmodes[4])
			wf_foc = prop(focus_diversity(astigmatism(wf)))
			
			plt.subplot(2, 3, 1 + ai + 3 * di)
			imshow_psf(wf_foc.power / norm, vmax=1, vmin=1e-5, colorbar=False)
			#plt.text(0, 10, '{:d} '.format(amp) + text[k], color='white', horizontalalignment = 'center', fontsize=14)
			if di == 0:
				plt.title('{:d} rad rms astigmatism'.format(amp), fontsize=16)
			if ai == 0:
				plt.ylabel(options[di] + ' defocus', fontsize=16)

			#plt.gca().get_xaxis().set_visible(False)
			#plt.gca().get_yaxis().set_visible(False)
			plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
			plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)

	plt.tight_layout()
	
	plt.savefig('./Plots/phase_diversity_example.pdf')
	plt.show()
