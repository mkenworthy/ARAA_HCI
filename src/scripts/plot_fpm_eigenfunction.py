from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

def find_eigenmode(aperture, mask_diameter, tolerance=1e-8):
	grid = aperture.grid

	focal_grid = make_focal_grid(q=11, num_airy=1.1 * mask_diameter / 2)
	fpm = Apodizer(evaluate_supersampled(make_circular_aperture(mask_diameter), focal_grid, 4))

	prop = FraunhoferPropagator(grid, focal_grid)

	met_tolerance = False
	
	# Normalize to unit power
	amplitude = aperture / np.sqrt(np.sum(aperture**2) * grid.weights)

	while not met_tolerance:
		# propagate the apodized through the mask
		wf = Wavefront(amplitude)
		wf_out = prop.backward(fpm(prop(wf)))
		
		# Find the new amplitude and normalize it to unit power
		new_amplitude = wf_out.amplitude * aperture
		new_amplitude = new_amplitude / np.sqrt(np.sum(new_amplitude**2) * grid.weights)

		if np.abs(new_amplitude - amplitude).max() / amplitude.max() < tolerance:
			met_tolerance = True

		amplitude = new_amplitude

	return amplitude

Dtel = 1
grid = make_pupil_grid(512, 1.1)

plt.figure(figsize=(12,3))

aperture_function = make_gmt_aperture(normalized=True)
aperture = evaluate_supersampled(aperture_function, grid, 4)
weak_apodized_aperture = find_eigenmode(aperture, 1)
strong_apodized_aperture = find_eigenmode(aperture, 2)

plt.subplot(1, 3, 1)
imshow_field(aperture / aperture.max(), vmin=0, vmax=1, cmap='gray')
plt.colorbar()
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title('No apodization', fontsize=18)

plt.subplot(1, 3, 2)
imshow_field(weak_apodized_aperture / weak_apodized_aperture.max(), vmin=0, vmax=1, cmap='gray')
plt.colorbar()
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title('FPM diameter = 1 $\lambda / D$', fontsize=18)

plt.subplot(1, 3, 3)
imshow_field(strong_apodized_aperture / strong_apodized_aperture.max(), vmin=0, vmax=1, cmap='gray')
plt.colorbar()
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
plt.title('FPM diameter = 2 $\lambda / D$', fontsize=18)

plt.tight_layout()
plt.savefig(paths.figures / 'fpm_eigenfunction.pdf')
