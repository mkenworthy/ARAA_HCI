from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

def accelerated_gradient_descent(gradient_function, eta, max_iterations=1e4, epsilon=1e-15, beta_start=None, callback=None, callback_frequentie=5):
	"""
	Nesterov's accelerated gradient descent

	Parameters
	----------
	model: optimization model object
	eta: learning rate
	max_iterations: maximum number of gradient iterations
	epsilon: tolerance for stopping condition
	beta_start: where to start (otherwise random)

	Output
	------
	solution: final beta value
	beta_history: beta values from each iteration
	"""

	# initialization
	if beta_start is not None:
		beta_current = beta_start
	else:
		beta_current = np.random.normal(loc=0, scale=1, size=d)

	y_current = beta_current
	t_current = 1.0

	for k in range(int(max_iterations)):
		# gradient update
		t_next = .5*(1 + np.sqrt(1 + 4*t_current**2))
		beta_next = y_current - eta * gradient_function(y_current)
		y_next = beta_next + (t_current - 1.0)/(t_next)*(beta_next - beta_current)

		# relative error stoping condition
		if np.linalg.norm(beta_next - beta_current) <= epsilon*np.linalg.norm(beta_current):
			break

		# restarting strategies
		if np.dot(y_current - beta_next, beta_next - beta_current) > 0:
			y_next = beta_next
			t_next = 1

		beta_current = beta_next
		y_current = y_next
		t_current = t_next
	
		if callback is not None and k % callback_frequentie == 0 and k > 0:
			print("iteration {:d} / {:d}".format(k, int(max_iterations)))
			callback(beta_current)

	print( 'accelerated GD finished after ' + str(k) + ' iterations' )

	return beta_current

grid = make_pupil_grid(256, 1.1)
aperture_func = make_obstructed_circular_aperture(1, 0., 4, 0.0)
aperture = evaluate_supersampled(aperture_func, grid, 4)
aperture_mask = aperture > 0

##
iwa = 2.0
owa = 10.
target_contrast = 1e-5

focal_grid = make_focal_grid(q=5, num_airy=owa+5)
prop = FraunhoferPropagator(grid, focal_grid)

# The dark hole pixels
dark_hole_mask = make_circular_aperture(2 * owa)(focal_grid) - make_circular_aperture(2 *  iwa)(focal_grid)
dark_hole_mask *= focal_grid.x > iwa

#
app = PhaseApodizer(grid.zeros())
zmodes = make_zernike_basis(3, 1.0, grid)

separation = 2.5
mfd = 1.4
stellar_smf = SingleModeFiberInjection(focal_grid, make_gaussian_fiber_mode(mfd), position=np.array([separation, 0.0]))
planet_smf = SingleModeFiberInjection(focal_grid, make_gaussian_fiber_mode(mfd))
photometric_aperture = evaluate_supersampled(make_circular_aperture(mfd, center=np.array([separation, 0.0])), focal_grid, 4)

num_waves = 5
bandwidth = 0.2
wavelengths = 1 + np.linspace(-bandwidth/2, bandwidth/2, num_waves)

num_eval_waves = 51
eval_wavelengths = 1 + np.linspace(-bandwidth/2, bandwidth/2, num_eval_waves)

# Make the wavefront
norm = 0
for wave in wavelengths:
	wf = Wavefront(aperture, wave)
	wf.total_power = 1
	norm += prop(wf).power.max()

new_phase = grid.zeros()
regularization = 0
def gradient_function(theta):
	new_phase[aperture_mask] = theta
	app.phase = new_phase

	delta_theta = 0
	for wave in wavelengths:
		wf = Wavefront(aperture, wave)
		wf.total_power = 1
	
		wf_app = app(wf)
		wf_foc = prop(wf_app)
		
		delta_wf = stellar_smf.backward(stellar_smf(wf_foc))
		delta_wf.electric_field *= 2

		delta_E = prop.backward(delta_wf).electric_field 
		delta_theta += np.imag(np.conj(wf_app.electric_field) * delta_E)
	
	delta_theta /= num_waves
	delta_theta = delta_theta - zmodes.linear_combination(zmodes.coefficients_for(delta_theta))
	
	return delta_theta[aperture_mask] + regularization * theta

def callback(theta):
	
	new_phase[aperture_mask] = theta
	app.phase = new_phase

	corim = 0
	eta_star = 0
	eta_wave = []
	eta_planet = []
	photo_star = []
	for wave in eval_wavelengths:
		wf = Wavefront(aperture, wave)
		wf.total_power = 1
	
		wf_app = app(wf)
		wf_foc = prop(wf_app)
		wf_smf = stellar_smf.backward(stellar_smf(wf_foc))
		wf_planet_smf = planet_smf(wf_foc)
		
		mono_im = wf_foc.power / (norm * num_eval_waves / num_waves)
		corim += mono_im
		
		photo_star.append(np.sum(mono_im * photometric_aperture))
		
		eta_star += wf_smf.total_power / num_eval_waves
		
		eta_planet.append(wf_planet_smf.total_power)
		eta_wave.append(wf_smf.total_power)

	print(corim.max(), eta_star, np.mean(eta_planet))

	plt.clf()

	plt.subplot(1,4,1)
	imshow_field(app.phase, grid, vmin=-np.pi, vmax=np.pi, cmap='twilight')
	
	plt.subplot(1,4,2)
	imshow_field(wf_foc.phase, vmin=-np.pi, vmax=np.pi, cmap='twilight')
	circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
	plt.gca().add_patch(circ)
	plt.xlim([separation - 2.5, separation + 2.5])
	plt.ylim([-2.5, 2.5])

	plt.subplot(1,4,3)
	imshow_psf(corim, vmax=1, vmin=target_contrast / 10)
	circ = plt.Circle((separation, 0.0), mfd/2, fill=False, lw=2, ls='--', color='white')
	plt.gca().add_patch(circ)
	plt.xlim([separation - 2.5, separation + 2.5])
	plt.ylim([-2.5, 2.5])
	
	plt.subplot(1,4,4)
	plt.plot(eval_wavelengths, eta_wave, 'C0', lw=2)
	plt.plot(eval_wavelengths, photo_star, 'C2', lw=2)
	plt.ylim([1e-8, 1])
	plt.yscale('log')
	#imshow_field(wf_foc.phase, vmin=-np.pi, vmax=np.pi, cmap='twilight')

	plt.draw()
	plt.pause(0.1)

starting_phase = 0 * np.random.randn(grid.size)[aperture_mask]

plt.figure(figsize=(12,5))
res = accelerated_gradient_descent(gradient_function, 3e-2, max_iterations=1000, epsilon=1e-15, beta_start=starting_phase, callback=None, callback_frequentie=50)
plt.close()

new_phase[aperture>0] = res
final_phase = write_field(Field(new_phase, grid), str(paths.data/'smf_app_phase_example.fits'))

callback(res)
plt.show()