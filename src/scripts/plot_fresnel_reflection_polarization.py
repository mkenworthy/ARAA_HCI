import numpy as np
from matplotlib import pyplot as plt
import paths

def fresnel_reflection_coefficients(n1, n2, angle_of_incidence):
	'''Calculates the fresnel reflection amplitude coefficient for an interface.

    Parameters
    ----------
    n1 : array_like
        Refractive index of the first medium
	n2 : array_like
		Refractive index of the second medium
	angle_of_incidence : array_like
		The angle of incidence with respect to the normal of the interface.

    Returns
    -------
	r_s, r_p
		The reflection coefficients for s and p polarization.
    
	'''
	n_rel = n2 / n1
	n_cos_theta_out = np.sqrt(n_rel**2 - np.sin(np.deg2rad(angle_of_incidence))**2)
	cos_theta_in = np.cos(np.deg2rad(angle_of_incidence))
	
	r_s = (cos_theta_in - n_cos_theta_out) / (cos_theta_in + n_cos_theta_out)
	r_p = (-n_rel**2 * cos_theta_in + n_cos_theta_out) / (n_rel**2 * cos_theta_in + n_cos_theta_out)

	return r_s, r_p



angle_of_incidence = np.linspace(0, 90.0, 128)

# n1 : air
n1 = 1.0
# n2 : silver
n2 = 0.051585 - 1j * 3.9046

r_s, r_p = fresnel_reflection_coefficients(n1, n2, angle_of_incidence)

amp_s = abs(r_s)	
phase_s = np.angle(r_s)
grad_s = np.gradient(phase_s, angle_of_incidence, edge_order = 2)

amp_p = abs(r_p)
phase_p = np.angle(r_p)
grad_p = np.gradient(phase_p, angle_of_incidence, edge_order = 2)

# Make the Taylor expansion around 45.0 deg angle of incidence
reduced_angles = np.linspace(40.0, 50.0, 128)
slope_s = np.interp(45.0, angle_of_incidence, phase_s) + np.interp(45.0, angle_of_incidence, grad_s) * (reduced_angles - 45.0)
slope_p = np.interp(45.0, angle_of_incidence, phase_p) + np.interp(45.0, angle_of_incidence, grad_p) * (reduced_angles - 45.0)

# Plot the results
plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.plot(angle_of_incidence, amp_s, color='C0', lw=2, label='S-polarization')
plt.plot(angle_of_incidence, amp_p, color='C1', lw=2, label='P-polarization')
plt.legend()
plt.xlabel('angle of incidence (deg)')
plt.ylabel('$|r|$')
plt.ylim([0.95, 1.01])

plt.subplot(2,1,2)
plt.axvline(45.0, 0, 1, color='lightgray', lw=2)
plt.plot(angle_of_incidence, phase_s, color='C0', lw=2, label='S-polarization')
plt.plot(reduced_angles, slope_s, color='C0', lw=10, alpha=0.25)
plt.plot(angle_of_incidence, phase_p, color='C1', lw=2, label='P-polarization')
plt.plot(reduced_angles, slope_p, color='C1', lw=10, alpha=0.25)
plt.legend()
plt.xlabel('angle of incidence (deg)')
plt.ylabel('$\phi$')

plt.tight_layout()
plt.savefig(paths.figures/'./fresnel_reflection_polarization.pdf')
#plt.show()