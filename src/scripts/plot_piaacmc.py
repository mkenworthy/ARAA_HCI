from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

target_contrast = 1e-8

grid = make_pupil_grid(512, 1.1)
apodized_aperture = Field(read_field(str(paths.scripts/'piaacmc_apodization_example.fits')), grid)

lyot_mask = evaluate_supersampled(make_circular_aperture(0.95), grid, 4)
lyot_stop = Apodizer(lyot_mask)

# focal plane parameters
focal_grid = make_focal_grid(q=5, num_airy=15)
prop = FraunhoferPropagator(grid, focal_grid)

# PIAACMC code
phase_mask_diameter = 1.2
coronagraph_grid = make_focal_grid(q=101, num_airy=4 * 5 * 1.10475 * phase_mask_diameter / 2)
phase_mask = read_field(str(paths.scripts/'piaacmc_fpm_phase_example.fits'))
num_pix = coronagraph_grid.shape[0] - phase_mask.grid.shape[0]
phase_mask = Field(np.pad(phase_mask.shaped, num_pix // 2).ravel(), coronagraph_grid)

cmc = PhaseApodizer(phase_mask)
piaacmc = LyotCoronagraph(grid, cmc, lyot_stop=None, focal_plane_mask_grid=coronagraph_grid)
coronagraph_propagator = piaacmc.prop

# The intermediate focal grid
#focal_plane_mask_low_res = Apodizer(1 - evaluate_supersampled(make_circular_aperture(fpm_diameter), focal_grid, 4))

#
wf = Wavefront(apodized_aperture)
wf.total_power = 1

norm = prop(wf).power.max()
norm2 = coronagraph_propagator(wf).power.max()

wf_apodized = wf

# First focal plane wavefronts
wf_pre_fpm = coronagraph_propagator(wf_apodized)
wf_post_fpm = cmc(wf_pre_fpm)

# Lyot Plane wavefronts
wf_lyot = piaacmc(wf_apodized)
wf_post_lyot = lyot_stop(wf_lyot)

# Science focal plane wavefronts
wf_foc = prop(wf_post_lyot)

corim = wf_foc.power / norm

plt.figure(figsize=(14,8))

plt.subplot(2,4,1)
plt.title('Apodizer')
imshow_field(apodized_aperture / apodized_aperture.max(), grid, cmap='gray', vmin=0, vmax=1)
plt.subplot(2,4,2)
plt.title('Focal plane mask')
imshow_field(np.angle(np.exp(1j * cmc.phase)), cmap='twilight', vmin=-np.pi, vmax=np.pi)

plt.subplot(2,4,5)
plt.title('PSF before focal plane mask')
imshow_psf(wf_pre_fpm.power / norm2, vmax=1, vmin=target_contrast, colorbar=False)
plt.subplot(2,4,6)
plt.title('PSF after focal plane mask')
imshow_psf(wf_post_fpm.power / norm2 + 1e-15, vmax=1, vmin=target_contrast, colorbar=False)

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
plt.savefig(paths.figures/'piaacmc.pdf')