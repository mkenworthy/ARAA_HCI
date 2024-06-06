from hcipy import *
import numpy as np
from matplotlib import pyplot as plt
import paths

if __name__ == "__main__":
	# Making the grids
	n = 512
	nbin = 2

	grid = make_pupil_grid(n * nbin)
	sub_grid = make_pupil_grid(n)
	theta = Field(grid.as_('polar').theta, grid)

	# The vortex mask generation code
	charge = 2
	vortex_mask = np.angle(subsample_field(np.exp(1j * charge * theta), nbin)) + np.pi
	
	# The mask generation code of the four quadrant phase mask
	f_charge = 2
	fq_mask = grid.ones()
	for i in range(f_charge):
		s = grid.rotated(i * np.pi/f_charge).x
		fq_mask *= (-1)**(i) * (2 * (s >= 0) - 1)
	fq_mask = subsample_field(Field(np.pi/2 * fq_mask + np.pi/2, grid), nbin)

	# Plotting settings
	fig = plt.figure(figsize=(6/0.97, 3. / (0.9-0.18)))
	gs = fig.add_gridspec(1, 2, left=0.01, bottom=0.18, right=0.99, top=0.9, wspace=0.01, hspace=0)
	(ax1, ax2) = gs.subplots(sharey='row')
	
	
	imshow_field(fq_mask, ax=ax1, vmin=0, vmax=2 * np.pi, cmap='twilight')
	ax1.set_title('Four-quadrant phase mask')
	
	im = imshow_field(vortex_mask, ax=ax2, vmin=0, vmax=2 * np.pi, cmap='twilight')
	ax2.set_title('Charge 2 vortex mask')

	for ax in fig.get_axes():
		ax.label_outer()
		ax.axis('off')
	
	cbar_ax = fig.add_axes([0.05, 0.09, 0.9, 0.075])
	plt.text(np.pi, -1., 'phase (rad)', horizontalalignment='center')

	fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

	plt.savefig(paths.figures / 'scale_invariant_masks.pdf')
