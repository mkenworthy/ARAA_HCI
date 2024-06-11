import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('MacOSX')

figsize = (14,6)
fig, ax = plt.subplots(1,1,figsize=figsize, facecolor='lightskyblue')

# for figure placement, I want to use the centre of the inserted figures as a coordinate, along with a width and height of the figure
# where we ALSO make sure the axes are the same scale. For rectangular figures, this doesn't work, so we normalise to the height of the Figure so that the new coord system y-axis goes from -0.5 to 0.5 and the x axis is scaled so that it can go greater than unity.

def placid(xc,yc,xs,ys,fig):
	'placid - given centre (xc,yc) and side lengths (xs,ys) and Figure return 4-tuple suitable for Figure.add_axes'

	w = fig.get_figwidth()
	h = fig.get_figheight()
	R = w/h

	d = lambda a,R: a/R+0.5 # correctly adjusts the aspect ratio for the Figure

	# calculate lower left and upper right corners of the Axes
	x1=d(xc-(xs/2),R)
	x2=d(xc+(xs/2),R)
	y1=d(yc-(ys/2),1.)
	y2=d(yc+(ys/2),1.)

	# return the lower left corner and the width and height
	return ((x1,y1,x2-x1,y2-y1))

ax.plot(np.arange(5))

a1 = fig.add_axes((0., 0,1,1))
a1.set_xlim(0,1)
a1.set_ylim(0,1)
s=0.2
p = 0.15
y = 0.3
a2 = fig.add_axes(placid(-5*p, y, s, s, fig))
a3 = fig.add_axes(placid(-3*p, y, s, s, fig))
a4 = fig.add_axes(placid(  -p, y, s, s, fig))
a5 = fig.add_axes(placid(   p, y, s, s, fig))
a6 = fig.add_axes(placid( 3*p, y, s, s, fig))
a7 = fig.add_axes(placid( 5*p, y, s, s, fig))


line = mpl.lines.Line2D((0,.1,.2),(0.1,0.1,0.3),color='red')
lens = mpl.patches.Ellipse((0.5, 0.5), 0.02, 0.4,color='red')

# TODO add color blind friendly pallette

a1.add_artist(line)
a1.add_artist(lens)

y0 = 0.4

rayheight = 0.2





plt.draw()
plt.show()