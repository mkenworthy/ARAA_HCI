import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('MacOSX')

figsize = (14,6)
#fig, ax = plt.subplots(1,1,figsize=figsize, facecolor='lightskyblue')

fig = plt.figure(figsize=figsize, facecolor='lightskyblue')
# for figure placement, I want to use the centre of the inserted figures as a coordinate, along with a width and height of the figure
# where we ALSO make sure the axes are the same scale. For rectangular figures, this doesn't work, so we normalise to the height of the Figure so that the new coord system y-axis goes from -0.5 to 0.5 and the x axis is scaled so that it can go greater than unity.

axall = fig.add_axes([0.1,0.1,0.8,0.8])

axray = fig.add_axes([0.15,0.2,0.7,0.4])

def s(d):
	'FREE SPACE: d - distance'
	return np.array([[1 , d],
    	[0, 1]])

def f(f):
	'LENS: f - focal length'
	return np.array([[1 , 0],
    	[-1/f, 1]])


def make_coro_rays(fp1, dx=5):

	to_lens1 = np.matmul(s(dx),fp1)
	lens1 = np.matmul(f(dx),to_lens1)
	to_pp1 = np.matmul(s(dx),lens1)
	to_lens2 = np.matmul(s(dx),to_pp1)
	lens2 = np.matmul(f(dx),to_lens2)
	to_fp2 = np.matmul(s(dx),lens2)

	rays = np.stack((fp1,to_lens1,to_pp1,to_lens2,to_fp2))

	# pull out the y positions of the rays at each optical element
	ray = rays[:,0,:]

	return ray

# fan of rays from a single point
thetas = np.linspace(-0.1,0.1,17)
x = np.zeros_like(thetas)+0.1
fp1 = np.vstack((x,thetas))

# draw all the rays through the coronagraph
ray = make_coro_rays(fp1)

xpos = np.arange(ray.shape[0])

#axray.plot(xpos,ray, color='gray', alpha=0.5, zorder=1)

axray.plot(xpos[0:3],ray[0:3], color='blue', zorder=5)

lens1 = mpl.patches.Ellipse((1.0, 0.0), 0.05, 1.4, color='gray',alpha=0.8)
lens2 = mpl.patches.Ellipse((3.0, 0.0), 0.05, 1.4, color='gray',alpha=0.8)

# TODO add color blind friendly pallette

axray.add_artist(lens1)
axray.add_artist(lens2)
axray.set_ylim(-0.75,0.75)

pupmax = 0.2
rfps = 0.1

# add the Lyot stop
lyot1 = mpl.patches.Rectangle((2.0,pupmax),0.05,(0.7-pupmax),color='blue')
lyot2 = mpl.patches.Rectangle((2.0,-pupmax),0.05,-(0.7-pupmax),color='blue')
axray.add_artist(lyot1)
axray.add_artist(lyot2)

clipped = (np.abs(ray[2])<=pupmax)

t = ray[2:]

plt.plot(xpos[2:],t[:,clipped], color='blue', alpha=0.5)

plt.show()

quit() 

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


# 




plt.draw()
plt.show()