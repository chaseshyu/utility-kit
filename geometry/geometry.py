# -*- coding: utf-8 -*-
"""
Test code of assignment-06 of TIGP 2022 Fall - Scientific Programming

    This scripts is written for testing assignment-06 of TIGP 2022 Fall - 
Scientific Programming. The hand-ins would be imported as a function and 
tested by two random convex and concave polygons, their vertices and middle 
points of sides, and 40 random points. The results will be visulized via 
two subplots with green-red points. Green: True; Red: False.

TA example of assignment-06

Created by Chase J. Shyu <chaseshyu@gmail.com> on Nov 19th, 2022.
"""
import numpy as np

def is_inside_polygon(polygon, px, py=None, include_edges=True, 
                      ROE=1.5e-14) -> np.ndarray:
    """
    Check if the point is inside the polygon
    
    Created by Chase J. Shyu <chaseshyu@gmail.com> on Nov 16th, 2022.

    Algorithm:
        Ray casting algorithm

    Reference:
        https://en.wikipedia.org/wiki/Point_in_polygon

    Summary:
        Choose a direction and count the times of intersection between
    polygon and the line from the point in chosen direction. The point 
    is inside the polygon when the number of times is odd. Otherwise, 
    the point is outside the polygon. Special treatment is needed when 
    point is on the edge (vertices or sides) of polygon. The handle of 
    round-off errors is essential when the point is very close to the 
    edge of the polygon.

    Args:
        polygon (list[float, float] or list[list[float, float]]):
            A or multiple list of vertex point (x,y) pairs of polygon.
        px (float or list[float]):
            A or multiple x coordinate of point.
            Or x-y coordinate pairs when no py input vaiable.
        py (float or list[float]):
            A or multiple y coordinate of point.
        include_edges (bool, optional):
            If true includes point on the edges of poligon. 
            Defaults to True.
        ROE (float, optional):
            Round-off error for dealing with the point on the edges. 
            Defaults to 1.5e-14.

    Returns:
        bool or list[bool]: If point is inside the polygon or not.
        
    Schmetic diagram:
                   ^
                   |   ^
           ^ ^     --- | 
           | |   / |   \
         -------   o   /
        |  | |        /|
        |  | o        \|
         -----         \
           |   \       |\
           o    \      o \
                  --------
                  
    Example:
        polygon = [[[ 0.,0.], [0. ,0.5], [0.5,0.5], [0.5,0.]],
                   [[-1.,0.], [-1., 2.], [0.5,1.0], [0.0,0.]]]
        points = [[0.25,0.25],
                  [ 1.0, 0.1]]
        
        is_inside  = is_inside_polygon(polygon[0],points[0,0],points[0,1])
        is_inside  = is_inside_polygon(   polygon,points[0,0],points[0,1])
        is_inside  = is_inside_polygon(polygon[0],points[0])
        are_inside = is_inside_polygon(polygon[0],   points)
        are_inside = is_inside_polygon(   polygon,points[0])
        are_inside = is_inside_polygon(   polygon,   points)
        
                  
    """
    import numpy as np
    
    # change to numpy array for vector operations
    nv = np.array(polygon).shape[-2]
    # polygon = np.array(polygon)
    polygon_arr = np.array(polygon).reshape(-1,nv,2)
    nplg = polygon_arr.shape[0]

    if py is None:
        point_arr = np.array(px).reshape(-1,2)
    else:
        point_arr = np.array([px,py]).reshape(2,-1).transpose()

    nt = point_arr.shape[0]
    
    are_inside = np.full((nplg,nt), False)
    # zero intersection represents the point is outside the polygon
    for k, polygon in enumerate(polygon_arr):
        for j, point in enumerate(point_arr):
            is_inside = False
            # check the intersection of the polygon and the line directing 
            # to upward from the point
            for i in range(len(polygon)):
                # get side#i
                side = polygon[[i-1,i]]
                # if the x of points is within the x range of side#i
                # -> side#i is above or below the point
                prod = np.prod(side[:,0]-point[0])
                if prod < 0. or np.isclose(prod+point[0],point[0],atol=ROE,rtol=ROE):
                    # create two vectors
                    v0 = side[1] - side[0]
                    v1 = point - side[0]
                    # cross product for identifing relative position of side#i 
                    # and point 
                    cross = np.cross(v1,v0)
                    # if cross product is zero -> the point is on the side#i
                    if np.isclose(cross,0.,atol=ROE):
                        is_inside = include_edges
                        break
                    # if the side#i is above the point
                    # -> upward line intersects the side#i of polygon
                    elif cross * (side[1,0]-point[0]) > 0:
                        is_inside = not is_inside
            are_inside[k,j] = is_inside

    if nplg == nt == 1:
        return are_inside[0,0]
    elif nplg > 1 and nt > 1:
        return are_inside
    elif nplg > 1:
        return are_inside[:,0]
    else:
        return are_inside[0,:]

def test():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    print('\nStart testing...')

    ### setting of polygon
    npoly = 2
    # number of vertices
    nv = 6
    # average radius
    radius = [6.,5.]
    # irregularity
    irr = [0.7,.8]
    # spikiness
    spk = [0.2,0.55]
    titles = ['convex','concave']
    # generate random convex and concave polygons
    polygons = np.empty((npoly,nv,2))
    # random seeds of polygon
    seeds = [6, 24]
    for i in range(npoly):
        rng = np.random.default_rng(seeds[i])
        # random.seed(seeds[i])
        polygons[i] = generate_polygon(center=(0.,0.),avg_radius=radius[i],
                                    nv=nv,irr=irr[i],spk=spk[i],rng=rng)
    # make the order of points clockwise
    polygons = polygons[:,::-1]

    ### generate random test points
    nt = 500
    # newest version of pseudo-random number generator
    rng = np.random.default_rng(1)
    xy = rng.uniform(-9,9,(nt,2))
    ms_bl = np.clip(8000/nt,3,6)
    ms_pt = np.clip(1600/nt,0.5,1)

    ### generate test points on the sides
    ns = 1
    # ratio array for points on sides
    ratios = np.arange(1,ns+1) / (ns+1)
    # index for selecting point pairs
    ind = np.linspace((-1,0),(nv-2,nv-1),nv,dtype=int)
    # point pairs of sides of polygon
    tmp = polygons[:,ind]
    side_points = np.quantile(tmp,ratios,axis=2)
    # switch quantile points if slope is negative
    ind = tmp[:,:,1] < tmp[:,:,0]
    side_points[:,ind] = side_points[::-1,ind]
    side_points = side_points.transpose(1,0,2,3)

    ### show information of test points
    show_point_msg = False
    if show_point_msg:
        fmt = '%6.3f'
        temp = f'%2d:({fmt},{fmt})\n'
        for k in range(npoly):
            print('Test %s polygon:' % (titles[k]))
            msg = ''
            for i in range(nv):
                msg += temp % (i,polygons[k,i,0],polygons[k,i,1])
            
            for j in range(ns):
                msg += '#%02d points on sides:\n' % j
                for i in range(nv):
                    msg += temp % (i,side_points[k,j,i,0],side_points[k,j,i,1])
            print(msg)

        print('Test random points:')
        n_half = int(nt/2)
        temp = f'%2d:({fmt},{fmt}) %2d:({fmt},{fmt})\n'
        msg = ''
        for i in range(n_half):
            tx, ty = xy[:,0], xy[:,1]
            msg += temp % (i,tx[i],ty[i],i+n_half,tx[i+n_half],ty[i+n_half])
        print(msg)

    ### concatenate all test points
    xy = np.tile(xy, (npoly,1,1))
    tmp = side_points.reshape(npoly,-1,2)
    xy = np.concatenate((xy,polygons,tmp),axis=1)

    ### vectorize function
    testPolygon = np.vectorize(lambda i: is_inside_polygon(polygons[i],xy[i]),otypes=[np.ndarray])
    are_inside = testPolygon(np.arange(npoly)).T
    
    # are_inside = np.apply_along_axis(
    #     lambda i: is_inside_polygon(polygons[i],xy[i]), 0 , np.arange(npoly).reshape(1,-1)).T

    # are_inside = np.empty(xy.shape[0:2],dtype=bool)
    # for i in range(npoly):
    #     are_inside[i] = is_inside_polygon(polygons[i],xy[i,:,0],xy[i,:,1])

    #     are_inside[i] = np.apply_along_axis(lambda p: is_inside_polygon(polygons[i],p[0],p[1]), 1, xy[i])

    fig, axes = plt.subplots(1,npoly,figsize=(5*npoly,5),dpi=150)
    axes = np.array(axes).reshape(-1)
    
    for i in range(npoly):
        ax = axes[i]

        ax.add_patch(Polygon(polygons[i], ec='k', fc='None',label='Polygon',zorder=3))
        ax.plot(polygons[i,:,0],polygons[i,:,1],'D',ms=8,mec='k',color='lightgrey',label='Vertices',zorder=2)
        tmp = side_points[i].reshape(-1,2)
        ax.plot(tmp[:,0],tmp[:,1],'o',ms=9,mec='k',color='lightgrey',label='Points on sides',zorder=2)

        ax.plot(xy[i,:,0],xy[i,:,1],'ko',ms=ms_pt,label='Point center',zorder=5)
        ind = are_inside[i] == True
        ax.plot(xy[i,ind,0],xy[i,ind,1],'o',ms=ms_bl,mfc='green',mec='None',alpha=0.7,label='True',zorder=4)
        ind = are_inside[i] == False
        ax.plot(xy[i,ind,0],xy[i,ind,1],'o',ms=ms_bl,mfc='red',mec='None',alpha=0.7,label='False',zorder=4)

        if not i:
            if npoly == 1:
                ax.legend(fontsize='small',ncol=3,loc=3)
            else:
                fig.legend(fontsize='small',ncol=6,loc=8,bbox_to_anchor=(0.5,0.01))

        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(ls='--')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Test of %s polygon' % titles[i])

    fig.suptitle('is_inside_polygon')
    fig.tight_layout()
    if npoly != 1:
        fig.subplots_adjust(bottom=0.16)
    plt.show()
    fig.savefig('is_inside_polygon.png',dpi=450)

### function for generate random 2d polygon
# Reference:
#     https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
# 
# Modification:
#     Use advanced numpy PCG64 pseudo-random number generator.
from typing import Tuple
def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irr: float, spk: float,
                     nv: int, rng=np.random.default_rng()) -> np.ndarray:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irr (float):
            irregularity: variance of the spacing of the angles between 
            consecutive vertices.
        spk (float):
            spikiness: variance of the distance of each vertex to the 
            center of the circumference.
        nv (int):
            the number of vertices of the polygon.
        rng (np.random.Generator):
            Posudo-random number generator. Defaults to 
            numpy.random.default_rng().
    Returns:
        np.ndarray: list of vertices, in CCW order.
    """
    ### Parameter check
    if irr < 0 or irr > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spk < 0 or spk > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irr *= 2 * np.pi / nv
    spk *= avg_radius

    angle_steps = random_angle_steps(nv, irr, rng)
    angles = np.cumsum(angle_steps)
    ### original algorithm (same value)
    # angles = np.cumsum(np.roll(angle_steps,1)) - angle_steps[-1]
    # angles += rng.uniform(0, 2 * np.pi)

    ### now generate the points
    radius = rng.normal(avg_radius, spk,nv)
    radius = np.clip(radius, 0, 2*avg_radius)
    xy = radius * np.vstack((np.cos(angles),np.sin(angles)))
    points = center + xy.T

    return points
#
def random_angle_steps(steps: int, irregularity: float,rng=np.random.default_rng()) -> np.ndarray:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
        rng (np.random.Generator):
            Posudo-random number generator. Defaults to 
            numpy.random.default_rng().
    Returns:
        np.ndarray: the list of the random angles.
"""
    # generate n angle steps
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    angles = rng.uniform(lower,upper,steps)

    # normalize the steps so that point 0 and point n+1 are the same
    angles /= np.sum(angles) / (2 * np.pi)

    return angles
#
### end of reference

def main():
    test()

if __name__ == '__main__':
    main()