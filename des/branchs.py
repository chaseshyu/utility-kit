#!/usr/bin/env python
'''
Usage:
$ python plotBasin.py {start} {end} {interval} {-options}

'''
import sys, math
from datetime import datetime
import numpy as np
from numpy import linalg as LA
from types import SimpleNamespace
from sklearn.cluster import DBSCAN

import multiprocessing
from multiprocessing import Pool

sys.path.append('/path/to/des')
from Dynearthsol import Dynearthsol

from scipy import spatial, stats, interpolate
from scipy.spatial import Delaunay
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.patches as mpatchs

import warnings
warnings.filterwarnings("ignore")

# assign core number
thread_num = int( multiprocessing.cpu_count() / 2 )

# input data
modelname = 'result'
markersetname = 'markerset'
shape_sensitivity = 1.5

# index of material type
mattype = {}
mattype["sediment"] = 4

####
figsize = (12, 8)
xmax, xmin, zmax, zmin =  65, 25, 5, -15.
resolution = 0.025 # unit
aspect = 1

# color
srII_max, srII_min = -13, -17 #log
color_srII = 'rainbow'

class Filter():
    def marker(self,x, z, m, t):
        ind = (xmin <= x) * (x <= xmax) * \
            (zmin <= z) * (z <= zmax)
        x = x[ind]
        z = z[ind]
        m = m[ind]
        t = t[ind]
        return x, z, m, t
        
    def node(self,x,z,f):
        ind = (f >= 32 ) * (f <= 34)
        x = x[ind]
        z = z[ind]
        return x, z


class Branch():
    def __init__(self,ibranch,x,z):
        self.x = x
        self.z = z
        self.index = ibranch
        if x.size < 1:
            self.if_use = False
            return

        self.width = np.amax(x) - np.amin(x)
        self.tall = np.amax(z) - np.amin(z)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,z)
        self.slope = slope
        self.intercept = intercept
        self.r_value = r_value
        self.p_value = p_value
        self.std_err = std_err
        self.locx = np.mean(x)
        self.locz = np.mean(z)

        phi = np.arctan(self.slope)
        self.length = self.tall / np.sin(phi)
        self.phi = phi * 180. / np.pi
    
        if (self.width > self.tall and self.tall < 2.5) or (x.size == 0) or (r_value**2 < 0.1):
            self.if_use = False
        else:
            self.if_use = True


class Cluster():
  def __init__(self,icluster,x,z):
    self.branchs = {}
    self.index = icluster
    self.nnodes = x.size
    self.locx = (np.amax(x) + np.amin(x))/2.
    self.locz = (np.amax(z) + np.amin(z))/2.
    self.width = np.amax(x) - np.amin(x)
    self.tall = np.amax(z) - np.amin(z)
    self.x = x
    self.z = z

    self.if_branch = False
    if self.width < 1. or self.tall < 1. or x.size < 50:
        self.if_use = False
        return
    else:
        self.if_use = True

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,z)

    self.slope = slope
    self.intercept = intercept
    self.r_value = r_value
    self.p_value = p_value
    self.std_err = std_err

    if r_value**2 > 0.9:
        self.branchs[0] = Branch(0, x,z)

    else:
        # if r^2 < 0.9, find branchs
        self.inc = True
        self.get_branch_node(x,z)

        if self.inc:
            if self.znodes_x.size > 0:

                # Start to divide cluster to branchs
                self.divide_cluster(x,z)

                for i in range(self.npieces):
                    ind = (self.branch_lables == i)
                    self.branchs[i] = Branch(i, x[ind],z[ind])

  def divide_cluster(self,x,z):
    nbranch = self.znodes_z.size
    self.inodes_in_piece = np.array([0 for a in range(nbranch)])
    self.branch_lables = np.zeros(x.size)
    self.npieces = 1

    sub_xz = np.array([x,z]).T

    sub_pieces = []
    sub_pieces.append(sub_xz)
    ind_nodes = [False for b in range(nbranch)]

    for a in range(nbranch):
      ref_x, ref_z = self.znodes_x[a], self.znodes_z[a]
      ipiece = self.inodes_in_piece[a]
      ind_pi = (self.branch_lables == ipiece)

      ind = np.where(self.inodes_in_piece[a+1:] == ipiece)
      ind = ind[0] + a+1
      sub_nodex = np.array([ self.znodes_x[b] for b in ind ])

      # get points below branch node
      ind = (sub_xz[:,1] <= ref_z)
      below_ind = ind_pi * ind
      below_xz = sub_xz[below_ind]
      try:
        below_dz = np.amax(below_xz[:,1]) - np.amin(below_xz[:,1])
      except:
        below_dz = 0.

      if (not below_dz < 2.5) or a == 0 :

        sub_pieces[ipiece] = sub_xz[ind]

        ind = (sub_xz[:,1] > ref_z) * (sub_xz[:,0] <= ref_x)
        left_ind = ind_pi * ind
        left_xz = sub_xz[left_ind]

        ind = (sub_xz[:,1] > ref_z) * (sub_xz[:,0] > ref_x)
        right_ind = ind_pi * ind
        right_xz = sub_xz[right_ind]

        if not (left_xz.size < 50 or right_xz.size < 50):

          sub_pieces.append(left_xz)
          self.branch_lables[left_ind] = len(sub_pieces) - 1

          sub_pieces.append(right_xz)
          self.branch_lables[right_ind] = len(sub_pieces) - 1

          ind_nodes[a] = True

          # judge which region node belong to
          ind = (sub_nodex <= ref_x)
          ind = [b in sub_nodex[ind] for b in self.znodes_x]
          self.inodes_in_piece[ ind ] = self.npieces
          self.npieces += 1

          ind = (sub_nodex > ref_x)
          ind = [b in sub_nodex[ind] for b in self.znodes_x]
          self.inodes_in_piece[ ind ] = self.npieces
          self.npieces += 1

    self.znodes_x = self.znodes_x[ind_nodes]
    self.znodes_z = self.znodes_z[ind_nodes]

    if self.znodes_x.size > 0:
      self.if_branch = True

  def get_branch_node(self,x,z):
    bl = False
    alpha = shape_sensitivity

    points = [geometry.shape( geometry.Point(x[k],z[k]) ) for k in range(x.size)]
    while bl == False:
        print(alpha)
        try:
            concave_hull, _ = alpha_shape(points,alpha)
            bl = True
        except:
            alpha -= 0.1
            print('Cluster ',self.index,' reduce alpha to ', alpha)
            if alpha < 0.1:
                self.inc = False
                return

    # find largest polygon if object is multipolygon
    if (isinstance(concave_hull, geometry.multipolygon.MultiPolygon)):
      amax = 0.
      for p in concave_hull:
        a = p.area
        if amax < a:
          amax = a
          tmp = p
      concave_hull = tmp

    patch_x, patch_z = concave_hull.exterior.coords.xy

    patch_x = np.array(patch_x)
    patch_z = np.array(patch_z)
    slopes_z = np.gradient(patch_z)
    moving_sz = moving_average(slopes_z,n=3)

    ind_p = (moving_sz > 0.)
    m_ind_p = moving_average(ind_p,n=2)
    ind_n = (moving_sz <= 0.)
    m_ind_n = moving_average(ind_n,n=2)
    znodes_ind = (m_ind_p != 0) * (m_ind_n != 0) * ind_p

    tmp1, tmp2 = patch_x, patch_z
    tmp1[1:] = patch_x[:-1]
    tmp2[1:] = patch_z[:-1]
    znodes_x = tmp1[znodes_ind]
    znodes_z = tmp2[znodes_ind]
    nbranch = znodes_z.size

    ind = np.argsort(znodes_z)
    znodes_x = np.array([znodes_x[a] for a in ind])
    znodes_z = np.array([znodes_z[a] for a in ind])

    # find valley point
    ind_nodes = [True for b in range(nbranch)]
    sub_points = np.array([x,z]).T
    kdtree = spatial.KDTree(sub_points)
    for a in range(nbranch):
      xx, zz = znodes_x[a], znodes_z[a] - .5
      nn = kdtree.query([xx,zz],10)
      ind = (nn[0] < .25)
      if not True in ind:
        ind_nodes[a] = False

    self.znodes_x = znodes_x[ind_nodes]
    self.znodes_z = znodes_z[ind_nodes]


class Framework():
    def __init__(self,x,z):
        self.clusters = {}
        self.nnodes = x.size
        labels, self.n_clusters = clustering_structure(x, z)

        pool = Pool(processes=thread_num)
        # create cluster
        for i in range(self.n_clusters):
            ind = (labels == i)
            print('Creating cluster %d... ' % i)

            self.clusters[i] = Cluster(i, x[ind], z[ind])

        # clean up
        pool.close()
        pool.join()


def init_var(par,var,i):
    # parameter initiation
    var.rx = 0.20 * 280 / (xmax-xmin) * resolution # fix original resolution
    var.rz = var.rx / aspect

    # Size of regular grid
    var.nx = int((xmax - xmin) / var.rx)
    var.nz = int((zmax - zmin) / var.rz)

    # dimension of domain
    var.width = (xmax - xmin)
    var.height = (zmax - zmin)

    var.frame = var.des.frames[i]
    nnode = var.des.nnode_list[i]
    nelem = var.des.nelem_list[i]

    var.des.read_header(var.frame)
    
    coord = var.des.read_field(var.frame, 'coordinate')
    coord0 = var.des.read_field(var.frame, 'coord0')
    
    horizon = np.zeros((nnode), dtype=coord.dtype)

    horizon[:] = coord0[:,1] / 1.e3
    
    coordx = np.array(coord[:,0]) / 1.e3
    coordz = np.array(coord[:,-1]) / 1.e3
    par.coordz = coordz

    connectivity = var.des.read_field(var.frame, 'connectivity')
    # Calculate the center of element
    ecoord = np.array([coord[connectivity[e,:],:].mean(axis=0) for e in range(nelem)])
    area = np.zeros((nelem), dtype=coord.dtype)
    ecoordx = ecoord[:,0] / 1.e3
    ecoordz = ecoord[:,-1] / 1.e3

    # Area of element
    for e in range(nelem):
        cc = coord[connectivity[e,:],:]
        v1 = cc[1] - cc[0]
        v2 = cc[2] - cc[0]
        area[e] = LA.norm(np.cross(v1,v2)) / 2.e6

    # get plastic strain on element
    pls = var.des.read_field(var.frame,'plastic strain')

    # get strain rate
    strain_rate = var.des.read_field(var.frame, 'strain-rate')
    srII = second_invariant(strain_rate)
    # log of srII
    srII = np.log10(srII+1e-45)

    # Get surface nodes
    bcflag = var.des.read_field(var.frame, 'bcflag')
    surfx, surfz = var.filter.node(coordx,coordz,bcflag)
    
    marker_data = var.des.read_markers(var.frame, markersetname)
    nmarkers = marker_data['size']
    
    t = marker_data[markersetname + '.time'] /1.e6
    m = marker_data[markersetname + '.mattype']
    field = marker_data[markersetname + '.coord']
    tmp = np.zeros((nmarkers, 3), dtype=field.dtype)
    tmp[:,:var.des.ndims] = field
    
    x = np.array(tmp[:,0]) / 1.e3
    z = np.array(tmp[:,1]) / 1.e3

    x, z, m, t = var.filter.marker(x,z,m,t)
    
    var.xi[0] = np.linspace(xmin, xmax, var.nx)
    var.zi[0] = np.linspace(zmin, zmax, var.nz)
    var.xi[0], var.zi[0] = np.meshgrid(var.xi[0], var.zi[0])

    var.surfxi = np.linspace(xmin, xmax, var.nx)

    orders = np.argsort(surfx)

    var.surfzi = np.interp(var.surfxi, surfx[orders], surfz[orders])

    var.surf_alti = np.amax(var.surfzi)

    print(str(datetime.now())+' **Start to interpolate data with multiprocessing (thread=%d)' % thread_num)
    # Interpolate using delaunay triangularization

    pool = Pool(processes=thread_num)

    s_points = np.vstack((x,z)).T
    points = np.vstack((coordx,coordz)).T
    e_points = np.vstack((ecoordx,ecoordz)).T

    result_t = pool.apply_async(interpolate.griddata, (s_points, t, (var.xi[0], var.zi[0]),'linear') )
    result_m = pool.apply_async(interpolate.griddata, (s_points, m, (var.xi[0], var.zi[0]),'linear') )
    result_h = pool.apply_async(interpolate.griddata, (points,   horizon, (var.xi[0], var.zi[0]),'linear') )
    result_p = pool.apply_async(interpolate.griddata, (e_points, pls, (var.xi[0], var.zi[0]),'linear') )
    result_s = pool.apply_async(interpolate.griddata, (e_points, srII, (var.xi[0], var.zi[0]),'linear') )
    result_a = pool.apply_async(interpolate.griddata, (e_points, area, (var.xi[0], var.zi[0]),'linear') )

    try:
        ti = result_t.get()
    except:
        return False

    mi = result_m.get()
    hi = result_h.get()
    pi = result_p.get()
    si = result_s.get()
    ai = result_a.get()

    # clean up
    pool.close()
    pool.join()

    var.ti[0] = np.array(ti)
    var.mi[0] = np.array(mi)
    var.pi[0] = np.array(pi)
    var.si[0] = np.array(si)  
    var.ai[0] = np.array(ai)

    var.hi[0] = hi

    return True

def find_base(var):

    # Check if nan
    var.ti[0][(var.ti[0] == np.nan)] = 0.
    var.pi[0][(var.pi[0] == np.nan)] = 0.

    for k in range(var.xi[0][0,:].size):
        ind = (var.zi[0][:,k] > var.surfzi[k]) + (var.mi[0][:,k] < mattype["sediment"] - 0.5)
        var.ti[0][ind,k] = 0.

    var.botzi = np.array([])
    var.nbotzi = np.array([])
    var.mohozi = np.zeros(var.surfzi.size,dtype=float)

    # Find sediment basement
    for k in range(var.xi[0][0,:].size):
        b = False
        for j in range(var.xi[0][:,0].size):
            if (var.mi[0][j,k] >= mattype["sediment"]-0.25) and (var.zi[0][j,k] < var.surfzi[k]):

                var.botzi = np.append(var.botzi,var.zi[0][j,k])
                var.nbotzi = np.append(var.nbotzi,j)
                var.nbotzi[k] = j
                b = True
                break
        if not b:
            var.botzi = np.append(var.botzi,var.surfzi[k])
            var.nbotzi = np.append(var.nbotzi,0)

    var.base_alti = np.mean(var.botzi)

def meshProcessing(var):
    # Document horizon
    for k in range(var.xi[0][0,:].size):
        ind = (var.zi[0][:,k] > var.botzi[k])
        var.hi[0][ind,k] = 1.

    # Filter high displacement out
    var.mi[0][(var.mi[0] == np.nan)] = 0.


def unstructureProcessing(var):
    # Undimensional grid
    xil = np.reshape(var.xi[0],(var.nx*var.nz,1))
    zil = np.reshape(var.zi[0],(var.nx*var.nz,1))
    mil = np.reshape(var.mi[0],(var.nx*var.nz,1))
    hil = np.reshape(var.hi[0],(var.nx*var.nz,1))
    pil = np.reshape(var.pi[0],(var.nx*var.nz,1))
    ail = np.reshape(var.ai[0],(var.nx*var.nz,1))
    sil = np.reshape(var.si[0],(var.nx*var.nz,1))

    print(str(datetime.now())+' **Start to filter data')

    # Sort basement out using horizon
    ind = (hil <= 0. )
    var.xi[2],var.zi[2],var.pi[2],var.ai[2],var.mi[2],var.si[2] = \
        xil[ind],zil[ind],pil[ind],ail[ind],mil[ind],sil[ind]

    ishow = (var.si[2] != np.nan) & (var.ai[2] != np.nan) & (var.mi[2] != np.nan)
    var.xi[3],var.zi[3],var.pi[3],var.ai[3],var.mi[3],var.si[3] = \
        var.xi[2][ishow],var.zi[2][ishow],var.pi[2][ishow],var.ai[2][ishow],var.mi[2][ishow],var.si[2][ishow]

    ind = (var.si[3] >= -14) 
    var.xi[9], var.zi[9], var.pi[9], var.ai[9], var.mi[9] = \
        var.xi[3][ind], var.zi[3][ind], var.pi[3][ind], var.ai[3][ind], var.mi[3][ind]


def structureAnalysis(var):
    var.framework = Framework(var.xi[9], var.zi[9])  

    cluster_keys = var.framework.clusters.keys()
    for key in cluster_keys:
        cluster = var.framework.clusters[key]
        branch_keys = var.framework.clusters[key].branchs.keys()

        for bkey in branch_keys:
            branch = cluster.branchs[bkey]

            if branch.if_use:
                sys.stdout.write('Branch-%d-%d ' % (key,bkey) )
                sys.stdout.write('Length = %6.2f ' % branch.length)
                sys.stdout.write('Angle = %6.2f ' % branch.phi)
                sys.stdout.write('r_value = %6.2f ' % branch.r_value**2)
                print('Position: ( %6.2f, %6.2f )' % (branch.locx,branch.locz))


def clustering_structure(xi, zi):

    X = np.array([xi,zi]).T

    # Compute DBSCAN
    db = DBSCAN(eps=0.7, min_samples=10).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    return labels, n_clusters_


def main(par):
    des = Dynearthsol(par.data_path + modelname)
  
    if par.start == -1 or par.start > len(des.frames):
        par.start = len(des.frames)

    if par.end == -1 or par.end > len(des.frames):
        par.end = len(des.frames)

    print('* Start to convert DES to image.')
    print('  Frame range(x): %4d to %3d' % (xmin, xmax) )
    print('  Frame range(z): %4d to %3d' % (zmin, zmax) )
    print('  Time: %3d to %3d (%3d)' % (par.start,par.end,par.delta) )
    print('')

    for iii in range(par.start, par.end, par.delta):
        i = iii

        var = SimpleNamespace()

        var.xi = {}
        var.zi = {}
        var.vi = {}
        var.ti = {}
        var.mi = {}
        var.pi = {}
        var.si = {}
        var.ai = {}
        var.hi = {}
        var.filter = Filter()
        
        var.time_in_yr = des.time[i] / (365.2425 * 86400)

        var.filename = 'branchs_%4.4d' % i

        print(now()+' **Start to draw crust profile (%4.4d) and get the data' % i)

        var.des = des

        # convert mesh and marker data to uniform mesh
        inc = init_var(par,var,i)

        if inc == False:
            print('Error occured. Skip frame %3.3d' % i)
            continue

        find_base(var)

        inc = init_var(par,var,i)

        if inc == False:
            print('Error occured. Skip frame %3.3d' % i)
            continue

        find_base(var)

        # create the polygon of whole model
        points_1 = np.vstack((var.surfxi,var.surfzi)).T
        points_2 = np.array([[xmax, zmin], [xmin,zmin]])
        model_points = np.concatenate((points_1,points_2),axis=0)
        var.poly_model = mpatchs.Polygon(model_points,alpha=1.0,linewidth=0.2,fc='none',ec='black')

        print(now()+' **Start to document mesh data')    
        meshProcessing(var)

        # Unstructured processing
        print(now()+' **Start to document unstructured data')
        unstructureProcessing(var)

        var.if_struc = True
        if var.zi[9].size == 0:
            var.if_struc = False
            print('**No Slope would be ploted')

        if var.if_struc:
            print(now()+' **Start structure clustering')
            structureAnalysis(var)

        # visualization data
        print(now()+' **Start to plot')
        visualize(var)

        print(now()+' **End of script')


def visualize(var):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    # Plot Second Invariant
    ax = axes[0]
    cmap = cm.get_cmap(color_srII,100)
    srII_nl = int(1+(srII_max - srII_min)/0.1)
    srII_plot = ax.contourf(var.xi[0],var.zi[0],var.si[0],
            levels=np.linspace(srII_min,srII_max,srII_nl),extend='both',cmap=cmap,zorder=2,alpha=.7,vmax=srII_max,vmin=srII_min)

    var.poly_model.set_transform(axes[0].transData)
    for col in srII_plot.collections:
        col.set_clip_path(var.poly_model)

    ax = axes[1]
    if var.if_struc:
        clusters = var.framework.clusters

        for cluster in clusters.values():
            for branch in cluster.branchs.values():
                x = branch.x
                z = branch.z
                
                index = branch.index
                index = [index for i in x]
                ax.plot(x,z,'o')

        for cluster in clusters.values():
            if cluster.if_branch:
                x = cluster.znodes_x
                z = cluster.znodes_z
                ax.plot(x, z,'o',color='black',zorder=9)


    # Plot surface
    for i in range(2):
        axes[i].plot(var.surfxi,var.surfzi,color='darkblue',zorder=9)

    # Plot color bars
    orientation = 'vertical'

    divider = make_axes_locatable(axes[0])
    ax_cb = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(srII_plot, cax=ax_cb,orientation=orientation)        
    cb.set_label(r'Strain rate II (log)')
    # Set tick interval
    cb.ax.yaxis.set_major_locator(MultipleLocator(1))
    # tick_locator = ticker.MaxNLocator(nbins=6)
    # cb.locator = tick_locator
    # cb.update_ticks()

    print(now()+' **Start to output .png')

    # or if you want differnet settings for the grids:
    for i in range(2):
        # Fix aspect of x-y
        axes[i].set_aspect(aspect)
        axes[i].set_xlim(xmin, xmax)
        axes[i].set_ylim(zmin, zmax)

        axes[i].xaxis.set_major_locator(MultipleLocator(10))
        axes[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        axes[i].yaxis.set_major_locator(MultipleLocator(5))
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(5))

        axes[i].grid(which='minor', linestyle='dotted', alpha=0.5)
        axes[i].grid(which='major', linestyle='dotted', alpha=0.5)

        axes[i].set_xlabel(r'x ($km$)', labelpad=0, fontsize=20)
        axes[i].set_ylabel(r'z ($km$)', fontsize=20)

    plt.tight_layout()
    plt.savefig(var.filename + '.png', dpi=300)
    plt.close()
    return


def now(): return str(datetime.now())


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n] = a[:n]*n
    return ret / n


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        if (s-a)<=0. or (s-b)<=0. or (s-c)<=0.:
          area = 0.
        else:
          area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area > 0.:
          circum_r = a*b*c/(4.0*area)
        else:
          circum_r = 0.
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

# from 2vtk of DES
def second_invariant(t):
    '''The second invariant of the deviatoric part of a symmetric tensor t,
    where t[:,0:ndims] are the diagonal components;
      and t[:,ndims:] are the off-diagonal components.'''
    nstr = t.shape[1]

    # second invariant: sqrt(0.5 * t_ij**2)
    if nstr == 3:  # 2D
        return np.sqrt(0.25 * (t[:,0] - t[:,1])**2 + t[:,2]**2)
    else:  # 3D
        a = (t[:,0] + t[:,1] + t[:,2]) / 3
        return np.sqrt( 0.5 * ((t[:,0] - a)**2 + (t[:,1] - a)**2 + (t[:,2] - a)**2) +
                        t[:,3]**2 + t[:,4]**2 + t[:,5]**2)


if __name__ == '__main__':    
    par = SimpleNamespace()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    else:
        for arg in sys.argv[1:]:
            if arg.lower() in ('-h', '--help'):
                print(__doc__)
                sys.exit(0)

    # data location
    par.data_path = './'
    if '-dir' in sys.argv:
        ind = sys.argv.index("-dir") + 1
        par.data_path = sys.argv[ind]
        del sys.argv[ind]
        del sys.argv[ind-1]

    if len(sys.argv) < 2:
        par.start = 0
    else:
        par.start = int(sys.argv[1])

    if len(sys.argv) < 3:
        par.end = par.start + 1
    else:
        par.end = int(sys.argv[2])

    if len(sys.argv) < 4:
        par.delta = 1
    else:
        par.delta = int(sys.argv[3])  

    main(par)
