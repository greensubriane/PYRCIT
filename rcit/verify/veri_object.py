import numpy as np
import scipy.ndimage as ndimage


def object_identify_veri(obj_arr, thresh, footprints=500, f_coef=1/15, datamin=None):
    # obj_arr = np.nan_to_num(obj_arr)
    # default thresh is set to 'auto' in default situation
    if datamin is not None:
        obj_arr[obj_arr < datamin] = datamin

    objects, r_tot, obj_array, x_com = obj_operate(obj_arr, thresh, footprints)
    nobjects = len(objects)
    maxobjn = find_max_objidx(objects)

    return objects, nobjects, maxobjn, x_com, r_tot


def obj_operate(obj_arr, thresh, footprint):
    # Rmax = np.max(obj_arr)
    # if thresh == 'auto':
    # rstar = f * Rmax
    # else:
    rstar = float(thresh)
    mask = np.copy(obj_arr)
    mask[obj_arr < rstar] = False
    mask[obj_arr >= rstar] = True
    masked_data = mask
    labeled, num_objects = ndimage.label(mask)

    sizes = ndimage.sum(mask, labeled, list(range(num_objects + 1)))

    masksize = sizes < footprint
    remove_pixel = masksize[labeled]
    labeled[remove_pixel] = 0

    labels = np.unique(labeled)
    label_im = np.searchsorted(labels, labeled)

    # dic['objects'] = {}
    objects = {}

    # Total inputimg for objects
    R_objs_count = 0

    for ln, l in enumerate(labels):
        cy, cx = ndimage.measurements.center_of_mass(obj_arr, labeled, l)
        if ln == 0:
            # This is the centre of mass for the entire domain p.
            x_com = (cx, cy)
        else:
            objects[l] = {}
            objects[l]['CoM'] = (cy, cx)
            objects[l]['Rn'] = ndimage.sum(obj_arr, labeled, l)
            objects[l]['RnMax'] = ndimage.maximum(obj_arr, labeled, l)
            objects[l]['Vn'] = objects[l]['Rn'] / objects[l]['RnMax']
            if not np.isnan(objects[l]['Rn']):
                R_objs_count += objects[l]['Rn']

    r_tot = R_objs_count
    obj_array = labeled
    return objects, r_tot, obj_array, x_com


def find_max_objidx(objects):
    # objects are dic type variable
    """ Get the largest object number.
    """
    maxidx = 0
    for k, v in objects.items():
        if int(k) > maxidx:
            maxidx = int(k)
    return maxidx


def hessian_operators(obj_array):
    from skimage.feature import blob_doh
    blob_doh(image=obj_array)


def active_px(obj_array, fmt='pc'):
    """ Return number of pixels included in objects.
        Args:
            fmt (bool): if True, returns the active pixel count expressed as percentage.
    """
    active_px = np.count_nonzero(obj_array)
    tot_px = obj_array.size

    if fmt == 'pc':
        return (active_px / (tot_px * 1.0)) * 100.0
    else:
        return active_px, tot_px


'''
def plot(self, fpath, fmt='default', W=None, vrbl='REFL_comp',
         # Nlim=None,Elim=None,Slim=None,Wlim=None):
         ld=None, lats=None, lons=None, fig=None, ax=None):
    """ Plot basic quicklook images.
        Setting fmt to 'default' will plot raw data,
        plus objects identified.
    """
    if ld is None:
        ld = dict()
    nobjs = len(self.objects)

    if fmt == 'default':
       # if fig is None:
       F = Figure(ncols=2, nrows=1, figsize=(8, 4), fpath=fpath)
       # F.W = W
       with F:
            ax = F.ax[0]
            # Plot raw array
            BE = BirdsEye(ax=ax, fig=F.fig)

            # Discrete colormap
            import matplotlib as M
            cmap_og = M.cm.get_cmap('tab20')
            # cmap_colors = [cmap_og(i) for i in range(cmap_og.N)]
            color_list = cmap_og(np.linspace(0, 1, nobjs))
            # cmap = M.colors.ListedColormap(M.cm.tab20,N=len(self.objects))
            cmap = M.colors.LinearSegmentedColormap.from_list('discrete_objects', color_list, nobjs)
            # bounds = np.linspace(0,nobjs,nobjs+1)
            # norm = M.colors.BoundaryNorm(bounds,cmap_og.N)
            masked_objs = np.ma.masked_less(self.obj_array, 1)

            BE.plot2D(plottype='pcolormesh', data=masked_objs, save=False,
                      cb='horizontal',
                      # clvs=np.arange(1,nobjs),
                      W=W,
                      cmap=cmap, mplkwargs={'vmin': 1}, **ld, lats=lats, lons=lons)

            ax = F.ax[1]
            S = Scales(vrbl)
            BE = BirdsEye(ax=ax, fig=F.fig)
            BE.plot2D(data=self.raw_data, save=False,
                      W=W,
                      cb='horizontal', lats=lats, lons=lons,
                      cmap=S.cm, clvs=S.clvs, **ld)
    return
'''
