import rasterio

import numpy as np
import pymc as pm
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from rasterio.warp import reproject
import shapely.geometry as geom
import geopandas as gpd
from rasterio.mask import geometry_mask
from rasterio.warp import reproject, calculate_default_transform
from matplotlib.colors import to_rgba
from matplotlib import patheffects
from scipy.signal import savgol_filter
import copy


# Various custom functions to deal with raster data

# Simple object made to carry around all this information
class GaussianProcessPrior:

    def __init__(self, sigma: float, lengthscale: float,
                 mean_function: callable, obs_variance: float or np.array):
        self.sigma = sigma
        self.lengthscale = lengthscale
        self.mean_function = mean_function
        self.obs_variance = obs_variance


# Object to easily deal with raster data and flowline data. Wraps rasterio functions.
class SimpleRaster:
    def __init__(self, im: np.array, meta: dict, tag=None, name=None):

        """
        :param im: 2D array from rasterio
        :param meta: dict from rasterio
        :param tag: short name
        :param name: longer name
        """

        self.im = im
        self.meta = meta
        self.tag = tag
        self.name = name
        self.extent = self._get_extent()
        self.r = None
        self.y = None
        self.f = None
        self.prior = None
        self.gp = None
        self.trace = None
        self.mesh = None
        self.cubic = True
        self.normalize = False
        self.obs_on_mesh = None

    def save(self, savepath=None):
        if savepath is None:
            assert self.tag is not None, 'Must either provide a savepath or define a tag for the SimpleRaster'
            savepath = self.tag + '.tif'

        self.meta['dtype'] = self.im.dtype

        with rasterio.open(savepath, 'w+', **self.meta) as out:
            out.write(self.im.astype(self.meta['dtype']), 1)

    def set_flowline_data(self, flowline, smooth=False):

        """
        given a shapely line object, extracts data profile y along coordinate r
        :param flowline: shapely line object
        :param smooth: bool, set to smoothen the profile with an arbitrarily defined savgol fiter using scipy
        :return:
        """

        self.r, self.y = extract_values_from_line(flowline, self.im, self.meta)

        if smooth and len(self.r) > 7:
            self.y = savgol_filter(self.y, 7, 1)

        if len(self.r) >= 2:
            self._update_f()

    def _update_f(self, dropna=False):

        # f is an interpolation function of the flowline profile

        if dropna:
            ii = ~np.isnan(self.y)
            self.r = self.r[ii]
            self.y = self.y[ii]
        if self.cubic:
            self.f = CubicSpline(self.r, self.y, extrapolate=False)
        else:
            self.f = interp1d(self.r, self.y, kind='slinear', fill_value=np.nan)

    def set_1Dprior(self, sigma: float = None, lengthscale: float = None,
                    mean_function: callable = None, obs_variance: float or np.array = 0,
                    normalize=False, obs_on_mesh=False):

        """
        Prior of the Gaussian Process (GP) used in the MCMC. Note that the prior is always 1D!
        For more information about Gaussian Processes and pymc2, click on this link from Patil (2010)
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.385.4366&rep=rep1&type=pdf
        :param sigma: GP amplitude, how wavy the random samples are on the y-axis
        :param lengthscale: GP lengthscale, how wavy the random samples are on the x-axis
        :param mean_function: How the random samples behave on average. Defaults to the 0-function
        :param obs_variance: How much should the random samples deviate from the observations.
                             Observations are taken from self.r and self.y
        :param normalize: Obsolete
        :param obs_on_mesh: Obsolete
        :return:
        """

        self.normalize = normalize
        self.obs_on_mesh = obs_on_mesh

        if sigma is None:
            sigma = self.y.std()

        if lengthscale is None:
            lengthscale = max(np.diff(self.r))

        if mean_function is None:
            mean_function = lambda x: self.y.mean() * np.ones_like(x)

        if isinstance(mean_function, str):
            mean_function = eval(mean_function)

        if obs_variance is None:
            obs_variance = (np.abs(self.y).max() * 0.1) ** 2

        if np.isscalar(obs_variance):
            obs_variance = obs_variance * np.ones_like(self.y)

        if len(obs_variance) != len(self.y):
            raise AttributeError('Length of obs_variance array must be the same as the length of observation array')

        self.prior = GaussianProcessPrior(sigma, lengthscale, mean_function, obs_variance)

    def set_gaussianprocess(self, mesh, positive_init=False, init_vals=None, lowest=False):

        """
        Defines the pymc2 GP according to the GaussianProcessPrior object
        :param mesh: Prescribed 1D mesh
        :param positive_init: Some GP needs to have strictly positive values (width)
        :param init_vals: If initial values need to be prescribed
        :param lowest: Special case for the bed elevation
        :return:
        """

        if self.prior is None:
            raise ValueError('A prior must have been set')

        self.mesh = mesh

        ii = self.r <= mesh[-1]
        self.r = np.array(self.r)[ii]
        self.y = self.y[ii]

        M = pm.gp.Mean(self.prior.mean_function)
        C = pm.gp.Covariance(eval_fun=pm.gp.cov_funs.gaussian.euclidean,
                             amp=self.prior.sigma, scale=self.prior.lengthscale, relative_precision=1e-6)

        if len(self.y) != 0:
            if self.f is not None and self.obs_on_mesh:
                obs_var = CubicSpline(self.r, self.prior.obs_variance, extrapolate=False)(self.mesh)
                ii = ~np.isnan(obs_var)
                pm.gp.observe(M, C, self.mesh[ii], self.f(self.mesh[ii]), obs_var[ii])
            else:
                pm.gp.observe(M, C, self.r, self.y, self.prior.obs_variance)

        self.M = M
        self.C = C

        if positive_init and init_vals is None:
            init_vals = np.abs(pm.gp.Realization(M, C)(self.mesh))

        if lowest and (init_vals is not None):
            ini2 = pm.gp.Realization(M, C)(self.mesh)
            init_vals = np.vstack((init_vals, ini2)).min(axis=0)

        self.gp = pm.gp.GPSubmodel(self.tag, M, C, self.mesh, init_vals=init_vals)

    def plot1D(self, ax=None, plot_fed_data=True, show_trace=True, show_prior=True,
               line=False, consecutive=True,
               std=True, conf=True, mean=True, smooth=True, **kwargs):

        # Wrapper to easily plot the flowline profile

        z0 = 0
        if 'zorder' in kwargs:
            z0 = kwargs.pop('zorder')

        if self.r is None and self.trace is None:
            raise AttributeError('Must extract flowline data first or define trace and mesh')
        if ax is None:
            fig, ax = plt.subplots()
        p = None
        if plot_fed_data is True or self.trace is None or self.f is None or line is True:
            if not line:
                p = ax.scatter(self.r, self.y, ec='k', alpha=0.6, zorder=z0+5, **kwargs)
            else:
                ii = np.ones_like(self.r).astype(bool)
                if consecutive:
                    ii = np.hstack((np.abs(np.diff(np.diff(self.r))) < 1e-6, True, True))
                p = ax.scatter(self.r[~ii], self.y[~ii], ec='k', alpha=0.6, zorder=z0+5, **kwargs)
                ax.plot(self.r[ii], self.y[ii], zorder=z0+5, ls='dashed', **kwargs)
        """elif self.f is not None:
            if (self.obs_on_mesh or self.obs_on_mesh is None) and (len(self.mesh) < 50):
                p = ax.scatter(self.mesh, self.f(self.mesh), ec='k', zorder=5, **kwargs)
            else:
                p = ax.plot(self.r, self.y, zorder=100, ls='dashed',
                            path_effects=[patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()],
                            solid_capstyle='round',
                            **kwargs)"""
        if p is None:
            color = 'tab:blue'
        else:
            try:
                color = to_rgba(p.get_facecolors()[0])
            except AttributeError:
                color = p[0].get_color()
        if 'c' in kwargs.keys():
            color = kwargs.pop('c')
        if self.mesh is not None:
            r = np.linspace(self.mesh[0], self.mesh[-1], 100)
            if show_trace and self.trace is not None:
                plot_trace(self.trace, self.mesh, color=color, ax=ax, n=len(r),
                           std1=std, std2=conf, mean=mean, smooth=smooth, zorder=z0, **kwargs)
                # gelman_rubin = pm.gelman_rubin(self.trace)
                # ax.text(0.05, 0.1, f'{gelman_rubin:.2f}', ha='center', va='center', transform=ax.transAxes)
            if (self.gp is not None and self.tag != 'velocity') and show_prior:
                up = self.gp.M(r) + np.sqrt(self.gp.C(r))
                low = self.gp.M(r) - np.sqrt(self.gp.C(r))
                ax.plot(r, up, ls='dashed', c='grey')
                ax.plot(r, low, ls='dashed', c='grey')
        return ax

    def plot2D(self, ax=None, **kwargs):

        # wrapper to plot the whole image

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.im, extent=self.extent, **kwargs)
        return ax

    def traceplot(self, ax=None, index=None, label=False, bounded=True, histogram=True, gr_lim=None):

        # plot the MCMC trace

        if ax is None:
            fig, ax = plt.subplots()
            label = True

        if index is not None:
            for trace in self.trace[:, :, index]:
                ax.plot(trace, alpha=0.5)
            if label:
                ax.set_xlabel('Step')
                ax.set_ylabel(self.name)
            gr = pm.gelman_rubin(self.trace[:, :, index])
            ax.text(0.8, 0.1, f'{gr:.2f}', transform=ax.transAxes)
        else:

            """ 
            # idea 1
            chain = self.trace[0].T
            ratio = chain.shape[1] / chain.shape[0] / 2
            chain = (chain - chain.min(axis=1)[:, None]) / (chain.max(axis=1) - chain.min(axis=1))[:, None]
            ax.imshow(chain)
            ax.set_aspect(ratio)
            ax.set_xlabel('Step')
            ax.set_ylabel('Grid number')
            """

            if histogram:
                gr = pm.gelman_rubin(self.trace)
                n = int(np.sqrt(len(gr)))
                ax.hist(gr, n, fc='tab:blue', ec='k', alpha=0.8)
                ax.set_xlim(*gr_lim)

            else:
                # idea 2
                y = np.arange(0, self.trace.shape[-2])
                dx = np.arange(0, self.trace.shape[-1])
                for true_chain in self.trace:
                    chain = true_chain.copy()
                    chain = (chain - chain.min(axis=0)) / (chain.max(axis=0) - chain.min(axis=0))
                    chain += dx - 0.5
                    c = None
                    for cell in chain.T:
                        l = ax.plot(cell, y, c=c, alpha=0.5)
                        c = l[0].get_color()
                if label:
                    ax.set_xlabel('Grid point')
                    ax.set_ylabel('Step')

                ax.set_xticks(dx[::2])
                ax.set_ylim(y[0], y[-1])
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                ax2 = ax.twinx()
                ax2.plot(pm.gelman_rubin(self.trace),
                         path_effects=[patheffects.Stroke(linewidth=3, foreground='black'), patheffects.Normal()])
                if bounded:
                    ax2.set_ylim(0.95, 1.1)

        return ax

    def _get_extent(self):
        try:
            extent = rasterio.transform.array_bounds(self.meta['height'], self.meta['width'], self.meta['transform'])
            extent = [extent[i] for i in [0, 2, 1, 3]]
        except TypeError:
            return None
        return extent

    def _process_trace(self):
        axis = tuple([i for i in range(len(self.trace.shape) - 1)])  # stack all the way to the last axis
        up = np.quantile(self.trace, 0.975, axis=axis)
        low = np.quantile(self.trace, 0.025, axis=axis)
        std = np.std(self.trace, axis=axis)
        mean = np.mean(self.trace, axis=axis)

        up_f = CubicSpline(self.mesh, up, extrapolate=False)
        low_f = CubicSpline(self.mesh, low, extrapolate=False)
        std_f = CubicSpline(self.mesh, std, extrapolate=False)
        mean_f = CubicSpline(self.mesh, mean, extrapolate=False)

        return mean_f, std_f, low_f, up_f

    def __add__(self, other):
        if isinstance(other, SimpleRaster):
            im = other.im
            y = other.y
        else:
            im = other
            y = other

        ras = copy.copy(self)
        try:
            ras.im = self.im + im
        except TypeError:
            pass
        try:
            ras.y = ras.y + y
        except TypeError:
            pass
        return ras

    def __sub__(self, other):

        if isinstance(other, SimpleRaster):
            im = other.im
            y = other.y
        else:
            im = other
            y = other

        ras = copy.copy(self)
        try:
            ras.im = self.im - im
        except TypeError:
            pass
        try:
            ras.y = ras.y - y
        except TypeError:
            pass
        return ras

    def __mul__(self, other):
        if isinstance(other, SimpleRaster):
            im = other.im
            y = other.y
        else:
            im = other
            y = other

        ras = copy.copy(self)
        try:
            ras.im = self.im * im
        except TypeError:
            pass
        try:
            ras.y = ras.y * y
        except TypeError:
            pass
        return ras

    def __truediv__(self, other):
        if isinstance(other, SimpleRaster):
            im = other.im
            y = other.y
        else:
            im = other
            y = other

        ras = copy.copy(self)
        try:
            ras.im = self.im / im
        except TypeError:
            pass
        try:
            ras.y = ras.y / y
        except TypeError:
            pass
        return ras


def extract_values_from_line(line, im, meta, no_nan=True):
    indices, points = indices_along_line(line, im.shape, meta)
    f = im[indices]
    r = np.array(cumulative_distances(*points))

    if no_nan:
        r = r[~np.isnan(f)]
        f = f[~np.isnan(f)]

    return np.array(r), np.array(f)


def plot_trace(trace, mesh, color=None, ax=None,
               mean=True, std1=True, std2=True, n=100, smooth=True, return_patches=False, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    if n > len(mesh):
        r = np.linspace(mesh[0], mesh[-1], n)
    else:
        r = mesh

    mean_f, std_f, low_f, up_f = trace_to_splines(trace, mesh, smooth)

    if 'label' in kwargs:
        kwargs.pop('label')
        return_patches = True

    z0 = 0
    if 'zorder' in kwargs:
        z0 = kwargs.pop('zorder')

    if mean:
        ax.plot(r, mean_f(r), zorder=z0 + 4, c=color,
                path_effects=[patheffects.Stroke(linewidth=0, foreground='black'), patheffects.Normal()],
                label='Mean',
                **kwargs)
    if std1:
        ax.fill_between(r, mean_f(r) - std_f(r), mean_f(r) + std_f(r),
                        alpha=0.6, zorder= z0 + 3, facecolor=color,
                        label='$\sigma$',
                        **kwargs)
    if std2:
        ax.fill_between(r, mean_f(r) - 2*std_f(r), mean_f(r) + 2*std_f(r), alpha=0.3, zorder=z0 + 2, facecolor=color,
                        label='$2\sigma$',
                        **kwargs)

    if return_patches:
        handle = ax.collections[-(std1 + std2):] + ax.lines[-1:]
        return ax, handle
    else:
        return ax


def trace_to_splines(trace, mesh, smooth=True):
    axis = tuple([i for i in range(len(trace.shape) - 1)])  # stack all the way to the last axis
    up = np.quantile(trace, 0.975, axis=axis)
    low = np.quantile(trace, 0.025, axis=axis)
    std = np.std(trace, axis=axis)
    mean = np.mean(trace, axis=axis)
    ii = ~np.isnan(mean)

    kind = 'linear'
    if smooth:
        kind = 'cubic'

    up_f = interp1d(mesh[ii], up[ii], kind=kind)
    low_f = interp1d(mesh[ii], low[ii], kind=kind)
    std_f = interp1d(mesh[ii], std[ii], kind=kind)
    mean_f = interp1d(mesh[ii], mean[ii], kind=kind)

    return mean_f, std_f, low_f, up_f

# old func?
def to_target_simpleraster(in_raster, target_raster):
    return SimpleRaster(*resize_ras_to_target(in_raster, target_raster.meta))


def raster_to_simpleraster(raster):
    im = raster.read(1)
    meta = raster.meta.copy()
    return SimpleRaster(im, meta)


def box_from_meta(meta):
    extent = rasterio.transform.array_bounds(meta['height'], meta['width'], meta['transform'])
    bbox = geom.box(*extent)
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])
    return gdf


def resample_im_to_target(im, input_meta, target_meta):
    src_bounds = rasterio.transform.array_bounds(input_meta['height'], input_meta['width'],
                                                 input_meta['transform'])
    target_dx, target_dy = target_meta['transform'][0], target_meta['transform'][4]
    dst_t, dst_w, dst_h = rasterio.warp.calculate_default_transform(input_meta['crs'], input_meta['crs'],
                                                                    im.shape[-2], im.shape[-1],
                                                                    *src_bounds, resolution=(target_dx, -target_dy))

    out_im = np.zeros((dst_h, dst_w))
    out_im, out_transform = rasterio.warp.reproject(im, out_im, src_transform=input_meta['transform'],
                                                    src_crs=input_meta['crs'], dst_transform=dst_t,
                                                    dst_crs=target_meta['crs'], dst_nodata=np.nan)
    out_meta = input_meta.copy()
    out_meta.update({'height': out_im.shape[-2],
                     'width': out_im.shape[-1],
                     'transform': out_transform,
                     'crs': target_meta['crs']})

    return out_im, out_meta


def crop_im_to_target_meta(im, input_meta, target_im, target_meta):
    inbox_gdf = box_from_meta(input_meta)
    outbox_gdf = box_from_meta(target_meta)
    box_gdf = outbox_gdf.intersection(inbox_gdf)
    indices = np.where(~geometry_mask(box_gdf.geometry, im.shape, input_meta['transform']))

    cr_im = im[np.min(indices[0]):np.max(indices[0]) + 1, np.min(indices[1]): np.max(indices[1] + 1)]
    cr_im = cr_im.astype('float')
    cr_im[cr_im == input_meta['nodata']] = np.nan

    cr_transform, cr_w, cr_h = rasterio.warp.calculate_default_transform(target_meta['crs'],
                                                                         target_meta['crs'],
                                                                         im.shape[-1], im.shape[-2],
                                                                         *box_gdf.bounds.values[0],
                                                                         dst_width=cr_im.shape[-1],
                                                                         dst_height=cr_im.shape[-2])

    dst_transform, out_w, out_h = rasterio.warp.calculate_default_transform(target_meta['crs'],
                                                                            target_meta['crs'],
                                                                            cr_im.shape[-1], cr_im.shape[-2],
                                                                            *outbox_gdf.bounds.values[0])#, dst_width=target_meta['width'], dst_height=target_meta['height'])

    out_im = np.zeros(shape=(out_h, out_w))

    out_im, out_transform = rasterio.warp.reproject(cr_im, out_im,
                                                    src_crs=target_meta['crs'],
                                                    dst_crs=target_meta['crs'],
                                                    src_transform=cr_transform,
                                                    dst_transform=dst_transform,
                                                    dst_nodata=np.nan)
    out_meta = input_meta.copy()
    out_meta.update({'height': out_h,
                     'width': out_w,
                     'transform': out_transform})  # transform is wrong here!!

    return out_im, out_meta

def reproject_simpleraster(sr, target_crs):

    from rasterio import warp

    transform, width, height = warp.calculate_default_transform(sr.meta['crs'], target_crs,
                               *sr.im.shape, *[sr.extent[i] for i in [0, 2, 1, 3]])

    out = np.zeros((height, width))
    out, out_transform = reproject(sr.im, out,
              src_transform=sr.meta['transform'], src_crs=sr.meta['crs'],
              dst_transform=None, dst_crs=target_crs)

    return out, out_transform


def resize_simpleraster_to_target(orig_sr, target_sr, resample=True):
    # could be subdivided further if need be

    if orig_sr.meta['crs'] != target_sr.meta['crs']:
        out, out_transform = reproject_simpleraster(orig_sr, target_crs=target_sr.meta['crs'])
        orig_sr.im = out
        orig_sr.meta['height'] = out.shape[0]
        orig_sr.meta['width'] = out.shape[1]
        orig_sr.meta['transform'] = out_transform
        orig_sr.meta['crs'] = target_sr.meta['crs']

    if resample:
        out_im, out_meta = resample_im_to_target(orig_sr.im, orig_sr.meta, target_sr.meta)
        orig_sr.im = out_im
        orig_sr.meta = out_meta

    out_im, out_meta = crop_im_to_target_meta(orig_sr.im, orig_sr.meta, target_sr.im, target_sr.meta)

    return SimpleRaster(out_im, out_meta, tag=orig_sr.tag, name=orig_sr.name)


def path_to_simpleraster(path, tag=None, name=None, target_sr=None, resample=True):
    if path is not None:
        if isinstance(path, str):
            with rasterio.open(path) as out:
                ras = raster_to_simpleraster(out)
        elif isinstance(path, SimpleRaster):
            ras = copy.deepcopy(path)
        else:
            raise ValueError('path variable needs to be either a string or a SimpleRaster object')

        #if target_sr is not None and ras.im is not None:
        #    ras = resize_simpleraster_to_target(ras, target_sr, resample=resample)

    else:
        im = np.ones_like(target_sr.im)
        im[im == 1] = np.nan
        ras = SimpleRaster(im, target_sr.meta.copy())
    ras.tag = tag
    ras.name = name
    return ras

def indices_along_line(linestring, im_shape, meta, transform=None):

    # Gets the indices from every cell a given linestring passes trough

    # A line is a vector of x and y coordinates
    # We get every x and y coordinates (points) constituting the line
    if type(linestring) == geom.MultiLineString:
        x, y = [], []
        for l in linestring:
            X, Y = l.xy
            x.append(X)
            y.append(Y)

        x = np.hstack(x)
        y = np.hstack(y)
    else:
        x, y = np.array(linestring.xy)

    x_indices, y_indices = [], []

    p1 = None
    # We extract the cells crossed by the line going through every two consecutive points
    for i in range(len(x) - 1):
        if i == 0:
            p0 = (x[i], y[i])  # the first point
        else:
            p0 = p1

        p1 = (x[i + 1], y[i + 1])

        # Gets every index the line goes through
        line = geom.LineString([p0, p1])

        if transform is not None:
            mask = geometry_mask(geometries=[line],
                                 out_shape=im_shape[-2:],
                                 transform=transform,
                                 all_touched=False,
                                 invert=True)
        else:
            mask = geometry_mask(geometries=[line],
                             out_shape=im_shape[-2:],
                             transform=meta['transform'],
                             all_touched=False,
                             invert=True)

        indices = np.array(np.where(mask)).transpose()

        # Computes the numbers of x and y cells traversed by the line
        if transform is None:
            y_index, x_index = get_index_from_points([p1, p0], meta=meta)
        else:
            y_index, x_index = get_index_from_points([p1, p0], meta=None, transform=transform)

        dy, dx = y_index[-1] - y_index[0], x_index[-1] - x_index[0]

        # Checks whether the line is horizontal or vertical
        last, first = 0, 1
        if np.abs(dy) < np.abs(dx):
            last, first = first, last

        # Sorts the array accordingly
        if dy > 0:
            if dx >= 0:
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]
            else:
                indices[:, 1] = -indices[:, 1]
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]
        else:
            indices[:, 0] = -indices[:, 0]
            if dx >= 0:
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]

            else:
                indices[:, 1] = -indices[:, 1]
                indices = indices[np.lexsort([indices[:, last], indices[:, first]])]

        # Makes sure the indices are positive, sometimes we need to sort descending
        indices = np.abs(indices)[::-1]

        ys, xs = indices[1:, 0], indices[1:, 1]
        y_indices.extend(ys)
        x_indices.extend(xs)

    indices = (y_indices, x_indices)
    points = get_points_from_index(indices, meta, transform=transform)

    # points = [linestring.interpolate(linestring.project(point)) for point in points]

    if len(im_shape) > 2:
        indices = np.array([np.zeros_like(indices[0]), *indices])

    return tuple(indices), points

def get_index_from_points(point_coordinates, meta, transform=None):

    if transform is None:
        transform = meta['transform']

    x0, y0, = transform[2], transform[5]
    dx, dy = transform[0], transform[4]

    x, y = np.array(point_coordinates).transpose()
    x_index, y_index = np.floor((x - x0) / dx).astype('int'), np.floor((y - y0) / dy).astype('int')

    return y_index, x_index

def get_points_from_index(yx_index, meta, transform=None):
    import salem

    if transform is None:
        transform = meta['transform']

    d = 0
    # Gets the important information from the metadata and gives them an actual meaning
    if type(meta) == salem.Grid:
        dx, dy = meta.dx, meta.dy
        x0, y0 = meta.x0, meta.y0
    else:
        x0, y0, = transform[2], transform[5]
        dx, dy = transform[0], transform[4]

    if len(yx_index) > 2:
        yx_index = yx_index[1:]
    y, x = np.array(yx_index)

    x = (x + 0.5) * dx + x0
    y = (y + 0.5) * dy + y0
    points = (x, y)

    return points

def cumulative_distances(x, y):
    return np.array(
        [0, *np.cumsum(np.sqrt([np.sum((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) for i in range(1, len(x))]))])

def resize_ras_to_target(orig_ras, meta, output_ras=None):
    cropped_im, cropped_meta = crop_raster_to_target(orig_ras, meta)

    transform, width, height = calculate_default_transform(orig_ras.crs, meta['crs'],
                                                           orig_ras.width, orig_ras.height, *orig_ras.bounds,
                                                           dst_width=meta['width'], dst_height=meta['height'])
    """meta.update({
        'transform': transform,
        'width': width,
        'height': height
    })"""

    out_im = np.zeros(shape=(height, width)).astype(meta['dtype'])
    out_im[out_im == 0] = np.NaN

    out_im, t = reproject(cropped_im,
                          destination=out_im,
                          src_transform=cropped_meta['transform'],
                          dst_transform=meta['transform'],
                          src_crs=orig_ras.crs,
                          dst_crs=meta['crs'],
                          resampling=rasterio.warp.Resampling.nearest)

    if output_ras is not None:
        if output_ras.endswith('.tif'):
            meta['driver'] = 'GTiff'

        # Writes the new raster with the target raster's metadata
        out = rasterio.open(output_ras, 'w+', **meta)
        out.write(out_im)

    return out_im, meta

def crop_raster_to_target(orig_ras, meta, output_ras=None):
    # Gets the data from the target raster and
    # creates a box from it's extent to resize the original raster to
    out_z, out_h, out_w = meta['count'], meta['height'], meta['width']
    extent = rasterio.transform.array_bounds(out_h, out_w, meta['transform'])
    bbox = geom.box(*extent)
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=meta['crs'])

    # Crops the original raster
    cropped, t = mask(orig_ras, shapes=gdf.geometry, crop=True, nodata=np.NaN)
    cropped = cropped.astype(meta['dtype'])

    out_meta = meta.copy()
    out_meta['transform'] = t
    out_meta['width'] = cropped.shape[-1]
    out_meta['height'] = cropped.shape[-2]

    if output_ras is not None:
        with rasterio.open(output_ras, 'w+', **out_meta) as out:
            out.write(cropped)

    return cropped, out_meta
