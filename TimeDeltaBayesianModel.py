import matplotlib.pyplot as plt
from typing import List
import pymc as pm
import numpy as np
import time
import dill as pickle
import SimpleRaster as sr
import GlacierModel as gm
import os
from matplotlib.gridspec import GridSpec
import pandas as pd
import copy
from scipy.interpolate import CubicSpline


class OneDTimeDeltaBayesianModel:

    def __init__(self, name: str, glaciers: List[gm.GlacierModel], flowline=None,
                 width_path=None, bdot_path=None, bed_path=None):


        """
        :param name: Name of the model
        :param glaciers: List of GlacierModel object on which the inversion will occur
        :param flowline: Flowline from which the data will be extracted. To be used if the data is 2D
        :param width_path: Path of the known width raster, can be also given a SimpleRaster object
        :param bdot_path: Path of the known mass balance raster, can be also given a SimpleRaster object
        :param bed_path: Path of the known bed elevation raster, can be also given a SimpleRaster object
        """

        self.name = name
        self.glaciers = glacier_list_to_dict(glaciers)
        self.flowline = flowline
        self.dts = np.array([glacier.dt for glacier in self.glaciers.values()])

        self.mesh = None
        self.model_vars = []
        self.full_timescale = None
        self.time_delta_model = sr.SimpleRaster(None, None)
        self.observed = None

        velocity_sr = glaciers[0].velocity

        # we open the rasters and assign them to the various variables
        self.bed = sr.path_to_simpleraster(bed_path, 'bed', 'Bed elevation [m asl]', velocity_sr)
        self.width = sr.path_to_simpleraster(width_path, 'width', 'Flowline width  [m]', velocity_sr)
        self.width.cubic = True
        self.bdot = sr.path_to_simpleraster(bdot_path, 'bdot', 'Mass balance $\\dot{b}$ [m$\,$year$^{-1}$]',
                                            velocity_sr)

        # if a flowline is defined, does not happen for synthetic models
        if flowline is not None:
            self._flowline_model()

        # mcmc parameters
        self.sampler = None
        self.niter = None
        self.nburn = None
        self.nthin = None
        self.nchains = None

    def setup_model(self, mesh, observed=False):

        """
        :param mesh: The 1D mesh on which to compute the inversion on.
        :return:
        """

        self._check_priors()
        self.observed = observed
        self.mesh = mesh

        for glacier in self.glaciers.values():
            glacier.check_priors()
            glacier.set_gp_priors(mesh)

        surface = self._get_lowest_surface(self.mesh)

        self.width.set_gaussianprocess(self.mesh)
        self.bdot.set_gaussianprocess(self.mesh)
        self.bed.set_gaussianprocess(self.mesh, init_vals=surface, lowest=True)

        for glacier in self.glaciers.values():
            self._update_time_invariant_priors(glacier)
            glacier.setup_model()
            variables = [glacier.velocity.gp, glacier.dhdt.gp, glacier.dem.gp,
                         glacier.slip_factor, glacier.forward_model]
            self.model_vars += variables

        @pm.potential
        def negative_width(w=self.width.gp):
            # pymc2 functions constraining the inversion

            if np.sum([w.f_eval < 0]) == 0:
                return 0
            else:
                return -np.inf

        @pm.deterministic
        def full_timescale(w=self.width.gp):
            # pymc2 likelihood function
            u_ms = np.array([g.velocity.gp.value*g.dt for g in self.glaciers.values()]).T
            return u_ms / self.dts.sum()

        self.full_timescale = full_timescale
        self._set_time_delta_model()
        if observed:
            self.time_delta_model.gp = pm.Normal(f'time_delta_model',
                                              mu=self.full_timescale,
                                              tau=1 / 100,
                                              observed=True, value=self.time_delta_model.y)

        else:
            self.time_delta_model.gp = pm.Normal(f'time_delta_model',
                                              mu=self.full_timescale,
                                              tau=1/100)

        self.model_vars += [self.time_delta_model.gp, self.full_timescale, self.bed.gp, self.width.gp, self.bdot.gp]

    def MAP(self):

        # BROKEN
        self.map = pm.MAP(self.model_vars)
        new_stochastics = []
        for sto in self.map.stochastics:
            if type(sto.value) == np.ndarray:
                new_stochastics.append(sto)
        self.map.stochastics = set(new_stochastics)
        self.map.fit('fmin')

        stochastics = dict()
        for elem in self.map.generations[0]:
            stochastics[elem.__name__] = elem.value

        variables = ['dhdt', 'dem']
        for key in self.glaciers.keys():
            glacier = self.glaciers[key]
            for v in variables:
                map = stochastics[f'{glacier.name}_{v}_f_eval']
                v = glacier.__getattribute__(v)
                v.map = map

            glacier.slip_factor_trace = stochastics[glacier.slip_factor.__name__]

        self.bed.map = stochastics['bed_f_eval']
        self.width.map = stochastics['width_f_eval']
        self.bdot.map = stochastics['bdot_f_eval']
        self.map = None

    def sample(self, niter, nburn, nthin, nchains):

        """
        :param niter: Number of iterations
        :param nburn: Burn-in number, i.e. number of first iterations rejected in the posterior
        :param nthin: Thinning factor k, iterations are rejected once every k iterations
        :param nchains: Number of chains
        :return:
        """
        # main mcmc wrapper
        if len(self.model_vars) == 0:
            raise AttributeError('Must have set up the model first')

        self.niter = niter
        self.nburn = nburn
        self.nthin = nthin
        self.nchains = nchains

        # we use the default sampling method from pymc2
        self.sampler = pm.MCMC(self.model_vars, reinit_model=True)
        self.sampler.assign_step_methods()

        times = np.zeros((nchains))

        start_time = time.time()
        for i in range(self.nchains):
            st = time.time()
            print(f'Sampling chain {i + 1} for {niter} iterations')
            self.sampler.sample(self.niter, self.nburn, self.nthin,
                                tune_interval=250, stop_tuning_after=10,
                                progress_bar=True)
            edt = time.time() - st
            print(f'Sampling chain {i + 1} with {niter} iterations finished in {edt / 60:.2f} minutes')
            times[i] = edt

        end_time = time.time() - start_time
        print(f'Sampling finished in {end_time / 60:.2f} minutes: {nchains} chains, {niter} iterations, '
              f'{nburn} burn-in and {nthin} thinning factor')
        self._parse_trace()
        return times

    def plot_model(self, savepath=None, showfig=False, smooth=True):

        # wrapper function to plot the posterior

        fig, axs = plt.subplots(nrows=6, sharex='col', tight_layout=True, figsize=(5, 10))

        for g in self.glaciers.values():
            # Time dependant plots
            g.velocity.plot1D(ax=axs[0], c=g.color, smooth=smooth)
            g.dhdt.plot1D(ax=axs[1], c=g.color, smooth=smooth)
            self._thickness_traceplot(g, axs[2], color=g.color, smooth=smooth)
            g.dem.plot1D(ax=axs[-1], c=g.color, smooth=smooth)

        if self.observed:
            self.time_delta_model.plot1D(ax=axs[0], c='red', smooth=smooth, show_prior=False)

        # Time independant plots
        self.bdot.plot1D(ax=axs[3], c='purple', smooth=smooth)
        self.width.plot1D(ax=axs[4], c='green', smooth=smooth)
        self.bed.plot1D(ax=axs[-1], c='grey', smooth=smooth)

        # Time dependant plots
        axs[0].set_title('Surface velocity')
        axs[0].set_ylabel('[m year$^{-1}$]')
        axs[1].set_title('Surface change over time')
        axs[1].set_ylabel('[m year$^{-1}$]')
        axs[2].set_title('Ice thickness')
        axs[2].set_ylabel('[m]')

        # Time independant plots
        axs[3].set_title('Mass  balance')
        axs[3].set_ylabel('[m year$^{-1}$]')
        axs[4].set_title('Flowline width')
        axs[4].set_ylabel('[m]')
        axs[-1].set_title('Surface and bed elevation')
        axs[-1].set_ylabel('[m asl]')
        axs[-1].set_xlabel('Distance from glacier head [m]')

        # Making the plot pretty
        axs[-1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.align_ylabels()

        if savepath is not None:
            fig.savefig(savepath)
            path = os.path.abspath(savepath).replace("\\", "/")
            print(f'Model plot at file:///{path}')

        if showfig:
            plt.show()
        return fig, axs

    def plot_small_model(self, savepath=None, showfig=False):

        # simpler wrapper function to plot the posterior

        fig, axs = plt.subplots(nrows=3, sharex='col', tight_layout=True, figsize=(7, 7))
        for g in self.glaciers.values():
            # Time dependant plots
            g.velocity.plot1D(ax=axs[0], c=g.color, show_prior=False)
            g.dhdt.plot1D(ax=axs[1], c=g.color, show_prior=False)
            g.dem.plot1D(ax=axs[-1], c=g.color, show_prior=False)

        # Time independant plots
        self.bed.plot1D(ax=axs[-1], c='grey', show_prior=False)
        # Time dependant plots
        axs[0].set_title('Surface velocity')
        axs[0].set_ylabel('[m year$^{-1}$]')
        axs[1].set_title('Surface change over time')
        axs[1].set_ylabel('[m year$^{-1}$]')
        axs[2].set_title('Ice thickness')
        axs[2].set_ylabel('[m]')
        # Time independant plots
        axs[-1].set_title('Surface and bed elevation')
        axs[-1].set_ylabel('[m asl]')
        axs[-1].set_xlabel('Distance from glacier head [m]')

        # Making the plot pretty
        [ax.grid() for ax in axs.flatten()]
        axs[-1].xaxis.set_major_locator(plt.MaxNLocator(6))
        fig.align_ylabels()
        try:
            if len(self.glaciers) != 1:
                handles = tuple([(axs[0].collections[i * 3], axs[0].lines[i], axs[0].collections[3 * i + 2])[::-1] for i in
                                 range(len(self.glaciers))])
                labels = (g.name for g in self.glaciers.values())
                fig.legend(handles, labels, framealpha=1)
        except Exception as e:
            print(e)
        if savepath is not None:
            fig.savefig(savepath)
            path = os.path.abspath(savepath).replace("\\", "/")
            print(f'Small model plot at file:///{path}')
        if showfig:
            plt.show()
        return fig, axs

    def traceplot(self, savepath=None, showfig=False, bounded=True, histogram=True, gr_lim=None, figsize=None):

        # wrapper function plotting the trace of mcmc chains.
        # WARNING, VERY LAGGY PLOT

        import warnings

        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

        if figsize is None:
            figsize = (10, 8)
            if histogram:
                figsize = (2, 5)

        fig = plt.figure(tight_layout=True, figsize=figsize)
        # fig.subplots_adjust(hspace=0.03)
        grid = GridSpec(6, len(self.glaciers), figure=fig)
        # fig, axs = plt.subplots(nrows=6, ncols=len(self.glaciers), sharex='col', tight_layout=True, figsize=(5, 10))
        for i, g in enumerate(self.glaciers.values()):
            v_ax = g.velocity.traceplot(ax=fig.add_subplot(grid[0, i]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)
            dhdt_ax = g.dhdt.traceplot(ax=fig.add_subplot(grid[1, i]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)
            dem_ax = g.dem.traceplot(ax=fig.add_subplot(grid[2, i]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)
            v_ax.set_title(g.name)

            if not histogram:
                v_ax.set_xticks([])
                dhdt_ax.set_xticks([])
                if i != 0:
                    for ax in [v_ax, dhdt_ax, dem_ax]:
                        ax.set_yticks([])
            else:
                v_ax.set_ylabel('Surface velocity')
                dhdt_ax.set_ylabel('Surface change')
                dem_ax.set_ylabel('Surface elevation')

        bdot_ax = self.bdot.traceplot(ax=fig.add_subplot(grid[3, :]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)
        width_ax = self.width.traceplot(ax=fig.add_subplot(grid[4, :]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)
        bed_ax = self.bed.traceplot(ax=fig.add_subplot(grid[5, :]), bounded=bounded, histogram=histogram, gr_lim=gr_lim)

        bdot_ax.set_title('Time independant variables')
        bdot_ax.set_ylabel('Mass balance')
        width_ax.set_ylabel('Flowline width')
        bed_ax.set_ylabel('Bed elevation')

        if savepath is not None:
            fig.savefig(savepath)
            path = os.path.abspath(savepath).replace("\\", "/")
            print(f'Trace plot saved at file:///{path}')

        if showfig:
            plt.show()

    def plot_pdf(self, savepath=None, showfig=False, smooth=True):

        # wrapper function for plotting

        # main plotting function for every axis
        def plot(ax, gp, color, interval=1, zorder=0, label='', plot_obs=True):
            _, handles = sr.plot_trace(gp.trace, self.mesh, color=color, ax=ax, zorder=zorder, return_patches=True)
            obs = None
            if plot_obs:
                ii = (gp.r > gp.mesh[0]) & (gp.r < gp.mesh[-1])
                obs = ax.errorbar(gp.r[ii][::interval], gp.y[ii][::interval],
                                  yerr=np.sqrt(gp.prior.obs_variance[:len(gp.r)][ii][::interval]).flatten(),
                                  ecolor='k', zorder=30, ls='none', alpha=1, capsize=4, lw=1)
            return tuple(handles), obs

        color_dict = {'Pre-surge': 'tab:blue',
                      'Syn-surge': 'tab:orange',
                      'Full-timescale': 'light gray',
                      'Long-timescale': 'tab:green'}

        # preparing the figure's axis labels
        fig, axs = plt.subplots(ncols=2, nrows=3, tight_layout=True, figsize=(12, 9))
        axs = axs.flatten()
        axs[0].set_ylabel('Surface velocity (m$\,$a$^{-1}$)')
        axs[1].set_ylabel('$\Delta S/\Delta t$ (m$\,$a$^{-1}$)')
        axs[2].set_ylabel('Ice thickness (m)')
        axs[3].set_ylabel('Flowline width')
        axs[4].set_ylabel('Mass balance (m$\,$a$^{-1}$)')
        axs[5].set_ylabel('Elevation (m asl.)')
        # iterating through the glacier eras
        i = 15
        handles = []
        labels = []
        for g in self.glaciers.values():
            h, o = plot(axs[0], g.velocity, g.color, interval=5, zorder=i, label=g.name)
            plot(axs[1], g.dhdt, g.color, interval=5, zorder=i, label=g.name)
            plot(axs[-1], g.dem, g.color, interval=5, zorder=i, label=g.name)
            self._thickness_traceplot(g, axs[2], color=g.color, zorder=i)
            handles.append(h)
            handles.append(o)
            labels += [f'{g.name} posterior', f'{g.name} input']
            i -= 5
        # manipulating the labels for the legend
        for ax in axs[:2]:
            ax.legend(handles, labels, framealpha=1, ncol=2, loc='upper left')
        axs[2].legend(handles, labels)
        # manually defining the legend's handles and labels is necessary
        h, o = plot(axs[3], self.width, 'green', interval=10, plot_obs=False)
        axs[3].legend([h, o], ['Posterior', 'Observations'], framealpha=1, loc='upper left')
        h, o = plot(axs[4], self.bdot, 'purple', interval=50)
        axs[4].legend([h, o], ['Posterior', 'Input'], framealpha=1, loc='upper left')
        # the bed's axis has both time state (surface) and the bed
        hbed, obed = plot(axs[-1], self.bed, 'gray', interval=2)
        handles.append(hbed)
        handles.append(obed)
        labels.append('Bed posterior')
        labels.append('Bed input')
        axs[-1].legend(handles, labels, framealpha=1, ncol=2, loc='upper right')
        # tweaking common to every axis
        for i, ax in enumerate(axs):
            k = 1 if i == 3 else 0

            ax.grid()
            # ax.minorticks_on()
            ax.grid(which='major', c='gray')
            # ax.grid(which='minor', c='gray', alpha=0.3)
            ax.set_xlabel('Distance from divide (km)')
            #maxx = 20000 if self.name != 'Pre-surge' else 17000
            #ax.set_xlim(0, maxx)
            xlim = ax.get_xlim()
            xticks = ax.get_xticks()
            xticks = xticks[(xlim[0] <= xticks) & (xticks <= xlim[1])]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{t:.0f}' for t in xticks / 1000])

            ylim = ax.get_ylim()
            yticks = ax.get_yticks()
            yticks = yticks[(ylim[0] <= yticks) & (yticks <= ylim[1])]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{t:.{k}f}' for t in yticks / 1])
            ax.set_yticks(yticks)

        if savepath is not None:
            fig.savefig(savepath)
            path = os.path.abspath(savepath).replace("\\", "/")
            print(f'PDF plot saved at file:///{path}')

        if showfig:
            plt.show()
        return fig, axs

    def save_sampler(self, savepath):

        # wrapper to clean and dump the model in a pickled file
        self._clean_pymc()
        savepath = savepath
        with open(savepath, 'wb') as f:
            pickle.dump(self, f)
        self._reassign_gps()

    def print_stats(self):

        # printing convergence statistics for every variable in the model
        # Gelman-Rubin statistic computed via the default pymc2 function

        print(f'{self.name} convergence statistics')
        for variable in [self.bed, self.width, self.bdot]:
            gr = pm.gelman_rubin(variable.trace)
            print(f'{variable.tag} gelman rubin: Max = {np.max(gr):.2f}, Min = {np.min(gr):.2f}, '
                  f'Mean = {np.mean(gr):.2f}, Std = {np.std(gr):.2f}')

        for g in self.glaciers.values():
            for variable in [g.dem, g.dhdt, g.velocity]:
                print(
                    f'{variable.tag} gelman rubin: Max = {np.max(gr):.2f}, Min = {np.min(gr):.2f}, '
                    f'Mean = {np.mean(gr):.2f}, Std = {np.std(gr):.2f}')

        print()

    def _update_time_invariant_priors(self, glacier):
        glacier.bed = self.bed
        glacier.width = self.width
        glacier.bdot = self.bdot

    def _clean_pymc(self):
        # drop everything related to pymc2 in order to pickle the TimeDeltaModel object
        variables = ['dhdt', 'dem']
        for key in self.glaciers.keys():
            glacier = self.glaciers[key]
            for v in variables:
                v = glacier.__getattribute__(v)
                v.gp = None

            glacier.velocity.gp = None
            glacier.forward_model = None
            glacier.slip_factor = None

        self.bed.gp = None
        self.width.gp = None
        self.bdot.gp = None
        self.full_timescale = None
        self.time_delta_model.gp = None
        self.model_vars = []
        self.sampler = None

    def _parse_trace(self):
        # parse the trace to a numpy array
        variables = ['dhdt', 'dem']
        for key in self.glaciers.keys():
            glacier = self.glaciers[key]
            for v in variables:
                ras = glacier.__getattribute__(v)
                trace = self._get_trace(f'{glacier.name}_{v}_f_eval', ras)
                ras.trace = trace

            glacier.velocity.trace = self._get_trace(glacier.forward_model.__name__, glacier.velocity)
            glacier.velocity.mesh = self.mesh
            glacier.slip_factor_trace = self._get_trace(glacier.slip_factor.__name__)

        self.bed.trace = self._get_trace('bed_f_eval', self.bed)
        self.width.trace = self._get_trace('width_f_eval', self.width)
        self.bdot.trace = self._get_trace('bdot_f_eval', self.bdot)
        if not self.observed:
            self.time_delta_model.trace = self._get_trace(self.time_delta_model.gp.__name__)
        else:
            self.time_delta_model.trace = self._get_trace(self.full_timescale.__name__)

    def _get_trace(self, name, ras=None):
        trace = np.array([self.sampler.db.trace(name, chain=i)[:] for i in range(self.nchains)])
        if ras is not None:
            if ras.normalize:
                trace = trace*np.sqrt(ras.C(self.mesh)) + ras.M(self.mesh)
        return trace

    def _flowline_model(self):
        self.bed.set_flowline_data(self.flowline)
        self.bdot.set_flowline_data(self.flowline)
        self.width.set_flowline_data(self.flowline)
        for glacier in self.glaciers.values():
            glacier.flowline_model(self.flowline)

    def _check_priors(self):
        priors = [self.bed, self.width, self.bdot]
        for variable in priors:
            assert type(variable.prior) == sr.GaussianProcessPrior, f'TimeDeltaModel {self.name} has an undefined ' \
                                                                    f'prior for {variable.tag}'
    def _set_time_delta_model(self):
        u = np.array([g.velocity.f((self.mesh)) for g in self.glaciers.values()]).T
        u = u.dot(self.dts) / self.dts.sum()
        self.time_delta_model = sr.SimpleRaster(None, None)
        self.time_delta_model.r = self.mesh
        self.time_delta_model.mesh = self.mesh
        self.time_delta_model.y = u
        self.time_delta_model.f = CubicSpline(self.mesh, u)
        self.time_delta_model.tag = 'time_delta_model'

    def _get_mean_surface(self):
        surface = 0
        for g in self.glaciers.values():
            surface += g.dem.f(self.mesh)
        return surface / len(self.glaciers)

    def _get_lowest_surface(self, mesh):
        try:
            surfs = np.vstack([g.dem.gp.f_eval.value for g in self.glaciers.values()])
        except AttributeError:
            surfs = np.vstack([g.dem.f(mesh) for g in self.glaciers.values()])
        return np.nanmin(surfs, axis=0) - 1

    def _reassign_gps(self):
        for variable in self.__dict__.values():
            if isinstance(variable, sr.SimpleRaster):
                if variable.tag == 'time_delta_model':
                    continue
                try:
                    variable.set_gaussianprocess(self.mesh)
                except NameError:
                    pass

        for glacier in self.glaciers.values():
            for variable in glacier.__dict__.values():
                if isinstance(variable, sr.SimpleRaster):
                    if variable.tag == 'velocity':
                        continue
                    try:
                        variable.set_gaussianprocess(self.mesh)
                    except NameError:
                        pass

    def _thickness_traceplot(self, g, ax=None, color=None, smooth=True, **kwargs):
        try:
            trace = g.dem.trace - self.bed.trace
            ras = sr.SimpleRaster(None, None, tag='time_delta')
            ras.mesh = self.mesh
            ras.trace = trace
            #ras.gp = pm.gp.GPSubmodel(self.tag, self.bed.gp.M + g.dem.gp.M, self.bed.gp.C + g.dem.gp.C, self.mesh)
            ras.plot1D(ax=ax, c=color, smooth=smooth, **kwargs)
        except Exception as e:
            print(e)


def load_folder(model_folder, get_param_func=None, constraints=''):
    df = pd.DataFrame()
    for file in os.listdir(model_folder):
        if file.endswith('.p'):
            if constraints not in file:
                continue
            fullpath = os.path.join(model_folder, file)
            try:
                model = load_model(fullpath)
            except (pickle.UnpicklingError, EOFError) as e:
                print(e)
                continue
            if get_param_func is not None:
                params = get_param_func(model)
                for k, v in params.items():
                    try:
                        df.loc[model, k] = v
                    except ValueError:
                        pass
            else:
                df = df.reindex(df.index.values.tolist() + [model])
            df.loc[model, 'path'] = fullpath
            del model
    return df


def load_model(model_path) -> OneDTimeDeltaBayesianModel:
    with open(model_path, 'rb') as file:
            model = pickle.load(file)
    try:
        model._reassign_gps()
    except NameError as e:
        print(f'Couldn\'t load the gaussian processes properly for {model.name}. Watch out for mean functions.')
        print(e)
        print()
    except AttributeError:
        model.observed = False

    return model


def glacier_list_to_dict(glaciers):
    glacier_dict = dict()
    for glacier in glaciers:
        glacier_dict[glacier.name] = copy.deepcopy(glacier)

    return glacier_dict


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.rcParams.update({'figure.autolayout': False})
