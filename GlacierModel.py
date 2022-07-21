import numpy as np
import SimpleRaster as sr
import rasterio
import pymc as pm
from scipy.integrate import cumtrapz

# Main glacier object, used for computing epoch-related mass fluxes

class GlacierModel:
    def __init__(self, name, velocity_path, t0, t1, dem_path,
                 dh_path=None, color=None):

        """
        :param name: GlacierModel object name
        :param velocity_path: Time averaged velocity raster path, can also take in a SimpleRaster object
        :param t0: Floating number, beginning time of the covered time interval. Can simply be 0.
        :param t1: Floating number, end time of the covered time interval. Can simply be = dt = t1 - t0
        :param dem_path: Time averaged surface raster path, can also take in a SimpleRaster object
        :param dh_path: Total elevation change over the time period raster path, can take in a SimpleRaster object
        :param color: Color to be used in plotting, optional parameter. Follow matplotlib's color rules.
        """

        self.name = name
        self.color = color
        # time delta coverage for the glacier's period
        if t0 == t1:
            t0 = 0
            t1 = 1
        self.t0 = t0
        self.t1 = t1
        self.dt = t1 - t0

        # the surface velocities are the grid on which the computing is performed
        # find a clear way to have the same gridding for every timescales
        self.velocity = sr.path_to_simpleraster(velocity_path, 'velocity', 'Surface velocity $u_s$ [m$\,$year$^{-1}$]')

        # we ensure that every image has the same grid as the velocity
        # if no path is provided, a nan image of the same size is instead defined
        self.dem = sr.path_to_simpleraster(dem_path, f'{self.name}_dem', 'Surface elevation [m asl]', self.velocity)
        self.dhdt = sr.path_to_simpleraster(dh_path, f'{self.name}_dhdt',
                                         '$\\frac{\\Delta S}{\\Delta t}$ [m$\,$year$^{-1}$]', self.velocity)
        self.dhdt = self.dhdt/self.dt
        # gaussian processes that are to be the same for all GlacierModel objects
        self.width = None
        self.bdot = None
        self.bed = None

        self.slip_factor = None
        self.forward_model = None
        self.mesh = None

    def flowline_model(self, flowline):
        # extract data along given flowline for every variable
        for key in self.__dict__.keys():  # we iterate over every defined attribute of the glacier model
            ras = getattr(self, key)
            if type(ras) == sr.SimpleRaster:  # we ensure that the attribute is a variable
                ras.set_flowline_data(flowline)  # extracting the data using class function

    def check_priors(self):
        priors = [self.velocity, self.dhdt, self.dem]
        for variable in priors:
            assert type(variable.prior) == sr.GaussianProcessPrior, f'Glacier model {self.name} has an undefined ' \
                                                              f'prior for {variable.tag}'

    def set_gp_priors(self, mesh):
        self.mesh = mesh
        n = 3
        a = 1
        b = 1.25
        mu = 1
        std = 0.05

        self.slip_factor = pm.TruncatedNormal(f'{self.name}_slip_factor', mu=mu, tau=1 / std ** 2, a=a, b=b)
        #pm.Uniform(f'{self.name}_slip_factor', 1, (n + 2) / (n + 1))

        self.dem.set_gaussianprocess(mesh, positive_init=True)
        self.dhdt.set_gaussianprocess(mesh)

    def setup_model(self, observed=True):

        @pm.deterministic
        def forward_model(r=self.mesh, S=self.dem.gp, dhdt=self.dhdt.gp, s=self.slip_factor,
                          bdot=self.bdot.gp, w=self.width.gp, B=self.bed.gp):

            f = w.f_eval * (bdot.f_eval - dhdt.f_eval)
            u_m = cumtrapz(f, r, initial=0)
            return s * u_m / (w.f_eval * (S.f_eval - B.f_eval))


        @pm.potential
        def negative_thickness(S=self.dem.gp, B=self.bed.gp):
            min_h = 0
            if np.sum([S.f_eval - B.f_eval < min_h]) == 0:
                return 0
            else:
                return -np.inf

        self.forward_model = forward_model

        @pm.potential
        def negative_velocities(u=self.forward_model):
            if np.sum(u < 0) == 0:
                return 0
            else:
                return -10

        forward_model.__name__ = f'{self.name}_{forward_model.__name__}'
        if observed:
            scale = self.velocity.prior.obs_variance.max()
            if self.velocity.normalize:
                loc = (self.forward_model - self.velocity.f(self.mesh))/np.sqrt(scale)
                self.velocity.M = lambda x: self.velocity.f(x)
                self.velocity.C = lambda x: np.sqrt(scale)*np.ones_like(x)
                scale = 1
            else:
                loc = self.forward_model

            self.velocity.gp = pm.Normal(f'{self.name}_{self.velocity.tag}',
                                         mu=loc,
                                         tau=1/scale,
                                         observed=True, value=self.velocity.f(self.mesh))
        else:
            self.velocity.gp = pm.Normal(f'{self.name}_{self.velocity.tag}',
                                         mu=self.forward_model,
                                         tau=1/self.velocity.prior.obs_variance.max())
