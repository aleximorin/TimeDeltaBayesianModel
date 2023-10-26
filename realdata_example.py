import numpy as np
import matplotlib.pyplot as plt
import TimeDeltaBayesianModel as tdbm
import SimpleRaster as sr
import geopandas as gpd
import GlacierModel as gm
from scipy.interpolate import interp1d
import copy

np.random.seed(999)

# data imports
dem1 = sr.path_to_simpleraster('data/dem_2016.tif')
dem2 = sr.path_to_simpleraster('data/dem_2018.tif')
dh = dem2 - dem1  # the SimpleRaster objects lets us easily do raster data manipulation
veloc = sr.path_to_simpleraster('data/velocity.tif')
bdot = sr.path_to_simpleraster('data/bdot.tif')

flowlines = gpd.read_file('data/centerlines_postsurge.shp').to_crs('epsg:32607')
surging_line = flowlines.geometry.iloc[0]

# We are defining the glacier model object, which will contain necessary information from one time period
post_surge = gm.GlacierModel(name='Syn-surge glacier', velocity_path=veloc, t0=2016, t1=2018,
                          dem_path=(dem1 + dem2)/2,
                          dh_path=dh,
                          color='tab:orange')

# We can define the Model object with a list of GlacierModels containing time dependant information.
model = tdbm.OneDTimeDeltaBayesianModel(name='Syn-surge only model',
                                             glaciers=[post_surge,],
                                             flowline=surging_line,
                                             bdot_path=bdot)


# defining the mesh on which we will sample the posterior
dx = 500
glacier = model.glaciers['Syn-surge glacier']  # we don't want the mesh to be longer than the velocity observations
mesh = glacier.velocity.r
mesh = np.arange(0, mesh[-1], dx)

# We enforce the bed elevation to be equal to the surface elevation at given points
ii = (mesh >= 17000) | (mesh == mesh[0]) | (mesh == mesh[-1])
model.bed.r = mesh[ii]
model.bed.y = glacier.dem.f(model.bed.r)

# a priori mean function for the bed
surf = model._get_lowest_surface(mesh)
dem_f = interp1d(mesh, surf)
model.bed.dem_prior = copy.deepcopy(dem_f)

# we give an unconstrained prior for the width
model.width.r = mesh
model.width.y = 0.5 * np.ones_like(mesh)

# we define the priors for the various gaussian processes, here unrelated to time
model.bed.set_1Dprior(sigma=250,
                      lengthscale=500,
                      mean_function='self.dem_prior',  # mean functions can be evaluated from the eval() python function
                      obs_variance=50)
model.bdot.set_1Dprior(sigma=10, lengthscale=1000,
                       mean_function='np.polynomial.Polynomial.fit(self.r.flatten(), self.y, deg=1)',
                       obs_variance=25)
model.width.set_1Dprior(sigma=0.1, obs_variance=0.1,
                        mean_function=lambda x: 0.5 * np.ones_like(x), lengthscale=500)

# we define the time dependant gaussian processes priors
for g in model.glaciers:
    model.glaciers[g].dem.set_1Dprior(sigma=100, lengthscale=500, mean_function='self.f',
                                      obs_variance=25)
    model.glaciers[g].dhdt.set_1Dprior(sigma=100, lengthscale=500, mean_function='self.f',
                                       obs_variance=250)
    # velocity is not defined as a gaussian process in the model, we are here only defining the likelihood's prior
    model.glaciers[g].velocity.set_1Dprior(obs_variance=100, obs_on_mesh=True)

# the model.setup_model() function instanciates every pymc object needed
model.setup_model(mesh, observed=False)

# various mcmc sampling parameters
pow = 5
niter = 10 ** pow
nburn = 10 ** (pow - 1)
nthin = 10
nchains = 3

# we can now sample the posterior for the determined number of samples
model.sample(niter, nburn, nthin, nchains)

# we can easily save and load samplers with these functions
model.save_sampler('sampler.p')
model = tdbm.load_model('sampler.p')
