import numpy as np
import TimeDeltaBayesianModel as tdbm
import SimpleRaster as sr
import GlacierModel as gm
from scipy.interpolate import CubicSpline
import xarray as xr


def sr_from_xy(x, y):
    ras = sr.SimpleRaster(None, None)
    ras.r = x
    ras.y = y
    try:
        ras.f = CubicSpline(x, y, extrapolate=False)
    except ValueError:
        pass
    return ras


def time_avg(y, t, ii=None):
    if len(t) != 1:
        integral = np.trapz(y, t)/(t[-1] - t[0])
    else:
        integral = y.reshape(-1)
    if ii is None:
        ii = np.ones(len(integral)).astype(bool)

    integral[~ii] = 0
    return integral


def netcdf_to_glaciermodel(path, name, return_model=True,
                           start_time_i=0, end_time_i=-1, arbitrary_length_treshold=-1,
                           deltaT=None, final=None, noise=0, dt_steady=0):

    # very hacky function to be able to use Andrew Nolan's netcdf files into the model
    data = xr.open_dataset(path)

    if arbitrary_length_treshold == -1:
        arbitrary_length_treshold = len(data.x)

    r = data.x.values[:arbitrary_length_treshold]
    t = data.t.values[start_time_i:end_time_i]
    bed = data.z_b.values[::-1, :1][:arbitrary_length_treshold]
    try:
        u = data.v_s.values[::-1, start_time_i:end_time_i][:arbitrary_length_treshold]
    except AttributeError:
        u = data.v_m.values[::-1, start_time_i:end_time_i][:arbitrary_length_treshold]
    z_s = data.z_s.values[::-1, start_time_i:end_time_i][:arbitrary_length_treshold]
    b_dot = data.b_dot.values[::-1, start_time_i:end_time_i][:arbitrary_length_treshold]
    h = z_s - bed

    ice_thickness_threshold = 0
    ii = np.zeros(len(r)).astype(bool)
    h0 = h[:, 0] <= ice_thickness_threshold
    h1 = h[:, -1] <= ice_thickness_threshold

    if final is None:
        try:
            final = np.where(h1[10:])[0][0] + 11
        except IndexError:
            final = -1
    ii[:final] = True

    if dt_steady:
        dt_steady = 10
        dt = t[1] - t[0]
        dt_ii = int(dt_steady / dt)

        t = np.arange(len(t) + dt_ii) * dt

        ones = np.ones((len(r), dt_ii))

        u_pre = ones * u[:, :1]
        u = np.hstack((u_pre, u))

        s_pre = ones * z_s[:, :1]
        z_s = np.hstack((s_pre, z_s))

        h_pre = ones * h[:, :1]
        h = np.hstack((h_pre, h))

        b_dot_pre = ones*b_dot[:, :1]
        b_dot = np.hstack((b_dot_pre, b_dot))

    dhdt = z_s[:, -1] - z_s[:, 0]

    ustd = noise * np.abs(u).mean()
    unoise = ustd * np.random.randn(len(u))
    jj = (u.mean(axis=1) + unoise) < 0

    dhdtstd = noise * np.abs(dhdt).mean()
    dhdtnoise = dhdtstd * np.random.randn(dhdt.size).reshape(dhdt.shape)

    u += unoise[:, None]
    dhdt += dhdtnoise

    r = r[ii]

    U = sr_from_xy(r, time_avg(u[ii], t))
    S = sr_from_xy(r, time_avg(z_s[ii], t))
    H = sr_from_xy(r, time_avg(h[ii], t))
    BDOT = sr_from_xy(r, time_avg(b_dot[ii], t))
    DHDT = sr_from_xy(r, dhdt[ii])
    WIDTH = sr_from_xy(r, np.ones_like(U.y))

    DHDT.noise = dhdtstd
    U.noise = ustd

    cond = h0[:final] | h1[:final]

    try:
        where = np.where(cond)[0]
        cond = np.hstack((where[0], np.arange(where[1], where[-1]+1)))
    except IndexError:
        cond[0] = cond[-1] = True

    BED = sr_from_xy(r[cond], bed[:final][cond])
    BED.f = None

    hii = (h[:, 0] <= ice_thickness_threshold) | (h[:, -1] <= ice_thickness_threshold)

    if deltaT is None:
        t0 = t[0]
        t1 = t[-1]
    else:
        t0 = 0
        t1 = deltaT

    glacier = gm.GlacierModel(name, U, t0, t1, S, dh_path=DHDT)

    glacier.dhdt._update_f()

    if return_model:
        return tdbm.OneDTimeDeltaBayesianModel('model', [glacier], flowline=None,
                                               width_path=WIDTH, bdot_path=BDOT, bed_path=BED)
    else:
        return glacier, WIDTH, BDOT, BED



