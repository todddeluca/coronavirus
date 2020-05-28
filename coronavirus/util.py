
import numpy as np
from scipy import stats


def fit_power_law(x, y, xlow=None, xhigh=None):
    '''
    log transform x and y.  fit a line. return xmin, xmax (range of the line), the slope
     (exponent in the power law), etc.
    xlow: minimum of range for fitting line
    xhigh: maximum of range for fitting line
    '''
    # print(f'np.min(x) {np.min(x)} np.max(x) {np.max(x)}')
    lx = np.log10(x)
    ly = np.log10(y)
    lxmin = np.min(lx) if xlow is None else np.log10(xlow)
    lxmax = np.max(lx) if xhigh is None else np.log10(xhigh)
    # print(f'lxmin {lxmin} lxmax {lxmax} np.min(lx) {np.min(lx)} np.max(lx) {np.max(lx)}')
    # print(len(lx), len(ly))
    lx2 = lx[(lx >= lxmin) & (lx <= lxmax)]
    ly2 = ly[(lx >= lxmin) & (lx <= lxmax)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(lx2, ly2)
    r_squared = r_value ** 2
    # predict some x,y points over the range using the fit.
    xs = np.linspace(10**lxmin, 10**lxmax, 500) # make a few hundred points to plot the line (could use just 2?)
    ys = (10**intercept)*(xs**slope) # y=cx**b form -> log(y) = logc + blogx
    return 10**lxmin, 10**lxmax, slope, intercept, p_value, std_err, r_value, r_squared, xs, ys


