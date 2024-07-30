import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gamma
from typing import List, Union
import matplotlib.pyplot as plt
from scipy import interpolate, integrate

plt.ion()
plt.close("all")
def mixture_gamma(x: np.ndarray, *params: float) -> np.ndarray:
    """
    Calculate the probability density function (PDF) of a mixture of gamma distributions.

    Parameters:
        x (np.ndarray): Input values at which to evaluate the PDF.
        params (float): Variable-length argument list containing the parameters.
                        The parameters should be provided in the following order:
                        [weights, shapes, scales]

    Returns:
        np.ndarray: The PDF values at the given input values x.
    """
    num_distributions = len(params) // 3
    weights = params[:num_distributions]
    shapes = params[num_distributions:2*num_distributions]
    scales = params[2*num_distributions:]

    pdf = np.zeros_like(x)
    for w, a, b in zip(weights, shapes, scales):
        pdf += w * gamma.pdf(x, a, scale=b)
    return pdf

# Generate sample data
x_data = np.linspace(0, 10, 50)
y_data = 70 * gamma.pdf(x_data, 2, scale=1) + 33 * gamma.pdf(x_data, 5, scale=2)

# Initial parameter estimates
initial_params = [50, 25, 2, 5, 1, 2]  # Example initial parameters

# Fit the mixture_gamma function to the data
params, _ = curve_fit(mixture_gamma, x_data, y_data, p0=initial_params)

ydf     = np.diff(y_data)
global_min_x = np.argmin(ydf)
xnew    = x_data[global_min_x:]
ind_sep = np.where(ydf[global_min_x:] > -.08)[0][0]
xsep = xnew[ind_sep]
ynew  = y_data[global_min_x:]

x1 = x_data[x_data<xsep]
x2 = x_data[x_data>xsep]
y1 = y_data[x_data<xsep]
y2 = y_data[x_data>xsep]

f1 = interpolate.interp1d(x1, y1, kind="quadratic", fill_value="extrapolate")
yp1 = f1(x_data)

f2 = interpolate.interp1d(x2, y2, kind="quadratic", fill_value="extrapolate")
yp2 = f2(x_data)

int1 = integrate.simps(y1, x1)
int2 = integrate.simps(y2, x2)

re1 = integrate.simps(x1**3*y1, x1)/integrate.simps(x1**2*y1,x1)
rm1 = integrate.simps(x1*y1, x1)/int1
nu1 = 2*rm1/(re1-rm1)
bet1 = rm1/nu1

re2 = integrate.simps(x2**3*y2, x2)/integrate.simps(x2**2*y2,x2)
rm2 = integrate.simps(x2*y2, x2)/int1
nu2 = 2*rm2/(re2-rm2)
bet2 = rm2/nu2

estimated_param =[int1, int2, nu1, nu2, bet1, bet2]

print("Estimated parameters:", estimated_param)
print("True parameters:", params)

plt.figure()
plt.plot(x_data, y_data, "ob")
plt.plot(x_data, mixture_gamma(x_data, *params), 'r')
plt.plot(xnew[ind_sep], ynew[ind_sep], 'ok')
plt.fill_between([x1[0], x1[-1]], [80, 80], alpha=.5)
plt.fill_between([x2[0], x2[-1]], [80, 80], alpha=.5)
# plt.plot(x_data, yp1, '*g')
# plt.plot(x_data, yp2, '*m')
plt.yscale("log")
plt.show()

plt.figure()
plt.plot(x_data[:-1], np.diff(y_data), 'ob')
plt.plot(x_data[:-1], np.zeros(len(x_data)-1), '-k')
plt.show()

# Generate sample data
x_data = np.linspace(0, 50, 100)
true_param = [8, .7]
y_data = gamma.pdf(x_data, true_param[0], scale=true_param[1])

plt.figure()
plt.plot(x_data, y_data, 'ob')
plt.yscale("log")
plt.show()

m3  = integrate.simps(x_data**3*y_data, x_data)
m2  = integrate.simps(x_data**2*y_data, x_data)
m1  = integrate.simps(x_data*y_data, x_data)
var = m2 - m1**2

re = m3/m2
rm = m1

v = 2*rm/(re-rm)
b = rm/v

print("True", true_param[0], true_param[1])
print("EStimated", m1**2/var, var/m1)
print("EStimated", v, b)