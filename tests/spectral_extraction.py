import numpy as np
import matplotlib.pyplot as plt

from transitspectroscopy import spectroscopy

# Test algorithms by creating a fake 2D (flat) spectrum. First, generate noise image:
nrows = 100
ncolumns = 1000
data = np.random.normal(0.,1., [nrows, ncolumns])

# Generate gaussians centered around row 50:
gaussian_center = 50
gaussian_width = 5
y = np.arange(nrows)

for i in range(ncolumns):

    data[:, i] = data[:, i] + np.exp(-((y - gaussian_center)**2)/ (2. * gaussian_width**2))

# First, perform simple extraction on this fake spectrum:
x = np.arange(ncolumns)
y = np.ones(ncolumns) * 50
aperture_radius = 15
correct_bkg = False
simple_spectrum = spectroscopy.getSimpleSpectrum(data, x, y, aperture_radius, correct_bkg = correct_bkg)

# Let's now try the *fast* simple extraction:
fast_simple_spectrum = spectroscopy.getFastSimpleSpectrum(data, y, aperture_radius)

# And now optimal extraction:
ron = 1.
gain = 1.
nsigma = 10
polynomial_spacing = 0.5
polynomial_order = 2
optimal_spectrum = spectroscopy.getOptimalSpectrum(data, y, aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, data_variance = np.ones(data.shape))

# Plot:
plt.plot(x, simple_spectrum, label = 'Simple spectrum', alpha = 0.5)
plt.plot(x, fast_simple_spectrum, label = 'Simple spectrum (fast)', alpha = 0.5)
plt.plot(x, optimal_spectrum[1,:], label = 'Optimal spectrum', alpha = 0.5)
plt.errorbar(x, optimal_spectrum[1,:], np.sqrt(1./optimal_spectrum[2,:]), fmt = '.')
plt.legend()
plt.show()

plt.plot(x, simple_spectrum - fast_simple_spectrum)
plt.title('Simple - (Fast) Simple')
plt.show()
