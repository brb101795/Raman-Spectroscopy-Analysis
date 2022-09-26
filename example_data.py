from function_library import *


#%%%% Gathering Data

p = "/Users/Bryce/Desktop/Research_Stuff/Newaz_Group/Coding/Raw_Data/attoCUBE/1_4_22"

wl, y_data = get_all_data(p)

rs = wl_rs(wl)

#%%%% Gathering Data

#%%%% Adjusting data

y_data = apply_baseline(y_data)

rs0, peak_g = d_range(rs, y_data, 1525, 1625)

rs1, g_peak1 = d_range(rs, y_data, 1540, 1615)

rs2d, peak_2d = d_range(rs, y_data, 2600, 2850)

peak_2d = apply_baseline(peak_2d)

peak_2d = neutrino_removal(peak_2d, 15)

g_peak1 = smoothing(rs1, g_peak1)

n_peak_2d = norm(peak_2d)

g_peak1 -= np.min(g_peak1)

g_peak1 = g_peak1*(1/np.max(g_peak1))

n_peak_g = norm(peak_g)

#%%%% Adjusting data

#%%%% Fitting

fits_g = np.zeros((len(peak_g[0]), 3))
errs_g = fits_g.copy()

fits_2d = np.zeros((len(peak_g[0]), 6))
errs_2d = fits_2d.copy()

guess_g = (1580, 650, 1)
guess_2d = (2700, 70, 1, 2750, 100, 1)

for i in range(len(peak_g[0])):
    
    fits_g[i], errs_g[i] = multigaussian_fit(rsg, peak_g[:, i], guess_g)
    
    # plt.plot(rsg, peak_g[:, i])
    # plt.plot(rsg, multigaussian(rsg, *fits_g[i]))
    
    # plt.show()
    
    fits_2d[i], errs_2d[i] =  multigaussian_fit(rs2d, peak_2d[:, i], guess_2d)
    
    # plt.plot(rs2d, peak_2d[:, i])
    # plt.plot(rs2d, multigaussian(rs2d, *fits_2d0))
    
    # plt.show()
        
#%%%% Fitting

#%%%% Analyzing Data

x0 = np.linspace(rs[0], rs[-1], 10 ** 6)

mag = np.linspace(0, 9, 91)

g_pos = fits_g[:, 0]

pos_2d = np.zeros(len(mag))

g_int = np.zeros(len(mag))
g_fwhm = np.zeros(len(mag))

fwhm_2d = np.zeros(len(mag))
int_2d = np.zeros(len(mag))

for i in range(len(peak_g[0])):
    
    if fits_2d[i, 3] > fits_2d[i, 0]:
        
        pos_2d[i] = fits_2d[i, 3]
        
    else:
        
        pos_2d[i] = fits_2d[i, 0]
    
    g_int[i] = intt.simps(multigaussian(x0, *fits_g[i]), x = x0)
    g_fwhm[i] = FWHM(x0, multigaussian(x0, *fits_g[i]))
    int_2d[i] = intt.simps(multigaussian(x0, *fits_2d[i]), x = x0)
    fwhm_2d[i] = FWHM(x0, multigaussian(x0, *fits_2d[i]))

ABA_data = np.zeros((3, 2, 91))

ABA_data[0, 0] = g_pos
ABA_data[1, 0] = g_int
ABA_data[2, 0] = g_fwhm
ABA_data[0, 1] = pos_2d
ABA_data[1, 1] = int_2d
ABA_data[2, 1] = fwhm_2d

#%%%% Analyzing Data

#%%%% Plotting

#%%%%% Raw Data

for i in range(3, 8):
    
    plt.plot(rs0, n_peak_g[:, i * 10] + i * 0.8)
    
plt.title('ABA G Peak')
plt.xlabel('Raman Shift $(cm^{-1})$')
plt.ylabel('Normalized Arbitrary Units')
plt.legend(('3T', '4T', '5T', '6T', '7T'))

ax = plt.gca()
ax.axes.yaxis.set_ticks([])

plt.show()

plt.plot(rs2d, peak_2d[:, 0])
plt.plot(rs2d, multigaussian(rs2d, *fits_2d[0]))

plt.plot(rs2d, multigaussian(rs2d, *fits_2d[0, :3])-1)
plt.plot(rs2d, multigaussian(rs2d, *fits_2d[0, 3:])-2)
# Gaussians shifted down for clarity

plt.title('ABA 2D Peak 0T')
plt.xlabel('Raman Shift $(cm^{-1})$')
plt.ylabel('Arbitrary Units')
plt.legend(('Raw Data', 'Double Gaussian', 'Gaussian 1', 'Gaussian 2'))

ax = plt.gca()
ax.axes.yaxis.set_ticks([])

plt.show()

#%%%%% Raw Data

#%%%%% Analyzed Data

plt.plot(mag, g_pos, 'o-')

plt.title('ABA G Peak Positions vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Raman Shift $(cm^{-1})$')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.plot(mag, g_int, 'o-')

plt.title('ABA G Peak Intensity vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Counts')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.plot(mag, g_fwhm, 'o-')

plt.title('ABA G Peak FWHM vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Raman Shift $(cm^{-1})$')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.plot(mag, pos_2d, 'o-')

plt.title('ABA 2D Peak Positions vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Raman Shift $(cm^{-1})$')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.plot(mag, int_2d, 'o-')

plt.title('ABA 2D Peak Intensity vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Counts')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.plot(mag, fwhm_2d, 'o-')

plt.title('ABA 2D Peak FWHM vs. Magnetic Field')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Raman Shift ($(cm^{-1})$')

plt.xticks(np.arange(0, 10, step = 1))

plt.show()

plt.contourf(rs1, mag, np.transpose(g_peak1), levels = np.arange(0, 1.1, step = 0.025), cmap = 'RdBu_r')

cbar = plt.colorbar( )

plt.title('ABA G Peak')
plt.xlabel('Raman Shift $(cm^{-1})$')
plt.ylabel('Magnetic Field (T)')

cbar.set_label("Normalized Arbitrary Units")

plt.show()

#%%%%% Analyzed Data

#%%%% Plotting