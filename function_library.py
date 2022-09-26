# SFSU Imports and Defined Functions
# Author: Bryce R. Baker
# Created: 3/18/22 on Spyder v5.0.0 IDE and Mac OS

# Last Edited: 3/21/22

#%% Imports

import numpy as np
import statsmodels.api as sm
import scipy.integrate as intt
import matplotlib.pyplot as plt
import os

from scipy import sparse
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.special import erf
from BaselineRemoval import BaselineRemoval
# ^ In computer terminal: pip install BaselineRemoval

#%% Imports

#%% Defining Functions

#%%% Data Manipulation

#%%%% Wavelength to Raman Shift Conversion

def wl_rs(data, wl = 531.7):
    
    return (1 / (wl * 10 ** -9) - 1 / (data * 10 ** -9)) * 10 ** -2

#%%%% Wavelength to Raman Shift Conversion

#%%%% Baseline Removal

def apply_baseline(y_data):
    
    if len(np.shape(y_data)) == 1:
        
        baseObj = BaselineRemoval(y_data)
        
        return baseObj.ZhangFit()
        
    if len(np.shape(y_data)) == 2:
        
        temp = np.zeros(np.shape(y_data))
        
        for i in range(len(y_data[0])):
            
            baseObj = BaselineRemoval(y_data[:, i])
            temp[:, i] = baseObj.ZhangFit()
            
        return temp

#%%%% Baseline Removal

#%%%% Locally Weighted Scatterplot Smoothing

lowess = sm.nonparametric.lowess

def smoothing(x_data, y_data, fraction = 1 / 20):
    
    if len(np.shape(y_data)) == 1:
    
        return lowess(y_data, x_data, frac = fraction)[:, 1]
    
    if len(np.shape(y_data)) == 2:
        
        temp = np.zeros(np.shape(y_data))
        
        for i in range(len(y_data[0])):
            
            temp[:, i] = lowess(y_data[:, i], x_data, frac = fraction)[:, 1]
                
        return temp
    

#%%%% Locally Weighted Scatterplot Smoothing

#%%%% Neutrino Removal

def neutrino_removal(y_data, diff = 50):
    
    if len(np.shape(y_data)) == 1:
        
        for i in range(len(y_data) - 1):
        
            k = 1
            
            while y_data[i + k] - y_data[i] > diff:
                
                if i + k == len(y_data) - 1 or k == 10:
                    
                    k = 1
                        
                    break
                
                k += 1
                
            if k != 1:
                
                for j in range(1, k):
                    
                    if y_data[i + k] == y_data[i]:
                        
                        y_data[i + j] = y_data[i]
                        
                    else:
                        
                        y_data[i + j] = y_data[i] + j * (y_data[i + k] - y_data[i]) / k
                        
    if len(np.shape(y_data)) == 2:
    
        for l in range(len(y_data[0])):
            
            for i in range(len(y_data) - 1):
            
                k = 1
                
                while y_data[i + k, l] - y_data[i, l] > diff:
                    
                    if i + k == len(y_data) - 1 or k == 10:
                        
                        k = 1
                        
                        break
                    
                    k += 1
                    
                if k != 1:
                    
                    for j in range(1, k):
                        
                        y_data[i + j,l] = y_data[i, l] + j * (y_data[i + k, l] - y_data[i, l]) / k
                        
    return y_data

#%%%% Neutrino Removal

#%%%% Adjusting the Data Range

def d_range(x, data, minimum, maximum):
    
    for i in range(len(x)):
            
        if x[i] >= minimum: 
                
            x_min = i # Sets minimum on the data set
                
            break

    for i in range(len(x)):
        
        if x[i] >= maximum:
            
            x_max = i # Sets maximum on the data set
                
            break
    
    x0 = x[x_min: x_max]
    data0 = data[x_min: x_max]
        
    return x0, data0

#%%%% Adjusting the Data Range

#%%%% Normalization

def norm(y_data, cut_off = 15):
    
    temp = np.zeros(np.shape(y_data))
    
    if len(np.shape(y_data)) == 1:
        
        temp = y_data - y_data.min()
        temp /= temp.max()
        
        return temp
        
    if len(np.shape(y_data)) == 2:
        
        for i in range(len(y_data[0])):
            
            if y_data[:, i].max() > cut_off:
                
                temp[:, i] = y_data[:, i] - y_data[:, i].min()
                temp[:, i] /= temp[:, i].max()
                
        return temp

#%%%% Normalization

#%%%% Outlier Removal

def outlier_removal(x_data, y_data):
    
    x0 = x_data.copy()
    y0 = y_data.copy()

    d75, d25 = np.percentile(y0, [75, 25])

    iqr = d75 - d25

    q1, q3 = np.quantile(y0, [0.25, 0.75])
    
    x0 = x_data[y_data >= q1 - 1.5 * iqr]
    y0 = y_data[y_data >= q1 - 1.5 * iqr]
    x1 = x0[y0 <= q3 + 1.5 * iqr]
    y1 = y0[y0 <= q3 + 1.5 * iqr]
    
    return x1, y1

#%%%% Outlier Removal

#%%% Data Manipulation

#%%% Data Collection

#%%%% Collecting Data

def get_data(path, file):
    
    temp_list0 = []
    
    os.chdir(path)

    with open(file, "r", encoding = 'cp1252') as f:
        
        lines = f.readlines()
        
        for l in lines[:]:
            
            temp_list0.append(l.split())
    
    f.close() 
    
    return np.array(temp_list0, dtype = float)

def get_all_data(path):
    
    temp_array = np.zeros((1024, len(os.listdir(path)) + 1))
    
    j = 1
    
    for filename in sorted(os.listdir(path), key = lambda x: os.path.getmtime(os.path.join(path, x))):
        
        with open(os.path.join(path, filename), 'r', encoding = 'cp1252') as f:
            
            lines = f.readlines()
            
            for i in range(len(lines)):
                
                if j == 1:
                    
                    temp_array[i, 0] = np.float(lines[i].split()[0])
                    temp_array[i, j] = np.float(lines[i].split()[1])
                    
                else:
                    
                    temp_array[i, j] = np.float(lines[i].split()[1])
                    
            j += 1
            
    f.close()
            
    return temp_array[:, 0], temp_array[:, 1:]

#%%%% Collecting Data

#%%%% Full Width at Half Maximum

def FWHM(x, y):
    
    HM = max(y) / 2.
    
    d = np.sign(HM - np.array(y[0:-1])) - np.sign(HM - np.array(y[1:]))
  
    l = np.where(d > 0)[0]
    r = np.where(d < 0)[-1]
    
    return (x[r] - x[l])[0]

#%%%% Full Width at Half Maximum

#%%%% Determining Bandwidth

def bandwidth(x, data, y0):
    
    temp0 = np.zeros(len(data[0]))
    
    for i in range(len(data[0])):
        
        min0 = 0
        max0 = 0
        
        if data[:, i].max() > y0:
            
            j = 0
        
            while j < len(data) - 3:
            
                if data[j, i] > y0 and data[j + 1, i] > data[j, i] and data[j + 2, i] > data[j + 1, i]:
                
                    min0 = x[j]
                    break
            
                j += 1
        
            k = len(data) - 1
        
            while k > j:
            
                if data[k, i] > y0 and data[k - 1, i] > data[k, i] and data[k - 2, i] > data[k - 1, i]:
                
                    max0 = x[k]
                    break
            
                k -= 1
            
            temp0[i] = max0 - min0
            
    return temp0

#%%%% Determining Bandwidth

#%%% Data Collection

#%%% Fitting

#%%%% Polynomial

def polynomial(x, *parameters):
    
    temp = 0
    
    for i in range(int(len(parameters))):
        
        temp += parameters[i] * x ** i
        
    return temp

#%%%% Polynomial

#%%%% Generalized Polynomial Fitting Procedure

def polynomial_fit(x, data, parameters):
    
    popt, pcov = curve_fit(polynomial, x, data, p0 = parameters)
    
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr

#%%%% Generalized Polynomial Fitting Procedure

#%%%% Guassians

def multigaussian(x, *parameters):
    
    temp = 0
    
    for i in range(int(len(parameters) / 3)):
        
        temp += parameters[1 + 3 * i] / (parameters[2 + 3 * i] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - parameters[0 + 3 * i]) / parameters[2 + 3 * i]) ** 2)
        
    return temp

#%%%% Guassians

#%%%% Generalized Gaussian Fitting Procedure

def multigaussian_fit(x, data, parameters):
    
    popt, pcov = curve_fit(multigaussian, x, data, p0 = parameters)
    
    perr = np.sqrt(np.diag(pcov))
    
    return abs(popt), perr

#%%%% Generalized Gaussian Fitting Procedure

#%%%% Lorentzian

def multilorentzian(x, *parameters):
        
    temp = 0
    
    for i in range(int(len(parameters) / 3)):
        
        temp += parameters[1 + 3 * i] / (parameters[2 + 3 * i] * np.pi) * (parameters[2 + 3 * i] ** 2 / ((x - parameters[0 + 3 * i]) ** 2 + parameters[2 + 3 * i] ** 2))
        
    return temp

#%%%% Lorentzian

#%%%% Generlaized Lorentzian Fitting Procedure

def multilorentzian_fit(x, data, parameters):
    
    popt, pcov = curve_fit(multilorentzian, x, data, p0 = parameters)
    
    perr = np.sqrt(np.diag(pcov))
    
    return abs(popt), perr

#%%%% Generlaized Lorentzian Fitting Procedure

#%%% Fitting

#%%% Plotting

#%%%% Contour Plots

def c_plot(data, ch1_min, ch1_max, ch1_step, ch2_min, ch2_max, ch2_step):
    
    x = np.linspace(ch2_min, ch2_max, ch2_step)
    y = np.linspace(ch1_min, ch1_max, ch1_step)
    
    max0 = data.max()
    min0 = np.min(data[np.nonzero(data)])

    data0 = data.reshape(ch2_step, ch1_step)

    cs = plt.contourf(y, x, data0, levels = np.linspace(min0, max0, 1001), cmap = 'seismic')
    # cs = plt.contourf(y, x, data0, levels = np.arange(25, 75, step = 0.1), cmap = 'seismic')
    plt.colorbar()
    cs.changed()
    plt.gca().set_aspect("equal")
    
    # plt.figure(dpi = 1200)
    # plt.colorbar().set_label('Raman Shift $(cm^{-1})$')
    # plt.title('2D peak FWHM 5T')
    plt.xlabel('Microns')
    plt.ylabel('Microns')
    
    plt.show()

#%%%% Contour Plots

#%%%% X and Y Plots

def y_plots(x, data, ch1_step, ch2_step):
    
    data0 = data.reshape(len(data), ch2_step, ch1_step)

    for i in range(len(data0[0])):
    
        plt.plot(x, data0[:, i, :])
        
        plt.grid()
        # plt.title('Cut of y-Axis')
        # plt.xlabel('Raman Shift (cm^-1)')
        # plt.ylabel('Intensity (A.U.)')
        
        plt.show()
        
def x_plots(x, data, ch1_step, ch2_step):
    
    data0 = data.reshape(len(data), ch2_step, ch1_step)

    for i in range(len(data0[0,0])):
    
        plt.plot(x, data0[:, :, i])
        
        plt.grid()
        # plt.title('Cut of y-Axis')
        # plt.xlabel('Raman Shift (cm^-1)')
        # plt.ylabel('Intensity (A.U.)')
        
        plt.show()

#%%%% X and Y Plots

#%%% Plotting

#%% Defining Functions







