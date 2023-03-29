import os, gzip
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve, fmin
from scipy import signal
import pandas as pd
import numpy as np
import gsw
from datetime import datetime as dt

### GENERAL SCRIPTS
def RunningMedian(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.NaN
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmedian(grid,axis=1)[N:-N]

def RunningMax(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.NaN
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmax(grid,axis=1)[N:-N]

def RunningMin(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.NaN
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmin(grid,axis=1)[N:-N]

def RunningMean(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.NaN
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmean(grid,axis=1)[N:-N]

def interp(x,y,xi):
    _gg = np.isfinite(x+y)
    return interp1d(x[_gg], y[_gg], bounds_error=False, fill_value=np.NaN)(xi)

def rmsd(x):
    return np.sqrt(np.nanmean(x**2))

def plog(msg):
    print(str(dt.now().replace(microsecond=0))+' : '+msg)
    return None

### DATA DOWNLOADss
def download_ERA5(years=[2021], months=[1], days=[1], hours=[0], area=[0,0,0,0], variables=None, output_name=None):
    """
    Returns xarray dataset.
    
    years = list of year numbers
    months = list of month numbers
    days = list of days, from 1 to 31
    hours = list of hours from 0 to 23
    area = list of [N,W,S,E]
    variables = default collects heat flux variables, otherwise, specify.
    output_name = file to save to if desired
    """
    
    # Area = [N,W,S,E]
    
    # First provided by Marcel du Plessis
    # Loading ERA5 data directly from the Copernicus Data Service
    # The goal of this notebook is to be able to access and analysis ERA5 data using cloud computing. 
    # This has the obvious advantage of not having to download and store the data on your local computer, 
    # which can quicly add up to terrabytes if you're looking for long term data.

    # I am following an example from https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0

    # Variables on the single levels reanalysis can be found here: 
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview

    # Let's see how we go:
    
    if variables is None:
        variables = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'sea_surface_temperature',
            'skin_temperature',
            'surface_pressure',
            'surface_latent_heat_flux',
            'surface_sensible_heat_flux',
            'surface_net_solar_radiation',
            'surface_net_thermal_radiation',
            'total_precipitation',
            'evaporation']
        
    import cdsapi
    import xarray as xr
    from urllib.request import urlopen # start the client

    import certifi
    import urllib3
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
        )

    cds = cdsapi.Client() # dataset you want to read

    dataset = "reanalysis-era5-single-levels" # flag to download data
    download_flag = True # api parameters 
    params = {
              'variable': variables,
              'product_type': 'reanalysis',
              'year': [str(x) for x in years],
              'month': [str(format(x, '02')) for x in months],
              'day': [str(format(x, '02')) for x in days],
              'time': [str(format(x, '02'))+':00' for x in hours],
              'area': area,
              'format': 'netcdf'
             }

    fl = cds.retrieve(dataset, params) # download the file 

    if output_name is not None:
        if output_name[-3:] != '.nc':
            output_name = output_name+'.nc'
            
        fl.download(str(output_name))

    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())
        
    return ds

### OPTICAL FUNCTIONS
def austinPetzold_1986(wavelength, k490):
    wave = np.array([350, 360, 370, 380, 390, 400,
        410, 420, 430, 440, 450, 460, 470, 480, 490, 500,
        510, 520, 530, 540, 550, 560, 570, 580, 590, 600,
        610, 620, 630, 640, 650, 660, 670, 680, 690, 700 ])
    M = np.array([2.1442, 2.0504, 1.9610, 1.8772, 1.8009, 1.7383,
        1.7591, 1.6974, 1.6108, 1.5169, 1.4158, 1.3077, 1.1982, 1.0955, 1.0000, 0.9118, 
        0.8310, 0.7578, 0.6924, 0.6350, 0.5860, 0.5457, 0.5146, 0.4935, 0.4840, 0.4903, 
        0.5090, 0.5380, 0.6231, 0.7001, 0.7300, 0.7301, 0.7008, 0.6245, 0.4901, 0.2891 ])
    Kdw = np.array([0.0510, 0.0405, 0.0331, 0.0278, 0.0242, 0.0217, 
        0.0200, 0.0189, 0.0182, 0.0178, 0.0176, 0.0176, 0.0179, 0.0193, 0.0224, 0.0280, 
        0.0369, 0.0498, 0.0526, 0.0577, 0.0640, 0.0723, 0.0842, 0.1065, 0.1578, 0.2409, 
        0.2892, 0.3124, 0.3296, 0.3290, 0.3559, 0.4105, 0.4278, 0.4521, 0.5116, 0.6514 ])

    for i in np.arange(1,36):
        if wave[i] >= wavelength:
            l1 = wave[i]
            k1 = Kdw[i]
            m1 = M[i]
            l0 = wave[i-1]
            k0 = Kdw[i-1]
            m0 = M[i-1]
            break

    num = wavelength - l0
    den = l1 - l0
    frac = num / den

    kdiff = k1 - k0
    Kdw_l = k0 + frac*kdiff

    mdiff = m1 - m0
    M_l = m0 + frac*mdiff

    ref = np.argmin(np.abs(wave-490))
    Kd = (M_l/M[ref]) * (k490 - Kdw[ref]) + Kdw_l
    return Kd

def parFraction(wavelength):
    """
    Returns PAR fraction assuming boring, standard, generic incoming solar radiation and flat response from PAR sensor.
    Units in "per nm".    
    """
    wavelengths = np.array([400, 412, 443, 490, 510, 555, 625, 670, 700]);
    parFraction = np.array([0.0029, 0.0032, 0.0035, 0.0037, 0.0037, 0.0036, 0.0032, 0.0030, 0.0024]);
    return interp1d(wavelengths,parFraction, bounds_error=True, fill_value=np.NaN)(wavelength)

def morelCoefficients(wavelength):
    """
    input: wavelength
    output: Kw, eChl, XChl
    
    link: 'http://onlinelibrary.wiley.com/doi/10.1029/2000JC000319/epdf'
    title: 'Bio-optical properties of oceanic waters: A reappraisal'
    author: 'André Morel, Stéphane Maritorena'
    doi: '10.1029/2000JC000319'
    algorithm: 'Kd(lambda) = Kw(lambda) + XChl(lambda) * [Chl]^(eChl(lambda))'
    """
    wavelengths = np.array([350,355,360,365,370,375,380,385,390,395,
                   400,405,410,415,420,425,430,435,440,445,
                   450,455,460,465,470,475,480,485,490,495,
                   500,505,510,515,520,525,530,535,540,545,
                   550,555,560,565,570,575,580,585,590,595,
                   600,605,610,615,620,625,630,635,640,645,
                   650,655,660,665,670,675,680,685,690,695,700])
    
    interp = lambda arr: interp1d(wavelengths, arr, bounds_error=True, fill_value=np.NaN)(wavelength)
    
    Kw = np.array([0.0271,0.0238,0.0216,0.0188,0.0177,0.01595,0.0151,0.01376,0.01271,0.01208,
          0.01042,0.0089,0.00812,0.00765,0.00758,0.00768,0.0077,0.00792,0.00885,0.0099,
          0.01148,0.01182,0.01188,0.01211,0.01251,0.0132,0.01444,0.01526,0.0166,0.01885,
          0.02188,0.02701,0.03385,0.0409,0.04214,0.04287,0.04454,0.0463,0.04846,0.05212,
          0.05746,0.06053,0.0628,0.06507,0.07034,0.07801,0.09038,0.11076,0.13584,0.16792,
          0.2231,0.25838,0.26506,0.26843,0.27612,0.284,0.29218,0.30176,0.31134,0.32553,
          0.34052,0.3715,0.41048,0.42947,0.43946,0.44844,0.46543,0.48642,0.5164,0.55939,0.62438])

    eChl = np.array([0.778,0.767,0.756,0.737,0.72,0.7,0.685,0.673,0.67,0.66,0.64358,0.64776,
            0.65175,0.65555,0.65917,0.66259,0.66583,0.66889,0.67175,0.67443,0.67692,
            0.67923,0.68134,0.68327,0.68501,0.68657,0.68794,0.68903,0.68955,0.68947,
            0.6888,0.68753,0.68567,0.6832,0.68015,0.67649,0.67224,0.66739,0.66195,0.65591,
            0.64927,0.64204,0.64,0.63,0.623,0.615,0.61,0.614,0.618,0.622,0.626,0.63,
            0.634,0.638,0.642,0.647,0.653,0.658,0.663,0.667,0.672,0.677,0.682,0.687,0.695,
            0.697,0.693,0.665,0.64,0.62,0.6])
            
    XChl = np.array([0.153,0.149,0.144,0.14,0.136,0.131,0.127,0.123,0.119,0.118,0.11748,0.12066,
            0.12259,0.12326,0.12269,0.12086,0.11779,0.11372,0.10963,0.1056,0.10165,
            0.09776,0.09393,0.09018,0.08649,0.08287,0.07932,0.07584,0.07242,0.06907,
            0.06579,0.06257,0.05943,0.05635,0.05341,0.05072,0.04829,0.04611,0.04419,
            0.04253,0.04111,0.03996,0.039,0.0375,0.036,0.034,0.033,0.0328,0.0325,
            0.033,0.034,0.035,0.036,0.0375,0.0385,0.04,0.042,0.043,0.044,0.0445,
            0.045,0.046,0.0475,0.049,0.0515,0.052,0.0505,0.044,0.039,0.034,0.03])
    
    return interp(Kw), interp(eChl), interp(XChl)
    
def betasw_ZHH2009(Tc,S,wavelength=700,theta=117,delta=0.039):
    # Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scatteirng by pure
    # seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710 
    #
    # wavelength: backscatter wavelength (nm)
    # Tc: temperauter in degree Celsius
    # S: salinity
    # delta: depolarization ratio, if not provided, default = 0.039 will be used.
    # theta = beam scattering angle in degrees
    # betasw: volume scattering at angles defined by theta. Its size is [x y],
    # where x is the number of angles (x = length(theta)) and y is the number
    # of wavelengths in wavelength (y = length(wavelength))
    # beta90sw: volume scattering at 90 degree. Its size is [1 y]
    # bw: total scattering coefficient. Its size is [1 y]
    # for backscattering coefficients, divide total scattering by 2
    #
    # Xiaodong Zhang, March 10, 2009
    
    def RInw():
        # refractive index of air is from Ciddor (1996,Applied Optics)
        n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (wavelength/1e3)**2) + 167917.0 / (57.362 - 1/(wavelength/1e3)**2)) / 1e8

        # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
        n0 = 1.31405
        n1 = 1.779e-4
        n2 = -1.05e-6
        n3 = 1.6e-8
        n4 = -2.02e-6
        n5 = 15.868
        n6 = 0.01155
        n7 = -0.00423
        n8 = -4382
        n9 = 1.1455e6

        nsw = n0  +  (n1 + n2*Tc + n3*Tc**2)*S  +  n4*Tc**2  +  (n5 + n6*S + n7*Tc)/wavelength  +  n8/wavelength**2  +  n9/wavelength**3 # pure seawater
        nsw = nsw*n_air
        dnswds = (n1 + n2*Tc + n3*Tc**2 + n6/wavelength) * n_air
        return nsw, dnswds
    
    def BetaT():
        # pure water secant bulk Millero (1980, Deep-sea Research)
        kw = 19652.21 + 148.4206*Tc - 2.327105*Tc**2 + 1.360477e-2*Tc**3 - 5.155288e-5*Tc**4
        Btw_cal = 1/kw
        # isothermal compressibility from Kell sound measurement in pure water
        # Btw = (50.88630+0.717582*Tc+0.7819867e-3*Tc**2+31.62214e-6*Tc**3-0.1323594e-6*Tc**4+0.634575e-9*Tc**5)./(1+21.65928e-3*Tc)*1e-6;
        # seawater secant bulk
        a0 = 54.6746 - 0.603459*Tc + 1.09987e-2*Tc**2 - 6.167e-5*Tc**3
        b0 = 7.944e-2 + 1.6483e-2*Tc - 5.3009e-4*Tc**2
        Ks = kw + a0*S + b0*S**1.5

        # calculate seawater isothermal compressibility from the secant bulk
        IsoComp = 1/Ks*1e-5 # unit is pa
        return IsoComp

    def rhou_sw():
        # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
        a0 = 8.24493e-1
        a1 = -4.0899e-3
        a2 = 7.6438e-5
        a3 = -8.2467e-7
        a4 = 5.3875e-9
        a5 = -5.72466e-3
        a6 = 1.0227e-4
        a7 = -1.6546e-6
        a8 = 4.8314e-4
        
        b0 = 999.842594
        b1 = 6.793952e-2
        b2 = -9.09529e-3
        b3 = 1.001685e-4
        b4 = -1.120083e-6
        b5 = 6.536332e-9

        # density for pure water 
        density_w = b0+b1*Tc+b2*Tc**2+b3*Tc**3+b4*Tc**4+b5*Tc**5
        # density for pure seawater
        density_sw = density_w +((a0+a1*Tc+a2*Tc**2+a3*Tc**3+a4*Tc**4)*S+(a5+a6*Tc+a7*Tc**2)*S**1.5+a8*S**2)
        return density_sw

    
    def dlnasw_ds():
        # water activity data of seawater is from Millero and Leung (1976,American
        # Journal of Science,276,1035-1077). Table 19 was reproduced using
        # Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
        # dlnawds is partial derivative of natural logarithm of water activity
        # w.r.t.salinity
        # lnaw = (-1.64555e-6-1.34779e-7*Tc+1.85392e-9*Tc**2-1.40702e-11*Tc**3)+......
        #            (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3)*S+......
        #            (1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3)*S**1.5+......
        #            (-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S**2;

        dlnawds = (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3) + 1.5*(1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3)*S**0.5 + 2*(-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S
        return dlnawds

    
    # density derivative of refractive index from PMH model
    def PMH(n_wat):
        n_wat2 = n_wat**2
        n_density_derivative = (n_wat2 - 1) * (1+2/3*(n_wat2+2) * (n_wat/3-1/3/n_wat)**2)
        return n_density_derivative
    
    # values of the constants
    Na = 6.0221417930e23   #  Avogadro's constant
    Kbz = 1.3806503e-23    #  Boltzmann constant
    Tk = Tc+273.15         #  Absolute tempearture
    M0 = 18e-3             #  Molecular weigth of water in kg/mol
    
    rad = theta*np.pi/180 # angle in radian as a colum variable

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw()

    # isothermal compressibility is from Lepple & Millero (1971,Deep
    # Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT()

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rhou_sw()

    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds()

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  ## PMH model

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = np.pi*np.pi/2*((wavelength*1e-9)**(-4))*Kbz*Tk*IsoComp*DFRI**2*(6+6*delta)/(6-7*delta)
    
    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S*M0*dnds**2/density_sw/(-dlnawds)/Na
    beta_cf = 2*np.pi*np.pi*((wavelength*1e-9)**(-4))*nsw**2*(flu_con)*(6+6*delta)/(6-7*delta)
    
    # total volume scattering at 90 degree
    beta90sw = beta_df+beta_cf
    bsw=8*np.pi/3*beta90sw*(2+delta)/(1+delta)
    
    betasw = beta90sw * (1+((np.cos(rad))**2)*(1-delta)/(1+delta))

    return betasw,beta90sw,bsw