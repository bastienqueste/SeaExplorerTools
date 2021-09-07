import os, gzip

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve, fmin
import pandas as pd
import numpy as np
import gsw

from tqdm import tqdm

###########################
#    SX PANDAS DF CLASS   #
###########################               
class sxdf(object):
    def __init__(self, *args):
        self.data = pd.DataFrame()
        for arg in args:
            if type(arg) is list:
                for k in arg:
                    self.load_gzfiles(k)
            else:
                self.load_gzfiles(arg)

    def load_gzfiles(self, file_dir):
        ''' Load payload data into one dataframe and convert main timestamp to datetime format. '''
        file_list = np.array(os.listdir(file_dir))
        file_list = file_list[ [(fn.split('.')[-1] == 'gz') for fn in file_list] ]
        file_list = file_list[ [(len(fn.split('.')) == 6) for fn in file_list] ]
        
        _tmp = []
        for fileName in tqdm(file_list):
            dive = pd.read_csv(gzip.open(file_dir+os.sep+fileName), sep=';')
            dive["diveNum"] = int(fileName.split('.')[-2])
            #dive["missionNum"] = int(fileName.split('.')[1])
            #dive["gliderID"] = fileName.split('.')[0]

            if 'Timestamp' in dive:
                dive['timeindex'] = pd.to_datetime(dive['Timestamp'], format="%d/%m/%Y %H:%M:%S", utc=True, origin='unix', cache='False')

            if 'PLD_REALTIMECLOCK' in dive:
                dive['PLD_REALTIMECLOCK'] = pd.to_datetime(dive['PLD_REALTIMECLOCK'], format="%d/%m/%Y %H:%M:%S.%f", utc=True, origin='unix', cache='False')
                dive.rename(columns={'PLD_REALTIMECLOCK': 'timeindex'}, inplace=True)

            dive['Timestamp'] = dive['timeindex'].values.astype("float")
            dive.set_index('timeindex', inplace=True)
            _tmp.append(dive[dive.index > '2020-01-01'].resample('S').mean())
                        
        for d in range(len(_tmp)):
            _tmp[d]['Timestamp'] = pd.to_datetime(_tmp[d]['Timestamp'].interpolate('linear'), utc=True, origin='unix', cache='False')
        
        self.data = self.data.append(pd.concat(_tmp, ignore_index=True), sort=True)
        self.data.sort_values('Timestamp', ignore_index=True, inplace=True)
        
    def save(self, file_name):
        if file_name:
            print('Saving to '+file_name)
            self.data.to_parquet(file_name, coerce_timestamps='ms', allow_truncated_timestamps=True, compression='ZSTD')

    def median_resample(self):
        self.data.set_index('Timestamp', inplace=True, drop=False)
        self.data['Timestamp'] = self.data['Timestamp'].values.astype("float")
        #self.data = self.data.resample('S').mean()
        self.data = self.data.resample('S').median()
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'].interpolate('linear'), utc=True, origin='unix', cache='False')
        
    def process_basic_variables(self):
        # TODO: move basic parsing from ipynb to here
        # parseGPS from NAV
        # fillNA in DeadReckoning with 1's (ie. 0 only valid GPS)
        # interpolate lat/lon?
        # interpolate pressure
        # Include printed output so user know what is going on.
        return None

        
        
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
#                          #
# GENERAL HELPER FUNCTIONS #
#                          #  
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
def load(parquet_file):
    out = sxdf()
    out.data = pd.read_parquet(parquet_file)
    return out

def parseGPS(sxGPS):    ## CALCULATE SUBSURFACE LAT / LON (TODO: USING DEAD RECKONING)
    return np.sign(sxGPS) * (np.fix(np.abs(sxGPS)/100) + np.mod(np.abs(sxGPS),100)/60)

def date2float(d, epoch=pd.to_datetime(0, utc=True, origin='unix', cache='False')):
    return (d - epoch).dt.total_seconds()


def grid2d(x, y, v, xi=1, yi=1):
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x)+xi, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y)+yi, yi)

    raw = pd.DataFrame({'x':x,'y':y,'v':v}).dropna()

    grid = np.full([np.size(xi),np.size(yi)], np.nan)
    
    raw['xbins'],xbin_iter = pd.cut(raw.x, xi,retbins=True,labels=False)
    raw['ybins'],ybin_iter = pd.cut(raw.y, yi,retbins=True,labels=False)

    _tmp = raw.groupby(['xbins','ybins'])['v'].agg('median')
    grid[_tmp.index.get_level_values(0).astype(int),_tmp.index.get_level_values(1).astype(int)] = _tmp.values

    YI,XI = np.meshgrid(yi,xi)
    #print(np.shape(xi),np.shape(yi),np.shape(XI),np.shape(YI),np.shape(grid))
    print('Gridded.')
    return grid,XI,YI

def applyLag(x,time,N):
    try:
        time = date2float(time)
        print('Applying lag after time conversion:')
    except:
        print('Applying lag:')
    _gg = np.isfinite(time+x)
    i_time = np.arange(time[0],time[-1])
    
    i_fn =  interp1d(time[_gg], x[_gg], bounds_error=False, fill_value=np.NaN)
    i_x = i_fn(i_time) 
    filt = np.concatenate([np.full(N-1,0),np.flip(np.arange(N))**2])
    filt = filt/sum(filt)

    i_x = np.convolve(i_x,filt,mode='same')
    i_x[0:N+1] = x[0:N+1] 

    _fn = interp1d(i_time, i_x, bounds_error=False, fill_value=np.NaN)
    return _fn(time)

def correctSalinity(conductivity,temperature,pressure,time,lag=None):
    time = date2float(time)

    def _salinityCorrection(x):
        _temperature_internal = temperature + (x * np.gradient(temperature,time))
        return gsw.conversions.SP_from_C(conductivity,_temperature_internal,pressure)
        
    if lag is None:
        from scipy import optimize 
        def _PolyArea(x,y):
            _gg = np.isfinite(x+y) 
            return 0.5*np.abs(np.dot(x[_gg],np.roll(y[_gg],1))-np.dot(y[_gg],np.roll(x[_gg],1)))
            # Doesn't work well with intersecting polygon.... Problem??
        def _f(x):
            return _PolyArea(_salinityCorrection(x),temperature)
        print('Regressing lag coefficient for salinity thermal hysteresis correction:')
        minimum = optimize.fmin(_f, 0.5, disp=True, full_output=True)
        print('Salinity lag coefficient = '+np.array2string(minimum[0]))
        lag = minimum[0]

    return _salinityCorrection(lag)

def lagCorrection(variable,referencevariable,time,lag=None):
    try:
        time = date2float(time)
        print('Applying lag after time conversion:')
    except:
        print('Applying lag:')
        
    def _Correction(x):
        return variable + (x * np.gradient(variable,time))
        
    if lag is None:
        from scipy import optimize 
        def _PolyArea(x,y):
            _gg = np.isfinite(x+y) 
            return 0.5*np.abs(np.dot(x[_gg],np.roll(y[_gg],1))-np.dot(y[_gg],np.roll(x[_gg],1)))
            # Doesn't work well with intersecting polygon.... Problem??
        def _f(x):
            return _PolyArea(_Correction(x),referencevariable)
        print('Regressing lag coefficient:')
        minimum = optimize.fmin(_f, 0.5, disp=True, full_output=True)
        print('Calculated lag coefficient = '+np.array2string(minimum[0]))
        lag = minimum[0]

    return _Correction(lag)

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
#                           #
# SLOCUM FLIGHT MODEL CLASS #
#                           #  
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
class SlocumModel(object):
    def __init__(self,time,sal,temp,pres,lon,lat,ballast,pitch,profile,navresource,ADCP_vel=None,**param):
        
        self.timestamp = time
        self.time = date2float(self.timestamp)
        self.pressure = pres
        self.longitude = lon
        self.latitude = lat
        self.profile = profile
        self.ADCP_vel = ADCP_vel
        
        self.temperature = temp
        self.salinity = sal
        
        self.ballast = ballast/1000000 # m^3
        self.pitch = np.deg2rad(pitch) # rad

        self.AR = 7
        self.eOsborne = 0.8
        self.Cd1_hull = 2.1
        self.Omega = 0.75

        self.param_reference = dict({
            'mass': 60.772, # Vehicle mass in kg
            'vol0': 59.015 / 1000, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'area_w': 0.09, # Wing surface area, m**2
            'Cd_0': 0.11781, #
            'Cd_1': 2.94683, #
            'Cl_w': 3.82807, # 
            'Cl_h': 3.41939, # 
            'comp_p': 4.5e-06, # Pressure dependent hull compression factor
            'comp_t': -6.5e-05 # Temperature dependent hull compression factor
        })

        self.param = self.param_reference.copy()
        for k,v in param.items():
            self.param[k] = v
        self.param_initial = self.param
        
        def fillGaps(x,y):
            f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
            return(f(x))
                
        def RM(x,N):
            big = np.full([N,len(x)+N-1],np.nan)
            for n in np.arange(N):
                if n == N-1:
                    big[n, n : ] = x
                else:
                    big[n, n : -N+n+1] = x
            return np.nanmedian(big[:,int(np.floor(N/2)):-int(np.floor(N/2))],axis=0)

        def smooth(x,N):
            return np.convolve(x, np.ones(N)/N, mode='same')
                
        #self.pressure = fillGaps(self.time, self.raw_pressure)
        #self.temperature = fillGaps(self.time, self.raw_temperature)
        #self.salinity = fillGaps(self.time, self.raw_salinity)

        #self.pressure = smooth(RM(self.pressure,5),20)
        #self.temperature = smooth(RM(self.temperature,5),20)
        #self.salinity = smooth(RM(self.salinity,5),20)
        
        self.depth = gsw.z_from_p(self.pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1

        self.g = gsw.grav(self.latitude,self.pressure)
        
        self.SA = gsw.SA_from_SP(self.salinity, self.pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.temperature, self.pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.pressure)

        ### Basic model
        # Relies on steady state assumption that buoyancy, weight, drag and lift cancel out when not accelerating.
        # F_B - cos(glide_angle)*F_L - sin(glide_angle)*F_D - F_g = 0 
        # cos(glide_angle)*F_L + sin(glide_angle)*F_D = 0

        # Begin with an initial computation of angle of attack and speed through water:
        self.model_function()

        ### Get good datapoints to regress over
        self._valid = np.full(np.shape(self.pitch),True)
        self._valid[self.pressure < 5] = False
        self._valid[np.abs(self.pitch) < 0.2] = False   # TODO change back to 15
        self._valid[np.abs(self.pitch) > 0.6] = False   # TODO change back to 15
        self._valid[np.abs(np.gradient(self.dZdt,self.time)) > 0.005] = False # Accelerations
        self._valid[np.gradient(self.pitch,self.time)==0] = False
        self._valid = self._valid & ((navresource == 100) | (navresource == 117) )
        
        print('Number of valid points: '+str(np.count_nonzero(self._valid))+' (out of '+str(len(self._valid))+')')
        
        # Do first pass regression on vol parameters, or volume and hydro?
        self.regression_parameters = ('vol0','Cd_0','Cd_1','comp_p','comp_t') 
        #('vol0','comp_p','comp_t','Cd_0','Cd_1','Cl_h','Cl_w') # Has to be a tuple, requires a trailing comma if single element

    ### Principal forces
    @property
    def F_B(self):
        return self.g * self.rho * (self.ballast + self.vol0 * (1 - self.comp_p*self.pressure - self.comp_t*(self.temperature-10)))

    @property
    def F_g(self):
        return self.mass * self.g

    @property
    def F_L(self):
        return self.dynamic_pressure * (self.Cd_1) * self.alpha

    @property
    def F_D(self):
        return self.dynamic_pressure * (self.Cd_0 + self.Cd_1 * self.alpha**2)

    @property
    def Pa(self):
        return self.pressure * 10000 # Pa

    ### Important variables
    @property
    def glide_angle(self):
        return self.pitch + self.alpha
    
    @property
    def dynamic_pressure(self):
        return self.rho * self.area_w * self.speed**2 / 2

    @property
    def w_H2O(self):
        # Water column upwelling
        return self.dZdt - self.speed_vert

    ### Basic equations
    def _solve_alpha(self):
        _pitch_range = np.linspace( np.deg2rad(0), np.deg2rad(90) , 100)
        _alpha_range = np.zeros_like(_pitch_range)
        for _istep, _pitch in enumerate(_pitch_range):
            _tmp = fsolve(self._equation_alpha, 0.001, args=(_pitch), full_output=True)
            _alpha_range[_istep] = _tmp[0]
        _interp_fn = interp1d(_pitch_range,_alpha_range)
        return _interp_fn(np.abs(self.pitch)) * np.sign(self.pitch)

    def _equation_alpha(self, _alpha, _pitch):
        return (self.Cd_0 + self.Cd_1 * _alpha**2) / ( (self.Cd_1) * np.tan(_alpha + _pitch)) - _alpha

    def _solve_speed(self):
        _dynamic_pressure = (self.F_B - self.F_g) * np.sin(self.glide_angle) / (self.Cd_0 + self.Cd_1 * self.alpha**2)
        return np.sqrt(2 * _dynamic_pressure / self.rho / self.area_w)
        #_dynamic_pressure = (self.F_B - self.F_g) * np.sin(self.glide_angle) / ((np.sin(self.glide_angle)**2 - np.cos(self.glide_angle)**2) * (self.Cd_0 + (self.Cd_1 * (self.alpha**2))))
        #return np.sqrt(2 * _dynamic_pressure / (self.rho * self.area_w))

    def model_function(self):
        self.alpha = self._solve_alpha()
        self.speed = self._solve_speed()
        self.speed_vert = np.sin(self.glide_angle)*self.speed
        self.speed_horz = np.cos(self.glide_angle)*self.speed
    
    def cost_function(self,x_initial):        
        for _istep, _key in enumerate(self.regression_parameters):
            self.param[_key] = x_initial[_istep] * self.param_reference[_key]
        self.model_function()
        return np.sqrt( np.nanmean( self.w_H2O[self._valid]**2 ) )

#     def cost_function(self,x_initial):
#         #print('using adcp vel')
#         for _istep, _key in enumerate(self.regression_parameters):
#             self.param[_key] = x_initial[_istep] * self.param_reference[_key]
#         self.model_function()
#         return np.sqrt( np.nanmean( (self.speed[self._valid] - self.ADCP_vel[self._valid])**2 ) ) * np.sqrt( np.nanmean( self.w_H2O[self._valid]**2 ) )
    
    def regress(self):
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Initial parameters: ', self.param)
        print('Non-optimised score: '+str(self.cost_function(x_initial)) )
        print('Regressing...')

        R = fmin(self.cost_function, x_initial, disp=True, full_output=True, maxiter=500)
        for _istep,_key in enumerate(self.regression_parameters):
            self.param[_key] = R[0][_istep] * self.param_reference[_key]
        
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Optimised parameters: ', self.param)
        print('Final Optimised score: '+str(self.cost_function(x_initial)) )
        self.model_function()

    ### Coefficients
    @property        
    def mass(self):
        return self.param['mass']
    
    @property        
    def vol0(self):
        return self.param['vol0']
    
    @property        
    def comp_p(self):
        return self.param['comp_p']
    
    @property        
    def comp_t(self):
        return self.param['comp_t']
    
    @property        
    def area_w(self):
        return self.param['area_w']
    
    @property        
    def Cd_0(self):
        return self.param['Cd_0']
    
    @property        
    def Cd_1(self):
        #Cd1_w = self.Cl_w**2 / (np.pi * self.eOsborne * self.AR)
        #Cd1_hull = self.Cd1_hull
        #return Cd1_w + Cd1_hull
        return self.param['Cd_1']
    
    #@property        
    #def Cl_w(self):
    #    return (2 * np.pi * self.AR) / (2 + np.sqrt( self.AR**2 * (1 + np.tan(self.Omega)**2) + 4 ))
    #    #return self.param['Cl_w']
    
    #@property        
    #def Cl_h(self):
    #    return self.param['Cl_h']
    
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
#                              #
# SEAGLIDER FLIGHT MODEL CLASS #
#                              #  
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
class SeagliderModel(object):
    def __init__(self,time,sal,temp,pres,lon,lat,ballast,pitch,profile,navresource,**param):
        
        self.timestamp = time
        self.time = date2float(self.timestamp)
        self.pressure = pres
        self.longitude = lon
        self.latitude = lat
        self.profile = profile
        
        self.temperature = temp
        self.salinity = sal
        
        self.ballast = ballast/1000000 # m^3
        self.pitch = np.deg2rad(pitch) # rad

        self._dives = np.full(len(pres), True)
        
        self.param_reference = dict({
            'mass': 60.772, # Vehicle mass in kg
            'vol0': 60 / 1000, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'hd_a': 0.015, # 0.003,  Wing surface area, m**2
            'hd_b': 0.018, # 0.0118
            'hd_c': 9.85e-6, #
            'hd_s': 0.2, # -0.25, # 
            'comp_p': 4.5e-06, # Pressure dependent hull compression factor
            'comp_t': -6.5e-05 # Temperature dependent hull compression factor
        })

        self.param = self.param_reference.copy()
        for k,v in param.items():
            self.param[k] = v
        self.param_initial = self.param
        
        def fillGaps(x,y):
            f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
            return(f(x))
                
        def RM(x,N):
            big = np.full([N,len(x)+N-1],np.nan)
            for n in np.arange(N):
                if n == N-1:
                    big[n, n : ] = x
                else:
                    big[n, n : -N+n+1] = x
            return np.nanmedian(big[:,int(np.floor(N/2)):-int(np.floor(N/2))],axis=0)

        def smooth(x,N):
            return np.convolve(x, np.ones(N)/N, mode='same')
                
        #self.pressure = fillGaps(self.time, self.raw_pressure)
        #self.temperature = fillGaps(self.time, self.raw_temperature)
        #self.salinity = fillGaps(self.time, self.raw_salinity)

        #self.pressure = smooth(RM(self.pressure,5),20)
        #self.temperature = smooth(RM(self.temperature,5),20)
        #self.salinity = smooth(RM(self.salinity,5),20)
        
        self.depth = gsw.z_from_p(self.pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1

        self.g = gsw.grav(self.latitude,self.pressure)
        
        self.SA = gsw.SA_from_SP(self.salinity, self.pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.temperature, self.pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.pressure)

        # Begin with an initial computation of angle of attack and speed through water:
        self.model_function()

        ### Get good datapoints to regress over
        self._valid = np.full(np.shape(self.pitch),True)
        self._valid[self.pressure < 5] = False
        self._valid[np.abs(self.pitch) < 0.2] = False   # TODO change back to 15
        self._valid[np.abs(self.pitch) > 0.6] = False   # TODO change back to 15
        self._valid[np.abs(np.gradient(self.dZdt,self.time)) > 0.0005] = False # Accelerations
        self._valid[np.gradient(self.pitch,self.time)==0] = False
        self._valid = self._valid & ((navresource == 100) | (navresource == 117) )
        
        # Do first pass regression on vol parameters, or volume and hydro?
        self.regression_parameters = ('vol0','hd_a','hd_b','hd_c','comp_t','comp_p') 
        #('vol0','comp_p','comp_t','Cd_0','Cd_1','Cl_h','Cl_w') # Has to be a tuple, requires a trailing comma if single element
        
    ### Principal forces and variables
    @property
    def F_B(self):
        return self.g[self._dives] * self.rho[self._dives] * (self.ballast[self._dives] + self.vol0 * (1 - self.comp_p*self.pressure[self._dives] - self.comp_t*(self.temperature[self._dives]-10)))

    @property
    def F_g(self):
        return self.mass * self.g[self._dives]
    
    @property
    def F_P(self):
        return self.pitch[self._dives]
    
    @property
    def glide_angle(self): ##############
        return self.F_P + self.alpha
    
    @property
    def w_H2O(self):
        # Water column upwelling
        return self.dZdt[self._dives] - self.speed_vert
    
    ### Basic equations
    def _flightvec(self):
        # Compute hydro constants
        buoy_force = self.F_B - self.F_g
        
        buoy_sign = np.sign(buoy_force)
        _buoy_pitch = (np.sign(self.F_P) * buoy_sign) > 0
        
        glider_length = 1.5 # 1.8
        l2 = glider_length**2
        l2_hd_b2 = 2 * l2 * self.hd_b
        hd_a2 = self.hd_a**2
        hd_bc4 = 4 * self.hd_b * self.hd_c;
        hd_c2 = 2 * self.hd_c;
        
        theta = (np.pi/4.0) * buoy_sign # Initial guess of 45 degree flight, to be reduced.
        alpha = theta*np.NaN
        
        dynamic_pressure = np.power( abs(self.F_B - self.F_g) / (12*self.hd_b), 1/(1+self.hd_s) )
        
        # Now start iterating...
        residual_test = 0.001
        for count in range(15):
            prev_dyn_pres = dynamic_pressure.copy()
            
            scaled_drag = np.power(prev_dyn_pres, -self.hd_s)
            discriminant_inv = hd_a2 * np.tan(theta)**2 * scaled_drag / hd_bc4          
            
            _flying = _buoy_pitch & (discriminant_inv > 1.0)
            
            sqrt_discriminant = np.sqrt(1 - 1/discriminant_inv); # MAY NEED TO SPECIFY A _flying IND
            
            dynamic_pressure = ((self.F_B - self.F_g) * np.sin(theta) * scaled_drag) / (l2_hd_b2) * (1.0 + sqrt_discriminant)
            dynamic_pressure[~_flying] = 0
        
            alpha = (-self.hd_a * np.tan(theta) / hd_c2) * (1.0 - sqrt_discriminant) # in degrees it seems
            theta = np.deg2rad(np.rad2deg(self.F_P) - alpha)
            theta[~_flying] = 0
            
            max_residual = np.nanmax(abs((dynamic_pressure - prev_dyn_pres)/dynamic_pressure))
            if (max_residual < residual_test) & (count > 3):
                break
        
        speed = np.sqrt(2.0 * dynamic_pressure / self.rho[self._dives])
        return alpha, speed

    def model_function(self):
        self.alpha, self.speed = self._flightvec()
        
        self.speed_vert = np.sin(self.glide_angle)*self.speed
        self.speed_horz = np.cos(self.glide_angle)*self.speed

    def cost_function(self,x_initial):        
        for _istep, _key in enumerate(self.regression_parameters):
            self.param[_key] = x_initial[_istep] * self.param_reference[_key]
        self.model_function()
        return np.sqrt( np.nanmean( self.w_H2O[self._valid[self._dives]]**2 ) ) 
    
    def regress(self):
        _dnum = np.ceil(self.profile/2)
        _tmp = np.unique(_dnum)[np.linspace(0, len(np.unique(_dnum))-1, 20).astype('int')] # Number of dives to regress over, 10 min, 100 probably good.
        self._dives = (np.isin(_dnum, _tmp))
        
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Initial parameters: ', self.param)
        print('Non-optimised score: '+str(self.cost_function(x_initial)) )
        print('Regressing...')

        R = fmin(self.cost_function, x_initial, disp=True, full_output=True, maxiter=1000)
        for _istep,_key in enumerate(self.regression_parameters):
            self.param[_key] = R[0][_istep] * self.param_reference[_key]
        
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Optimised parameters: ', self.param)
        print('Final Optimised score: '+str(self.cost_function(x_initial)) )
        
        self._dives[:] = True
        self.model_function()

    ### Coefficients
    @property        
    def mass(self):
        return self.param['mass']
    
    @property        
    def vol0(self):
        return self.param['vol0']
    
    @property        
    def comp_p(self):
        return self.param['comp_p']
    
    @property        
    def comp_t(self):
        return self.param['comp_t']
    
    @property        
    def hd_a(self):
        return self.param['hd_a']
    
    @property        
    def hd_b(self):
        return self.param['hd_b']
    
    @property        
    def hd_c(self):
        return self.param['hd_c']
    
    @property        
    def hd_s(self):
        return self.param['hd_s']
