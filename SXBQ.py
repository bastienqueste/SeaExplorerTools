import os, gzip

from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve, fmin
from scipy import signal
import pandas as pd
import numpy as np
import gsw

from glob import glob
from tqdm import tqdm

###########################
#    SX PANDAS DF CLASS   #
###########################               
class sxdf(object):
    def __init__(self, *args, gzipped=True):
        self.data = pd.DataFrame()
        for arg in args:
            if type(arg) is list:
                for k in arg:
                    self.load_files(k, gzipped=gzipped)
            else:
                self.load_files(arg, gzipped=gzipped)

    def load_files(self, file_dir, gzipped=True):
        ''' Load payload data into one dataframe and convert main timestamp to datetime format. '''
        
        file_list = glob(file_dir, recursive=True)
        #print(file_list)
        
        def extract_gzipped(filename):
            f = gzip.open(fileName)
            dive = pd.read_csv(f, sep=';')
            f.close()
            dive["diveNum"] = int(fileName.split('.')[-2])
            dive["missionNum"] = int(fileName.split('.')[1])
            return dive
        def extract_plaintext(filename):
            return pd.read_csv(filename, sep=';')
        
        if gzipped:
            extract_fn = extract_gzipped
        else:
            extract_fn = extract_plaintext
        
        _tmp = []
        for fileName in tqdm(file_list):
            dive = extract_fn(fileName)
            # print(fileName)

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
        self.data = self.data.resample('S').mean()
        #self.data = self.data.resample('S').median()
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'].interpolate('linear'), utc=True, origin='unix', cache='False')
        
    def process_basic_variables(self):
        # TODO: move basic parsing from ipynb to here
        if ('Lon' in self.data.columns) and ('Lat' in self.data.columns) and ('DeadReckoning' in self.data.columns):
            print('Parsing GPS data from NAV files and creating latitude and longitude variables.')
            print('True GPS values are marked as false in variable "DeadReckoning".')
            self.data['longitude'] = parseGPS(self.data.Lon).interpolate('index').fillna(method='backfill') # WRONG ?
            self.data['latitude']  = parseGPS(self.data.Lat).interpolate('index').fillna(method='backfill') # WRONG ? issue with the interpolation?
            self.data['DeadReckoning'] = self.data['DeadReckoning'].fillna(value=1).astype('bool')
        else:
            print('Could not parse GPS data from NAV files.')

        # interpolate pressure ??
        # Include printed output so user know what is going on.


        
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
    print('Loaded '+parquet_file)
    return out

def parseGPS(sxGPS):    ## CALCULATE SUBSURFACE LAT / LON (TODO: USING DEAD RECKONING)
    return np.sign(sxGPS) * (np.fix(np.abs(sxGPS)/100) + np.mod(np.abs(sxGPS),100)/60)

def date2float(d, epoch=pd.to_datetime(0, utc=True, origin='unix', cache='False')):
    return (d - epoch).dt.total_seconds()

def grid2d(x, y, v, xi=1, yi=1, fn='median'):
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x)+xi, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y)+yi, yi)

    raw = pd.DataFrame({'x':x,'y':y,'v':v}).dropna()

    grid = np.full([np.size(yi),np.size(xi)], np.nan)
    
    raw['xbins'],xbin_iter = pd.cut(raw.x, xi,retbins=True,labels=False)
    raw['ybins'],ybin_iter = pd.cut(raw.y, yi,retbins=True,labels=False)

    _tmp = raw.groupby(['xbins','ybins'])['v'].agg(fn)
    grid[_tmp.index.get_level_values(1).astype(int),_tmp.index.get_level_values(0).astype(int)] = _tmp.values

    XI,YI = np.meshgrid(xi, yi, indexing='ij')
    return grid,XI.T,YI.T

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





def correctSalinityGarau(data,coefs=None):
    if 'speed' in data.data:
        flowSpeed = data.data.speed.interpolate('index').values
    else:
        print('Estimating flow speed...')
        flowSpeed = np.abs(np.gradient(data.data.LEGATO_PRESSURE,date2float(data.data.Timestamp)) / np.sin(np.deg2rad(data.data.Pitch.interpolate('index')))).fillna(value=0.0001)
    
    flowSpeed[flowSpeed > 1] = 1
    flowSpeed[flowSpeed < 0.03] = 0.03

    time = date2float(data.data['Timestamp'])
    temp = data.data['LEGATO_TEMPERATURE'].values
    _gd = np.isfinite(time+temp)
    data.data['temperature'] = temp

    print('Performing thermal mass correction...')
    Fs = np.mean(1/np.gradient(time))
    print('         Assuming a sampling frequency of '+str(Fs)+' Hz.')

    def _calcSal(temp, cond, pres, coefs):
        a_offset = coefs[0]
        a_slope = coefs[1]
        t_offset = coefs[2]
        t_slope = coefs[3]
        alpha = a_offset + a_slope / flowSpeed
        tau = t_offset + t_slope / np.sqrt(flowSpeed)
        #tau = 11 # Parameter for Lueck and Picklo (1990)
        #alpha = 0.57/tau + 0.03122 # Parameter for Lueck and Picklo (1990)

        alpha[~np.isfinite(alpha)] = a_offset
        tau[~np.isfinite(tau)] = t_offset

        beta = 1/tau # Parameter for Lueck and Picklo (1990)
        fn = Fs/2 # Nyquist frequency

        a = 4*fn*alpha*tau/(1+4*fn*tau) # Parameter for Lueck and Picklo (1990)
        b = 1-2*a/alpha # Parameter for Lueck and Picklo (1990)

        # Compute temperature correction to obtained water temperature within the conductivity cell (Morison, 1994)
        _internal_bias = np.full_like(temp,0)

        for sample in np.arange(1,len(_internal_bias)):
            # Recursive filter from Morison (1994)
            # if np.isfinite(temp[sample-1]):
            _internal_bias[sample] = -b[sample] * _internal_bias[sample-1] + a[sample] * (temp[sample] - temp[sample-1])
        return gsw.SP_from_C(cond, temp + _internal_bias, pres) # Practical salinity

    def _regressSal():
        _dives = data.data.diveNum.values[np.isfinite(data.data.diveNum.values)]
        _dives = _dives[np.unique(np.linspace(1,len(_dives),30).astype('int') - 1)]
        _dives = (np.isin(data.data.diveNum.values, _dives)) & (np.isfinite(data.data.temperature.values)) & (data.data.LEGATO_PRESSURE.values > 7) & (data.data.LEGATO_PRESSURE.values < 250)
        _temp = data.data.temperature.values[_dives]
        _cond = data.data.LEGATO_CONDUCTIVITY.values[_dives]
        _pres = data.data.LEGATO_PRESSURE.values[_dives]
        _dnum = data.data.diveNum.values[_dives] ###NEW
        
        # Use these generic values to scale parameters to same order of magnitude to help fmin.
        scaler = np.array([0.0135, 0.0264, 7.1499, 2.7858])

        T,_,_ = grid2d(data.data.profileNum.values[_dives],data.data.LEGATO_PRESSURE.values[_dives],data.data.LEGATO_TEMPERATURE.values[_dives],xi=1,yi=0.3,fn='mean')
        
        def _PolyArea_triangles(sal):
            S,X,Y = grid2d(data.data.profileNum.values[_dives],data.data.LEGATO_PRESSURE.values[_dives],sal,xi=1,yi=0.3,fn='mean')
            def merge_profiles(idx):
                x1 = S[:,idx]
                x2 = S[:,idx+1]
                y1 = T[:,idx]
                y2 = T[:,idx+1]
                _gd = np.isfinite(x1+x2+y1+y2)
                x1 = x1[_gd]
                x2 = x2[_gd]
                y1 = y1[_gd]
                y2 = y2[_gd]
                return x1,x2,y1,y2
            def triangleArea(x1,y1,x2,y2,x3,y3):
                return abs((0.5)*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)))
            area = 0 
            for idd in range(np.shape(T)[1]-1):
                x1,x2,y1,y2 = merge_profiles(idd)
                area = area + np.nansum(triangleArea(
                    np.append(x1,np.NaN),np.append(y1,np.NaN),
                    np.append(x2,np.NaN),np.append(y2,np.NaN),
                    np.append(np.NaN,x1),np.append(np.NaN,y1)  )) 
                area = area + np.nansum(triangleArea(
                    np.append(np.NaN,x1),np.append(np.NaN,y1),
                    np.append(x2,np.NaN),np.append(y2,np.NaN),
                    np.append(np.NaN,x2),np.append(np.NaN,y2)  ))
            return area
        
        def _PolyArea_shoelace(x,y):
            _gg = np.isfinite(x+y)
            return 0.5*np.abs(np.dot(x[_gg],np.roll(y[_gg],1))-np.dot(y[_gg],np.roll(x[_gg],1)))
        
        def _scoreFunction(x_vals):
            return _PolyArea_triangles(_calcSal(_temp, _cond, _pres, x_vals*scaler))
        
#         def _scoreFunction(x_vals):
#             return _PolyArea_shoelace( _calcSal(_temp, _cond, _pres, x_vals*scaler) , _temp)        
        

        print('Beginning regression (slow)...')
        print('         Initial minimisation score: '+str(_scoreFunction([1,1,1,1]))+'.')
        
        with tqdm(total=100) as pbar:
            def callbackF(Xi):
                pbar.update(1)
            R = fmin(_scoreFunction, [1,1,1,1], callback=callbackF, disp=True, full_output=True, maxiter=200, ftol=0.01)
        print('         Final minimisation score: '+str(_scoreFunction(R[0]))+'.')
        return R[0]*scaler

    if coefs is None:
        coefs = _regressSal()
        print('Regressed coefficients:')
    else:
        print('Correcting using supplied coefficients:')

    print(coefs)
    print('Applying correction to all data (slow)...')
    # We have to interpolate temperature here as it's a recursive filter and NaNs propagate.
    # Both temperature and the alpha and tau parameters have to be finite.
    # This means interpolate and fill temperature and flowSpeed, 
    # and avoid divisions by zero (with flowSpeed = 0 or pitch = 0)
    data.data['salinity'] = _calcSal(
        data.data.temperature.interpolate('index').fillna(method='backfill'), 
        data.data.LEGATO_CONDUCTIVITY, 
        data.data.LEGATO_PRESSURE, 
        coefs)

    data.data.loc[~(abs(data.data['salinity']-data.data['LEGATO_SALINITY']) < 1),'salinity'] = np.NaN  # Can definitely play around with this threshold
    return data



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
#                                                       #
# SEMI-DYNAMIC FLIGHT MODEL CLASS WITH DUAL CONSTRAINTS #
#                                                       #  
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
    
#New Steady State model (With adcp constraint)
class SemiDynamicModel(object):
    def __init__(self, time, sal, temp_ext, pres_ext, lon, lat, ballast, pitch, profile, navresource, horz_vert_weighting, speed_through_water, **param):
        # Questions:
        # is dzdt spiky?
        # do values need to be interpolated?

        # Parse input variables:
        self.timestamp = time
        self.time = date2float(self.timestamp)
        
        self.external_pressure = pres_ext
        self.external_temperature = temp_ext
        
        self.longitude = lon
        self.latitude = lat
        self.profile = profile
        
        self.salinity = sal
        
        self.ballast = ballast/1000000 # m^3
        self.pitch = np.deg2rad(pitch) # rad
        
        self.navresource=navresource
        
        self.hv_weighting=horz_vert_weighting
        self.speed_through_water=speed_through_water
        
        
        # Rerefence parameters for scaling and initial guesses:
        self.param_reference = dict({
            'mass': 60, # Vehicle mass in kg
            'vol0': 0.06, # Reference volume in m**3, with ballast at 0 (with -500 to 500 range), at surface pressure and 20 degrees C
            'area_w': 0.24, # Wing surface area, m**2
            
            'Cd_0': 0.046, #
            'Cd_1': 2.3, #             
            'Cl': 2.0, # Negative because wrong convention on theta
              
            'comp_p': 4.7e-06, #1.023279317627415e-06, #Pressure dependent hull compression factor
            'comp_t': 8.9e-05, #1.5665248101730484e-04, # Temperature dependent hull compression factor
            
            'SS_tau': 13, #characteristic response time of the glider in sec
            
        })        

        self.param = self.param_reference.copy()
        for k,v in param.items():
            self.param[k] = v
        self.param_initial = self.param
        
        self.depth = gsw.z_from_p(self.external_pressure,self.latitude) # m . Note depth (Z) is negative, so diving is negative dZdt
        self.dZdt = np.gradient(self.depth,self.time) # m.s-1        

        self.g = gsw.grav(self.latitude,self.external_pressure)
        
        self.SA = gsw.SA_from_SP(self.salinity, self.external_pressure, self.longitude, self.latitude)
        self.CT = gsw.CT_from_t(self.SA, self.external_temperature, self.external_pressure)
        self.rho = gsw.rho(self.SA, self.CT, self.external_pressure)

        ### Basic model
        # Relies on steady state assumption that buoyancy, weight, drag and lift cancel out when not accelerating.
        # F_B - cos(glide_angle)*F_L - sin(glide_angle)*F_D - F_g = 0 
        # cos(glide_angle)*F_L + sin(glide_angle)*F_D = 0

        # Begin with an initial computation of angle of attack and speed through water:
        self.model_function()
        
        ### Get good datapoints to regress over
        self._valid = np.full(np.shape(self.pitch),True)
        self._valid[self.external_pressure < 3] = False
        self._valid[np.abs(np.rad2deg(self.pitch)) < 10] = False
        self._valid[np.abs(np.rad2deg(self.pitch)) > 60] = False 
        self._valid[np.abs(np.gradient(self.dZdt,self.time)) > 0.005] = False # Accelerations
        self._valid[np.gradient(self.pitch,self.time)==0] = False # Rotation
        # self._valid = self._valid & ((navresource == 100) | (navresource == 117) ) #100=glider going down & 117=glider going up (=> not at surface or inflecting)
               
        # Do first pass regression on vol parameters, or volume and hydro?
        self.regression_parameters = ('vol0','Cd_0','Cd_1','Cl','comp_p','comp_t')
        # ('vol0','comp_p','comp_t','Cd_0','Cd_1','Cl', 'SS_tau') # Has to be a tuple, requires a trailing comma if single element 
        
        print('Number of valid points: '+str(np.count_nonzero(self._valid))+' (out of '+str(len(self._valid))+')')
        
        
    ### Principal forces    
    @property
    def F_B(self):
        return self.g * self.rho * (self.ballast + self.vol0 * 
                                    (1 - self.comp_p*(self.external_pressure) + self.comp_t*(self.external_temperature)))
    @property
    def F_g(self):
        return self.mass * self.g
    
    @property
    def vert_dir(self):
        return np.sign(self.F_B-self.F_g) #Positive is buoyancy force up & negative is buoyancy force down

#     @property
#     def F_L(self):
#         return self.dynamic_pressure * self.area_w * (self.Cl) * self.alpha

#     @property
#     def F_D(self):
#         return self.dynamic_pressure * self.area_w * (self.Cd_0 + self.Cd_1*(self.alpha)**2)

    ### Important variables
    @property
    def glide_angle(self):
        return self.pitch + self.alpha
    
    @property
    def dynamic_pressure(self):
        return self.rho * self.speed**2 / 2

    @property
    def w_H2O(self):
        # Water column upwelling
        return self.dZdt - self.speed_vert
    
    @property
    def R1(self):
        return np.sqrt(np.nanmean(   (1-self.hv_weighting)*self.w_H2O[self._valid]**2   +   self.hv_weighting*((self.speed_through_water - self.speed)[self._valid]**2)   ))
    
    ### Basic equations    
    def _solve_alpha(self):
        _pitch_range = np.linspace( np.deg2rad(0), np.deg2rad(90) , 90)
        _alpha_range1 = np.zeros_like(_pitch_range)
        _alpha_range2 = np.zeros_like(_pitch_range)      
        
        # Resolve for normal flight
        for _istep, _pitch in enumerate(_pitch_range):
            _tmp = fsolve(self._equation_alpha, 0.01, args=(_pitch), full_output=True)
            _alpha_range1[_istep] = _tmp[0]
        
        # Resolve for stall ## VERIFY WHAT THEY DID HERE
        for _istep, _pitch in enumerate(_pitch_range):
            if (np.sign(_pitch)>0) :
                _tmp = fsolve(self._equation_alpha, (-np.pi/2 -_pitch + 0.01), args=(_pitch), full_output=True)
                _alpha_range2[_istep] = _tmp[0]
            else :
                _tmp = fsolve(self._equation_alpha, (np.pi/2 -_pitch - 0.01), args=(_pitch), full_output=True)
                _alpha_range2[_istep] = _tmp[0]
            
        _interp_fn1 = interp1d(_pitch_range,_alpha_range1)
        _interp_fn2 = interp1d(_pitch_range,_alpha_range2)
        
        Res=_interp_fn1(np.abs(self.pitch)) * np.sign(self.pitch) #Résolution noramle
        Res[self.vert_dir*np.sign(self.pitch)<0]=(_interp_fn2(np.abs(self.pitch)) * np.sign(self.pitch))[self.vert_dir*np.sign(self.pitch)<0] #Résolution décrochage
        return Res
    
    def _equation_alpha(self, _alpha, _pitch):
        return ( self.Cd_0 + self.Cd_1 * (_alpha**2) ) / ( (self.Cl) * np.tan(_alpha + _pitch) ) - _alpha

    def _solve_speed(self):
        _dynamic_pressure = (self.F_B - self.F_g) * np.sin(self.glide_angle) / ( self.Cd_0 + self.Cd_1 * (self.alpha)**2 )
        return np.sqrt(2 * _dynamic_pressure / self.rho / self.area_w)
    
    def model_function(self):    
        
        def fillGaps(x,y):
            fill = lambda arr: pd.DataFrame(arr).fillna(method='bfill').fillna(method='ffill').values.flatten()
            f = interp1d(x[np.isfinite(x+y)],y[np.isfinite(x+y)], bounds_error=False, fill_value=np.NaN)
            return(fill(f(x)))

        sos = signal.butter(1, 1/(2*np.pi*self.SS_tau), 'lowpass', fs=1, output='sos') #order, cutoff frequency, "lowpass", sampling frequency,
        _lowpassDynamics = lambda arr: signal.sosfilt(sos, arr)

        
        self.alpha = fillGaps(self.time, self._solve_alpha()) #
        
        self.raw_speed = self._solve_speed()  # fillGaps(self.time, self._solve_speed())              
        self.raw_speed [ (self.external_pressure < 1) & (self.navresource == 116) ] = 0 #Set speed to be zero at the surface ## TODO: Not a good way to do
        
        self.speed = _lowpassDynamics(fillGaps(self.time, self.raw_speed))
        self.speed_vert = np.sin(self.glide_angle)*self.speed
        self.speed_horz = np.cos(self.glide_angle)*self.speed

    def cost_function(self,x_initial):        
        for _istep, _key in enumerate(self.regression_parameters):
            self.param[_key] = x_initial[_istep] * self.param_reference[_key]
        self.model_function()
        return self.R1 
    
    def regress(self, maxiter = 300):
        x_initial = [self.param[_key] / self.param_reference[_key] for _istep,_key in enumerate(self.regression_parameters)]
        print('Initial parameters: ', self.param)
        print('Non-optimised score: '+str(self.cost_function(x_initial)) )
        print('Regressing...')

        with tqdm(total=maxiter) as pbar:
            def callbackF(Xi):
                pbar.update(1)
            R = fmin(self.cost_function, x_initial, callback=callbackF, disp=True, full_output=True, maxiter=maxiter, ftol=0.00001)
            
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
        return self.param['Cd_1']
    
    @property        
    def Cl(self):
        return self.param['Cl']
      
    @property        
    def SS_tau(self):
        return self.param['SS_tau']