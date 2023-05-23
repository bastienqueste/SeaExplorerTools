import gzip
import pandas as pd
import numpy as np
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

        # print(file_list)

        def extract_gzipped(filename):
            f = gzip.open(fileName)
            try:
                dive = pd.read_csv(f, sep=';')
                f.close()
                dive["diveNum"] = int(fileName.split('.')[-2])
                dive["missionNum"] = int(fileName.split('.')[1])
            except:
                print('Failed to load :  ' + filename)
                dive = pd.DataFrame()
            return dive

        def extract_plaintext(filename):
            try:
                dive = pd.read_csv(filename, sep=';')
                dive["diveNum"] = int(fileName.split('.')[-1])
                dive["missionNum"] = int(fileName.split('.')[1])
            except:
                print('Failed to load :  ' + filename)
                dive = pd.DataFrame()
            return dive

        if gzipped:
            extract_fn = extract_gzipped
        else:
            extract_fn = extract_plaintext

        _tmp = []
        for fileName in tqdm(file_list):
            dive = extract_fn(fileName)
            # print(fileName)

            if 'Timestamp' in dive:
                dive['timeindex'] = pd.to_datetime(dive['Timestamp'], format="%d/%m/%Y %H:%M:%S", utc=True,
                                                   origin='unix', cache='False')

            if 'PLD_REALTIMECLOCK' in dive:
                dive['PLD_REALTIMECLOCK'] = pd.to_datetime(dive['PLD_REALTIMECLOCK'], format="%d/%m/%Y %H:%M:%S.%f",
                                                           utc=True, origin='unix', cache='False')
                dive.rename(columns={'PLD_REALTIMECLOCK': 'timeindex'}, inplace=True)

            dive['Timestamp'] = dive['timeindex'].values.astype("float")
            dive.set_index('timeindex', inplace=True)
            _tmp.append(dive[dive.index > '2020-01-01'].resample('S').mean())

        for d in range(len(_tmp)):
            _tmp[d]['Timestamp'] = pd.to_datetime(_tmp[d]['Timestamp'].interpolate('linear'), utc=True, origin='unix',
                                                  cache='False')

        self.data = self.data.append(pd.concat(_tmp, ignore_index=True), sort=True)
        self.data.sort_values('Timestamp', ignore_index=True, inplace=True)

    def save(self, file_name):
        if file_name:
            print('Saving to ' + file_name)
            self.data.to_parquet(file_name, coerce_timestamps='ms', allow_truncated_timestamps=True, compression='ZSTD')

    def median_resample(self):
        self.data.set_index('Timestamp', inplace=True, drop=False)
        self.data['Timestamp'] = self.data['Timestamp'].values.astype("float")
        self.data = self.data.resample('S').mean()
        # self.data = self.data.resample('S').median()
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'].interpolate('linear'), utc=True, origin='unix',
                                                cache='False')

    def process_basic_variables(self):
        # TODO: move basic parsing from ipynb to here
        if ('Lon' in self.data.columns) and ('Lat' in self.data.columns) and ('DeadReckoning' in self.data.columns):
            print('Parsing GPS data from NAV files and creating latitude and longitude variables.')
            print('True GPS values are marked as false in variable "DeadReckoning".')
            self.data['longitude'] = parseGPS(self.data.Lon).interpolate('index').fillna(method='backfill')  # WRONG ?
            self.data['latitude'] = parseGPS(self.data.Lat).interpolate('index').fillna(
                method='backfill')  # WRONG ? issue with the interpolation?
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
    print('Loaded ' + parquet_file)
    return out


def parseGPS(sxGPS):  ## CALCULATE SUBSURFACE LAT / LON (TODO: USING DEAD RECKONING)
    return np.sign(sxGPS) * (np.fix(np.abs(sxGPS) / 100) + np.mod(np.abs(sxGPS), 100) / 60)


def grid2d(x, y, v, xi=1, yi=1, fn='median'):
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi, yi)

    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()

    grid = np.full([np.size(yi), np.size(xi)], np.nan)

    raw['xbins'], xbin_iter = pd.cut(raw.x, xi, retbins=True, labels=False)
    raw['ybins'], ybin_iter = pd.cut(raw.y, yi, retbins=True, labels=False)

    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg(fn)
    grid[_tmp.index.get_level_values(1).astype(int), _tmp.index.get_level_values(0).astype(int)] = _tmp.values

    XI, YI = np.meshgrid(xi, yi, indexing='ij')
    return grid, XI.T, YI.T
