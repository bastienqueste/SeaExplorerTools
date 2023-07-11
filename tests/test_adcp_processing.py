import xarray as xr
import numpy as np
from pathlib import Path
import sys
import gsw
module_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(module_dir))
from seaexplorertools import process_adcp


def test_processing():
    adcp_path = 'ADCP_refactoring_test_files/sea045_M44.ad2cp.00000_1.nc'
    glider_pqt_path = 'ADCP_refactoring_test_files/Skag_test.pqt'
    options = {
        'debug': False,
        'correctADCPHeading': True,
        'ADCP_discardFirstBins': 0,
        'ADCP_correlationThreshold': 70,
        'ADCP_amplitudeThreshold': 75,
        'ADCP_velocityThreshold': 0.8,
        'correctXshear': False,
        'correctYshear': False,
        'correctZshear': False,
        'correctZZshear': False,
        'ADCP_regrid_correlation_threshold': 20,
    }
    ADCP, data, ADCP_settings, options = process_adcp.load_adcp_glider_data(adcp_path, glider_pqt_path, options)
    data = data[data.diveNum < 100]
    ADCP = ADCP.where(ADCP.time < data.Timestamp.values[-1]).dropna(dim="time", how="all")

    data["date_float"] = data['Timestamp'].values.astype('float')
    data["sa"] = data["salinity"]
    data["soundspeed"] = gsw.sound_speed(data['salinity'], data['temperature'], data['LEGATO_PRESSURE'])
    ADCP = process_adcp.remapADCPdepth(ADCP, options)
    ADCP = process_adcp.correct_heading(ADCP, data, options)
    ADCP = process_adcp.soundspeed_correction(ADCP)
    ADCP = process_adcp.remove_outliers(ADCP, options)
    ADCP = process_adcp.correct_shear(ADCP, options)
    ADCP = process_adcp.correct_backscatter(ADCP, data)
    ADCP = process_adcp.regridADCPdata(ADCP, ADCP_settings, options)
    ADCP = process_adcp.calcXYZfrom3beam(ADCP, options)
    ADCP = process_adcp.calcENUfromXYZ(ADCP, data, options)

    # get your gridded shear here
    xaxis, yaxis, taxis, days = process_adcp.grid_shear_data(ADCP, data)
    out = process_adcp.grid_data(ADCP, data, {}, xaxis, yaxis)

    profiles = np.arange(out["Pressure"].shape[1])
    depth_bins = np.arange(out["Pressure"].shape[0])

    ds_dict = {}
    for key, val in out.items():
        ds_dict[key] = (("depth_bin", "profile_num",), val)
    coords_dict = {"profile_num": ("profile_num", profiles),
                   "depth_bin": ("depth_bin", depth_bins)
                   }
    ds = xr.Dataset(data_vars=ds_dict, coords=coords_dict)
    ds_min = ds[['Sh_E', 'Sh_N', 'Sh_U']]
    ds_min_test = xr.open_dataset("tests/test_files/ds_out_min.nc")
    for var in list(ds_min):
        assert np.allclose(ds_min[var], ds_min_test[var], equal_nan=True, atol=1e-7, rtol=1e-3)

    data = process_adcp.get_DAC(ADCP, data)
    dE, dN, dT = process_adcp.getSurfaceDrift(data)
    ADCP = process_adcp.bottom_track(ADCP, adcp_path, options)
    out = process_adcp.verify_bottom_track(ADCP, data, dE, dN, dT, xaxis, yaxis, taxis)
    out = process_adcp.grid_data(ADCP, data, out, xaxis, yaxis)
    out = process_adcp.calc_bias(out, yaxis, taxis, days)

    ds_dict = {}
    for key, val in out.items():
        ds_dict[key] = (("depth_bin", "profile_num",), val)
    coords_dict = {"profile_num": ("profile_num", profiles),
                   "depth_bin": ("depth_bin", depth_bins)
                   }
    ds = xr.Dataset(data_vars=ds_dict, coords=coords_dict)
    ds_min = ds[['ADCP_E', 'ADCP_N']]
    for var in list(ds_min):
        assert np.allclose(ds_min[var], ds_min_test[var], equal_nan=True, atol=1e-7, rtol=1e-3)

