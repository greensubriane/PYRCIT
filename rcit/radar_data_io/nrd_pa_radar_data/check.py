import os
import sys
import time

import numpy as np

pathnow = os.path.split(os.path.realpath(__file__))[0]

import pyximport;

pyximport.install()

sys.path.append(f"{pathnow}/")

import rcit.radar_data_io.nrd_pa_radar_data.PAFile_read as AXPT_NRD

if "__main__" == "__main__":

    time0 = time.time()

    # filename = f"{pathnow}/Z_RADR_I_ZGZ01_20200819000008_O_DOR_DXK_CAR.bin.bz2"

    filename = "Z_RADR_I_ZBJ02_20210815155836_O_DOR_DXK_CAR.bin.bz2"

    rdata = AXPT_NRD.PA2NRadar(AXPT_NRD.PABaseData(filename))

    print(rdata)

    for child_var in rdata.fields.keys():
        vol_array = rdata.fields[child_var]

        print(
            f'      {child_var}: {vol_array.shape},  {np.unique(vol_array)[0:3]}    {np.min(vol_array)} ~ {np.max(vol_array)})')

    print(f'    radar_io: {time.time() - time0} seconds')
