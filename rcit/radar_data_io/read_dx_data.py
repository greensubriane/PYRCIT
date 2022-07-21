# read DWD base format radar data by wradlib
import numpy as np
import wradlib as wrl


def read_dwd_dx_radar_data(filename, radar_loc_x, radar_loc_y, radar_elev, radar_pol_range):
    radar_location = (radar_loc_x, radar_loc_y, radar_elev)
    # radar_location = (6.96454, 51.40643, 152)
    elevation = 0.8  # in degree
    azimuths = np.arange(0, 360)  # in degrees
    ranges = np.arange(0, radar_pol_range, 1000.)
    polargrid = np.meshgrid(ranges, azimuths)
    coords, rad = wrl.georef.spherical_to_xyz(polargrid[0], polargrid[1], elevation, radar_location)
    x = coords[..., 0]
    y = coords[..., 1]

    # projection to UTM Zone32
    utm = wrl.georef.epsg_to_osr(32632)
    utm_coords = wrl.georef.reproject(coords, projection_source=rad, projection_target=utm)
    xgrid = np.linspace(utm_coords[..., 0].min(), utm_coords[..., 0].max(), 256)
    ygrid = np.linspace(utm_coords[..., 1].min(), utm_coords[..., 1].max(), 256)

    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()
    xy = np.concatenate([utm_coords[..., 0].ravel()[:, None], utm_coords[..., 1].ravel()[:, None]], axis=1)

    # reading original radar data
    data_dBZ, metadata = wrl.io.read_dx(filename)

    # Clutter removing
    clutter = wrl.clutter.filter_gabella(data_dBZ, tr1=12, n_p=6, tr2=1.1)
    data_no_clutter = wrl.ipol.interpolate_polar(data_dBZ, clutter)
    # Attenuation correction
    pia = wrl.atten.correct_attenuation_constrained(data_no_clutter, a_max=1.67e-4, a_min=2.33e-5,
                                                    n_a=100, b_max=0.7, b_min=0.65, n_b=6,
                                                    gate_length=1., constraints=[wrl.atten.constraint_dbz,
                                                                                 wrl.atten.constraint_pia],
                                                    constraint_args=[[59.0], [20.0]])
    data_attcorr = data_no_clutter + pia

    # transfer decibel into rainfall intensity
    data_Z = wrl.trafo.idecibel(data_attcorr)
    intensity = wrl.zr.z_to_r(data_Z, a=256., b=1.42)

    gridded_ref = wrl.comp.togrid(xy, grid_xy, radar_pol_range,
                                  np.array([utm_coords[..., 0].mean(), utm_coords[..., 1].mean()]),
                                  data_attcorr.ravel(), wrl.ipol.Idw)

    gridded_intensity = wrl.comp.togrid(xy, grid_xy, radar_pol_range,
                                        np.array([utm_coords[..., 0].mean(), utm_coords[..., 1].mean()]),
                                        intensity.ravel(), wrl.ipol.Idw)

    gridded_refs = np.ma.masked_invalid(gridded_ref).reshape((len(xgrid), len(ygrid)))
    where_are_inf_ref = np.isinf(gridded_refs)
    gridded_refs[where_are_inf_ref] = 0

    gridded_intensities = np.ma.masked_invalid(gridded_intensity).reshape((len(xgrid), len(ygrid)))
    where_are_inf_intensity = np.isinf(gridded_intensities)
    gridded_intensities[where_are_inf_intensity] = 0

    return gridded_refs, gridded_intensities
