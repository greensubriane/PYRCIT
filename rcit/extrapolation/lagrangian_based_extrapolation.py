import time

import numpy as np
import scipy.ndimage.interpolation as ip


def semi_lagrangian_expol(precip,
                          velocity,
                          num_timesteps,
                          outval=np.nan,
                          inverse=True,
                          xy_coords=None,
                          allow_nonfinite_values=False,
                          **kwargs):
    """Apply semi-Lagrangian backward extrapolation to a two-dimensional
    precipitation field.
    Parameters
    ----------
    precip : array-like
        Array of shape (m,n) containing the input precipitation field. All
        values are required to be finite by default.
    velocity : list-like
        Array of shape (2, m, n) containing the x- and y-components of the m*n
        position and advection field in previus t time. All values are required to be finite by default.
    num_timesteps : int
        Number of time steps to semi_lagrangian_expol.
    outval : float, optional
        Optional argument for specifying the value for pixels advected from
        outside the domain. If outval is set to 'min', the value is taken as
        the minimum value of inputimg.
        Default : np.nan
    inverse : bool
        If True, the extrapolation trajectory is computed backward along the flow,
        forward otherwise. Inverse extrapolation is the default, as it usually
        gives better results.
    xy_coords : ndarray, optional
        Array with the coordinates of the grid dimension (2, m, n ).
        * xy_coords[0] : x coordinates
        * xy_coords[1] : y coordinates
        By default, the *xy_coords* are computed for each extrapolation.
    allow_nonfinite_values : bool, optional
        If True, allow non-finite values in the precipitation and advection
        fields. This option is useful if the input fields contain a radar mask
        (i.e. pixels with no observations are set to nan).
    Other Parameters
    ----------------
    D_prev : array-like
        Optional initial displacement vector field of shape (2,m,n) for the
        extrapolation.
        Default : None
    n_iter : int
        Number of inner iterations in the semi-Lagrangian scheme. If n_iter > 0,
        the integration is done using the midpoint rule. Otherwise, the advection
        vectors are taken from the starting point of each interval.
        Default : 1
    return_displacement : bool
        If True, return the total advection velocity (displacement) between the
        initial input field and the advected one integrated along
        the trajectory. Default : False
    Returns
    -------
    out : array or tuple
        If return_displacement=False, return a time series extrapolated fields
        of shape (num_timesteps,m,n). Otherwise, return a tuple containing the
        extrapolated fields and the total displacement along the advection
        trajectory.
    References
    ----------
    :cite:`GZ2002` Germann et al (2002)
    """

    if len(precip.shape) != 2:
        raise ValueError("precip must be a two-dimensional array")

    if len(velocity.shape) != 3:
        raise ValueError("velocity must be a three-dimensional array")

    if not allow_nonfinite_values:
        if np.any(~np.isfinite(precip)):
            raise ValueError("precip contains non-finite values")

        if np.any(~np.isfinite(velocity)):
            raise ValueError("velocity contains non-finite values")

    # defaults
    verbose = kwargs.get("verbose", False)
    D_prev = kwargs.get("D_prev", None)
    n_iter = kwargs.get("n_iter", 3)
    return_displacement = kwargs.get("return_displacement", False)

    if verbose:
        print("Computing the advection with the semi-lagrangian scheme.")
        t0 = time.time()

    if outval == "min":
        outval = np.nanmin(precip)

    if xy_coords is None:
        x_values, y_values = np.meshgrid(np.arange(precip.shape[1]),
                                         np.arange(precip.shape[0]))

        xy_coords = np.stack([x_values, y_values])

    def interpolate_motion(D, V_inc):
        XYW = xy_coords + D
        XYW = [XYW[1, :, :], XYW[0, :, :]]

        VWX = ip.map_coordinates(velocity[0, :, :], XYW, mode="nearest",
                                 order=0, prefilter=False)
        VWY = ip.map_coordinates(velocity[1, :, :], XYW, mode="nearest",
                                 order=0, prefilter=False)

        V_inc[0, :, :] = VWX
        V_inc[1, :, :] = VWY

        if n_iter > 1:
            V_inc /= n_iter

    R_e = []
    if D_prev is None:
        D = np.zeros((2, velocity.shape[1], velocity.shape[2]))
        V_inc = velocity.copy()
    else:
        D = D_prev.copy()
        V_inc = np.empty(velocity.shape)
        interpolate_motion(D, V_inc)

    if not inverse:
        coeff = 1.0
    else:
        coeff = -1.0

    for t in range(num_timesteps):
        if n_iter > 0:
            for k in range(n_iter):
                interpolate_motion(D - V_inc / 2.0, V_inc)
                # D -= V_inc
                D += coeff * V_inc
                interpolate_motion(D, V_inc)
        else:
            if t > 0 or D_prev is not None:
                interpolate_motion(D, V_inc)
            # D -= V_inc
            D += coeff * V_inc

        XYW = xy_coords + D
        XYW = [XYW[1, :, :], XYW[0, :, :]]
        IW = ip.map_coordinates(precip, XYW, mode="nearest", cval=outval,
                                order=0, prefilter=False)
        R_e.append(np.reshape(IW, precip.shape))

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    if not return_displacement:
        return np.stack(R_e)
    else:
        return np.stack(R_e), D


def eulerian_persistence(precip, velocity, timesteps, outval=np.nan, **kwargs):
    """A dummy extrapolation method to apply Eulerian persistence to a
    two-dimensional precipitation field. The method returns the a sequence
    of the same initial field with no extrapolation applied (i.e. Eulerian
    persistence).
    Parameters
    ----------
    precip : array-like
        Array of shape (m,n) containing the input precipitation field. All
        values are required to be finite.
    velocity : array-like
        Not used by the method.
    timesteps : int or list
        Number of time steps or a list of time steps.
    outval : float, optional
        Not used by the method.
    Other Parameters
    ----------------
    return_displacement : bool
        If True, return the total advection velocity (displacement) between the
        initial input field and the advected one integrated along
        the trajectory. Default : False
    Returns
    -------
    out : array or tuple
        If return_displacement=False, return a sequence of the same initial field
        of shape (num_timesteps,m,n). Otherwise, return a tuple containing the
        replicated fields and a (2,m,n) array of zeros.
    References
    ----------
    :cite: Germann et al (2002)
    """
    del velocity, outval  # Unused by _eulerian_persistence

    if isinstance(timesteps, int):
        num_timesteps = timesteps
    else:
        num_timesteps = len(timesteps)

    return_displacement = kwargs.get("return_displacement", False)

    extrapolated_precip = np.repeat(precip[np.newaxis, :, :, ], num_timesteps, axis=0)

    if not return_displacement:
        return extrapolated_precip
    else:
        return extrapolated_precip, np.zeros((2,) + extrapolated_precip.shape)
