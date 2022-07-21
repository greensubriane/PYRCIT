import pandas as pd
import numpy as np
cimport numpy as np
from skimage import measure

# from cython.parallel import prange, parallel
import cython


ctypedef np.float64_t float64

from .cell_segment_methods import segment_empirical
from .cell_segment_methods import segment_watershed
from .cell_segment_methods import conv_rain_cell_segmentation


@cython.cdivision(True)
cdef inline ref_from_labelled_cell(np.ndarray[long, ndim=2] label_cell,
                                   int label_num,
                                   np.ndarray[float64, ndim=2] filtered_image):

    cdef int row = filtered_image.shape[0]
    cdef int col = filtered_image.shape[1]
    cdef np.ndarray[float64, ndim=2] ref_cell = np.zeros((row, col), dtype=float)
    ref_cell[label_cell == label_num] = filtered_image[label_cell == label_num]

    return ref_cell


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# compute region properties
def get_props_modelled_cell(np.ndarray[float64, ndim=2] intensity_image,
                            np.ndarray[long, ndim=2] label_cell):

    cell_props = []
    append_cell_props = cell_props.append
    cells_props = measure.regionprops(label_image=label_cell, intensity_image=intensity_image)


    for p in cells_props:
        entry_rainy = [p.label,
                       p.max_intensity,
                       p.area,
                       p.centroid,
                       p.orientation,
                       p.minor_axis_length,
                       p.major_axis_length,
                       p.minor_axis_length / p.major_axis_length,
                       p.mean_intensity,
                       p.coords,
                       p.bbox,
                       [p.bbox[0] - 10, p.bbox[1] - 10, p.bbox[2] + 10, p.bbox[3] + 10]
                       ]

        append_cell_props(entry_rainy)
    # get the sizes for each of the remaining objects and store in dataframe
    df_cells = pd.DataFrame(cell_props, columns=['Label',
                                                       'Peak Intensity',
                                                       'Area',
                                                       'Center',
                                                       'Orientation',
                                                       'Minor_Axis',
                                                       'Major_Axis',
                                                       'Elliptcity',
                                                       'Mean Intensity',
                                                       'Coords',
                                                       'Boundary Box',
                                                       'Boundary Box by search box'])

    return cell_props, df_cells


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# compute region properties of onvective cells , which is only valied for RCIT and Watershed algorithm
def get_props_conv_cell(np.ndarray[float64, ndim=2] intensity_image,
                        np.ndarray[long, ndim=2] label_cell,
                        seg_type,
                        float64 thr_conv_intensity,
                        float64 thr_area,
                        float64 intensity_diff = 10
                        ):

    cell_props_conv = []
    append_cell_props_conv = cell_props_conv.append

    # acquiring the segment method. segment type: 'RCIT': RCIT algorithm, 'WaterShed': watershed algorithm
    # label_cell = rain_cell_modelling(intensity_image, thr_intensity, thr_area)
    cells_props = measure.regionprops(label_image=label_cell, intensity_image=intensity_image)


    for p in cells_props:
        entry_rainy = [p.label,
                       p.max_intensity,
                       p.area,
                       p.centroid,
                       p.orientation,
                       p.minor_axis_length,
                       p.major_axis_length,
                       p.minor_axis_length / p.major_axis_length,
                       p.mean_intensity,
                       p.coords,
                       p.bbox,
                       [p.bbox[0] - 10, p.bbox[1] - 10, p.bbox[2] + 10, p.bbox[3] + 10]
                       ]

        # get the intensity image with only segmented rain cells,
        ref_cell = ref_from_labelled_cell(label_cell, p.label, intensity_image)

        # for each cell from RCIT or Watershed, get the convective cells and their properties
        segment_rule = {'RCIT': segment_empirical, 'WaterShed': segment_watershed}
        method = segment_rule.get(seg_type)
        label_cell_conv_ini = method(ref_cell, thr_conv_intensity, thr_area)
        prop_label_cell_conv_ini = measure.regionprops(label_image=label_cell_conv_ini)

        label = 1
        for p1 in prop_label_cell_conv_ini:
            ref_cell_ini = ref_from_labelled_cell(label_cell_conv_ini, p1.label, intensity_image)

            # get convective rain cells by the method similar to the 'Trace3D' algorithm
            label_cell_conv_second = conv_rain_cell_segmentation(ref_cell_ini, seg_type)

            prop_label_cell_conv_second =  measure.regionprops(label_image=label_cell_conv_second,
                                                               intensity=intensity_image)

            # indexing the properties of convective cells identified in the rainy cells
            for p2 in prop_label_cell_conv_second:
                if p2.label > 0 and p2.max_intensity - intensity_diff >= thr_conv_intensity:
                    entry_conv_second = [p2.label,
                                         p2.max_intensity,
                                         p2.area,
                                         p2.centroid,
                                         p2.orientation,
                                         p2.minor_axis_length,
                                         p2.major_axis_length,
                                         p2.minor_axis_length / p2.major_axis_length,
                                         p2.mean_intensity,
                                         p2.coords,
                                         p2.bbox,
                                         [p2.bbox[0] - 10, p2.bbox[1] - 10, p2.bbox[2] + 10, p2.bbox[3] + 10]]
                    label = label + 1
                append_cell_props_conv(entry_conv_second)

    # same with rainy cells, but for the convective rain cells
    df_conv_cells = pd.DataFrame(cell_props_conv, columns=['Label',
                                                           'Peak Intensity',
                                                           'Area',
                                                           'Center',
                                                           'Orientation',
                                                           'Minor_Axis',
                                                           'Major_Axis',
                                                           'Elliptcity',
                                                           'Mean Intensity',
                                                           'Coords',
                                                           'Boundary Box',
                                                           'Boundary Box by search box'])


    return cell_props_conv