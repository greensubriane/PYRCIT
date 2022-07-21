import cv2
import cython
import numpy as np
cimport numpy as np

from scipy import ndimage
from skimage import measure
from skimage import morphology

from rcit.util.pre_process.input_neis import neighbors_eight as get_neighbor

ctypedef np.float64_t float64


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def segment_empirical(np.ndarray[float64, ndim=2] filtered_image,
                      float64 thr_intensity,
                      float64 thr_area):
    """
    segmentation method based on RCIT algorithm
    """
    filtered_image[filtered_image <= thr_intensity] = np.nan
    filtered_image = np.nan_to_num(filtered_image)
    fimage = filtered_image.astype('int')
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors_count = ndimage.convolve(fimage, k, mode='constant', cval=1)
    neighbors_count[~fimage.astype('bool')] = 0
    spur_image = neighbors_count > 1
    for _ in range(1):
        mimage = spur_image(filtered_image)
    binary_image = mimage.astype('int')

    # 8 connective region labeling
    labels = measure.label(binary_image, background=0, connectivity=2)
    label_cells = morphology.remove_small_objects(labels, min_size=thr_area, connectivity=1)
    return label_cells


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def segment_watershed(np.ndarray[float64, ndim=2] filtered_image,
                      float64 thr_intensity,
                      float64 thr_area):
    """
    segmentation method based on WaterShed Segment method
    """
    filtered_image[filtered_image <= thr_intensity] = np.nan
    filtered_image = np.nan_to_num(filtered_image)

    # siz = filtered_image.shape
    cdef int row = filtered_image.shape[0]
    cdef int col = filtered_image.shape[1]
    filtered_image_RGB = np.ndarray((row, col, 3), dtype=np.uint8)
    filtered_image_RGB[:, :, 0] = filtered_image_RGB[:, :, 1] = filtered_image_RGB[:, :, 2] = filtered_image

    # binary matrix process based on threshold
    ret0, thresh_mat = cv2.threshold(filtered_image_RGB[:, :, 0], thr_intensity, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ret0, thresh_mat = cv2.threshold(filtered_image_RGB[:, :, 0], thr_intensity, 1, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)

    # identifying background zone
    data_bg = cv2.dilate(thresh_mat, kernel, iterations=3)

    # identifying forehead zone by Euclidean distance
    dist = cv2.distanceTransform(thresh_mat, cv2.DIST_L2, 3)  # acquiring distance
    ret1, data_fg = cv2.threshold(dist, dist.max() * 0.1, 1, cv2.THRESH_BINARY)  # acquiring foreground

    # identifying unknown zone
    data_fg = np.uint8(data_fg)
    unknown = cv2.subtract(data_bg, data_fg)

    # marking labels
    ret2, markers = cv2.connectedComponents(data_fg)
    markers1 = markers + 1
    markers1[unknown == 1] = 0

    # applying watershed segmenting method
    segment = cv2.watershed(filtered_image_RGB, markers=markers1)
    segment[segment == -1] = 0
    segment[segment == 1] = 0

    # using morphology method to remove small sized objects, the size threshold is thr_area
    label_cells = morphology.remove_small_objects(segment, min_size=thr_area, connectivity=1)
    return label_cells


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def conv_rain_cell_segmentation(np.ndarray[float64, ndim=2] seg_ref_ini,
                                segment_type,
                                float64 diff_ref = 10,
                                float64 thr_conv_area = 4):
    """
    convective rain cell segment, this part is based on the algorithm of CELLTRACK and TRACE3D:
    Hana Kyznarová and Petr Novák: CELLTRACK — Convective cell tracking algorithm
    and its use for deriving life cycle characteristics
    Jan Handwerker: Cell tracking with TRACE3D — a new algorithm
    """
    cdef float64 max_ref_seg_ini = seg_ref_ini.max()
    cdef float64 con_thr = max_ref_seg_ini - diff_ref

    # defining the segment method,
    # this segment rule is used for segmenting convective cells based RCIT and Watershed method
    segment_rule = {'RCIT': segment_empirical, 'WaterShed': segment_watershed}
    seg_method = segment_rule.get(segment_type)
    label_conv_cells = seg_method(seg_ref_ini, con_thr, thr_conv_area)
    return label_conv_cells

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline neighbors_process(np.ndarray[float64, ndim=2] cells,
                              np.ndarray[long, ndim=2] cell_peak,
                              float64 thr_intensity,
                              label_index):
    index = np.where(cell_peak == label_index)

    for i in range(len(index[0])):
        nei_0 = cells[index[0][i], index[1][i]]

        if 0 <= index[0][i] + 1 <= cells.shape[0] - 1 and 0 <= index[1][i] + 1 <= cells.shape[1] - 1:
            nei_1 = cells[index[0][i], index[1][i] + 1]
            nei_2 = cells[index[0][i] + 1, index[1][i] + 1]
            nei_3 = cells[index[0][i] + 1, index[1][i]]
            nei_4 = cells[index[0][i] + 1, index[1][i] - 1]
            nei_5 = cells[index[0][i], index[1][i] - 1]
            nei_6 = cells[index[0][i] - 1, index[1][i] - 1]
            nei_7 = cells[index[0][i] - 1, index[1][i]]
            nei_8 = cells[index[0][i] - 1, index[1][i] + 1]

            if nei_0 > thr_intensity:
                if nei_1 <= nei_0 and cell_peak[index[0][i], index[1][i] + 1] == 0:
                    cell_peak[index[0][i], index[1][i] + 1] = label_index

                if nei_2 <= nei_0 and cell_peak[index[0][i] + 1, index[1][i] + 1] == 0:
                    cell_peak[index[0][i] + 1, index[1][i] + 1] = label_index

                if nei_3 <= nei_0 and cell_peak[index[0][i] + 1, index[1][i]] == 0:
                    cell_peak[index[0][i] + 1, index[1][i]] = label_index

                if nei_4 <= nei_0 and cell_peak[index[0][i] + 1, index[1][i] - 1] == 0:
                    cell_peak[index[0][i] + 1, index[1][i] - 1] = label_index

                if nei_5 <= nei_0 and cell_peak[index[0][i], index[1][i] - 1] == 0:
                    cell_peak[index[0][i], index[1][i] - 1] = label_index

                if nei_6 <= nei_0 and cell_peak[index[0][i] - 1, index[1][i] - 1] == 0:
                    cell_peak[index[0][i] - 1, index[1][i] - 1] = label_index

                if nei_7 <= nei_0 and cell_peak[index[0][i] - 1, index[1][i]] == 0:
                    cell_peak[index[0][i] - 1, index[1][i]] = label_index

                if nei_8 <= nei_0 and cell_peak[index[0][i] - 1, index[1][i] + 1] == 0:
                    cell_peak[index[0][i] - 1, index[1][i] + 1] = label_index

    return cell_peak

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def rain_cell_modelling(np.ndarray[float64, ndim=2] filtered_image,
                        float64 thr_intensity,
                        float64 thr_area,
                        float64 thr_intensity_peak,
                        ):
    """
    Rainy pixel segment, there are two segment approach,
    one is based on a heuristic segmenting approach from RCIT algorithm(segment_RCIT);
    another way is based on watershed segment method (segment_RCIT).

    for both segmenting methods, inputs are:
    filtered_image - radar reflectivity ref_img after filtering;
    thr_intensity - thresholds for generating binary ref_img, the default value is 19 dBZ;
    thr_area - area thresholds for eliminating small segment area after segmenting process, the default value is 4km2

    outputs are segment - segmented area with labeled number

    cite: rain cell modelling code which is inspired from the reference:
    'Spatial patterns in thunderstorm rainfall events and their coupling with watershed hydrological response,
    Efrat Morin et al 2005'

    :param filtered_image: radar intensity or reflectivity maps with filtering procedure
    :param thr_intensity: intensity or ref threshold for segment identification
    :param thr_area: area threshold for eliminating or merging 'small' segment
    :param thr_intensity_peak: peak intensity or ref threshold for eliminating or merging 'small' or 'not compatible'
    segments
    :return: label_cells 'labels of final segments'
    """

    cdef int rows = filtered_image.shape[0]
    cdef int cols = filtered_image.shape[1]
    cdef np.ndarray[long, ndim=2] label_cells = np.zeros((rows, cols), dtype=int)
    cdef np.ndarray[float64, ndim=2] labels_cell = np.zeros((rows, cols), dtype=float)
    cdef np.ndarray[long, ndim=2] cell_labels = np.zeros((rows, cols), dtype=int)

    # the first step is search the local peak
    #  identification is based on pixel neighbors, 4 or 8 neighbor is for option
    # the output for this step is 'temp_peaks'

    neighbor_intent = get_neighbor(filtered_image)
    neighbor_matrix = np.full((neighbor_intent.shape[0], 1), np.nan)
    idx = np.transpose(np.amax(neighbor_intent[:, 1:8], axis=1))
    neighbor_matrix[neighbor_intent[:, 0] > idx, 0] = neighbor_intent[neighbor_intent[:, 0] > idx, 0]
    matrix_peak = np.reshape(neighbor_matrix, (rows, cols), order='C')

    # intensity of local peaks should also be greater than the thr_intensity
    temp_peaks = np.where(matrix_peak > thr_intensity)

    if len(temp_peaks[0]) != 0:
        for index_peak in range(len(temp_peaks[0])):
            # the second step is to identify all candidate segments for all loca peaks,
            # The segments are gradually expanded to neighboring pixels with rainfall values
            # lower than or equal to the ones already existing in the segment
            # but higher than a rainfall threshold (thr_intensity, user defined parameter).
            # the segment is not in any existed segment
            # the outputs of this step is：label_cells, segment_label_matrix

            segment_label_matrix = np.zeros((rows, cols), dtype=int)
            segment_label_matrix[int(temp_peaks[0][index_peak]), int(temp_peaks[1][index_peak])] = index_peak + 1

            for expand_num in range(1, np.min([rows, cols]) - 1):
                segment_label_matrix = neighbors_process(filtered_image,
                                                         segment_label_matrix,
                                                         thr_intensity,
                                                         index_peak + 1)
                if np.max(segment_label_matrix) == expand_num:
                    break

            temp_index = np.where((segment_label_matrix == index_peak + 1))
            label_cells[segment_label_matrix == index_peak + 1] = index_peak + 1

        # the third step is to eliminate or merge 'small' or 'not compatible' segments,
        # if the segment is 'isolated', is the area of segment is lower than thr_area,
        # or peak intensity is lower than thr_intensity_peak, then it is eliminated,
        # otherwise it is merged with its neighbor segment.

        nei_label_cells = get_neighbor(label_cells, 8)
        labels = label_cells[label_cells > 0]
        nei_label_cells_1 = nei_label_cells[:, 0]
        unique_label = np.unique(labels)

        for index in range(unique_label.size):
            temp = nei_label_cells[nei_label_cells_1 == int(unique_label[index])]
            index_label = np.where(label_cells == int(unique_label[index]))
            intensity = filtered_image[label_cells == int(unique_label[index])]

            if temp.size != 0:
                if len(index_label[0]) < thr_area or np.nanmax(intensity) < thr_intensity_peak:

                    # judge the cell is isolated or neighboured
                    if np.nanmax(temp) == int(unique_label[index]) and np.nanmin(temp) == 0:

                        #isolated situation
                        #label of isolated segments are set to 0
                        label_cells[label_cells == int(unique_label[index])] = 0

                    # neighbored situation
                    temp1 = temp[:, 1:8]
                    temp2 = temp1[(temp1 != int(unique_label[index])) & (temp1 != np.nan) & (temp1 > 0)]
                    indexes = np.unique(temp2)

                    if indexes.size > 0:
                        for j in range(indexes.size):
                            nei_label_cells_1[nei_label_cells_1 == int(indexes[j])] = int(unique_label[index])

                            #label of neighbored segments are merged to its neighbor
                            unique_label[unique_label == int(indexes[j])] = int(unique_label[index])

        labels_cell = np.reshape(nei_label_cells_1, (rows, cols), order='C')
        unique_labels = np.unique(labels_cell)
        label_index = 0

        for i in range(unique_labels.size):
            temp_index = np.where(labels_cell == unique_labels[i])
            cell_labels[temp_index] = label_index
            label_index = label_index + 1

    return cell_labels
