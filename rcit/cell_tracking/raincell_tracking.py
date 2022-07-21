import math as mt

import numpy as np
# import cupy as cp
from ismember import ismember


def get_child_rain_cell(
        time_step, cell_label, cell_coords,
        cell_boundary_box, cell_boundary_box_search_box,
        cell_center, cell_area
):
    candidate_child_cells_center = [[] for x in range(0, time_step)]
    candidate_child_cells_area = [[] for x in range(0, time_step)]
    candidate_child_cells_label = [[] for x in range(0, time_step)]
    candidate_child_cells_coords = [[] for x in range(0, time_step)]

    cells_boundary_box_next_time = [[] for x in range(0, time_step + 1)]
    cells_center_next_time = [[] for x in range(0, time_step + 1)]
    cells_area_next_time = [[] for x in range(0, time_step + 1)]
    cells_label_next_time = [[] for x in range(0, time_step + 1)]
    cells_coords_next_time = [[] for x in range(0, time_step + 1)]

    for i in range(time_step):
        cells_boundary_box_next_time[i] = cell_boundary_box[i]
        cells_center_next_time[i] = cell_center[i]
        cells_area_next_time[i] = cell_area[i]
        cells_label_next_time[i] = cell_label[i]
        cells_coords_next_time[i] = cell_coords[i]

    for i1 in range(time_step):
        boundary_box_next_time = cells_boundary_box_next_time[i1 + 1]
        center_next_time = cells_center_next_time[i1 + 1]
        area_next_time = cells_area_next_time[i1 + 1]
        label_next_time = cells_label_next_time[i1 + 1]
        coords_next_time = cells_coords_next_time[i1 + 1]

        boundary_box_search_box = cell_boundary_box_search_box[i1]
        rows = len(boundary_box_search_box)

        actual_cells_center = [[] for x in range(0, rows)]
        actual_cells_area = [[] for x in range(0, rows)]
        actual_cells_label = [[] for x in range(0, rows)]
        actual_cells_coords = [[] for x in range(0, rows)]

        for i2 in range(rows):

            boundary_box_search_box_x_min = boundary_box_search_box[i2][0]
            boundary_box_search_box_x_max = boundary_box_search_box[i2][2]

            boundary_box_search_box_y_min = boundary_box_search_box[i2][1]
            boundary_box_search_box_y_max = boundary_box_search_box[i2][3]

            if boundary_box_next_time is not None:
                rows_1 = len(boundary_box_next_time)
                candidate_child_cell_center = [[] for x in range(0, rows_1)]
                candidate_child_cell_area = [[] for x in range(0, rows_1)]
                candidate_child_cell_label = [[] for x in range(0, rows_1)]
                candidate_child_cell_coords = [[] for x in range(0, rows_1)]

                for i3 in range(rows_1):

                    boundary_box_next_time_x_min = boundary_box_next_time[i3][0]
                    boundary_box_next_time_x_max = boundary_box_next_time[i3][2]

                    boundary_box_next_time_y_min = boundary_box_next_time[i3][1]
                    boundary_box_next_time_y_max = boundary_box_next_time[i3][3]

                    if (
                            boundary_box_search_box_x_max <= boundary_box_next_time_x_min or
                            boundary_box_next_time_x_max <= boundary_box_search_box_x_min) and \
                            (boundary_box_search_box_y_max <= boundary_box_next_time_y_min or
                             boundary_box_next_time_y_max <= boundary_box_search_box_y_min):
                        candidate_child_cell_center[i3] = [0, 0]
                        candidate_child_cell_area[i3] = 0
                        candidate_child_cell_label[i3] = 0
                        candidate_child_cell_coords[i3] = []

                    else:
                        candidate_child_cell_center[i3] = center_next_time[i3]
                        candidate_child_cell_area[i3] = area_next_time[i3]
                        candidate_child_cell_label[i3] = label_next_time[i3]
                        candidate_child_cell_coords[i3] = coords_next_time[i3]
            else:
                candidate_child_cell_center = []
                candidate_child_cell_area = []
                candidate_child_cell_label = []
                candidate_child_cell_coords = []

            candidate_child_cell_label_1 = [i for i in candidate_child_cell_label if i != 0]
            candidate_child_cell_area_1 = [i for i in candidate_child_cell_area if i != 0]
            candidate_child_cell_center_1 = [i for i in candidate_child_cell_center if i != [0, 0]]
            candidate_child_cell_coords_1 = [i for i in candidate_child_cell_coords if i != []]

            actual_cells_center[i2] = candidate_child_cell_center_1
            actual_cells_area[i2] = candidate_child_cell_area_1
            actual_cells_label[i2] = candidate_child_cell_label_1
            actual_cells_coords[i2] = candidate_child_cell_coords_1

        candidate_child_cells_center[i1] = actual_cells_center
        candidate_child_cells_area[i1] = actual_cells_area
        candidate_child_cells_label[i1] = actual_cells_label
        candidate_child_cells_coords[i1] = actual_cells_coords

    return candidate_child_cells_label, candidate_child_cells_center, \
           candidate_child_cells_area, candidate_child_cells_coords


def get_most_likely_child_rain_cell(
        time_step, cell_center, cell_label, cell_coords, cell_area, candidate_child_cells_center,
        candidate_child_cells_label, candidate_child_cells_coords, candidate_child_cells_area, global_vectors,
        angle_1, angle_2, distance_coefficient
):
    most_likely_child_cells = [[] for x in range(0, time_step)]

    for i in range(time_step - 1):
        center = cell_center[i]
        area = cell_area[i]
        label = cell_label[i]
        coords = cell_coords[i]
        child_cell_labels = candidate_child_cells_label[i]
        child_cell_center = candidate_child_cells_center[i]
        child_cell_coords = candidate_child_cells_coords[i]
        child_cell_area = candidate_child_cells_area[i]
        most_likely_child_cell = [[] for x in range(0, len(label))]

        for i1 in range(len(label)):
            number_child_cells = len(child_cell_labels[i1])

            # determination of most likely child rain cells from the distance and
            # angle difference (for the case of single child rain cells)
            if number_child_cells == 1:
                a = np.asarray(child_cell_coords[i1][number_child_cells - 1])
                b = np.asarray(coords[i1])
                Iloc, idx = ismember(a, b, 'rows')
                a[Iloc] == b[idx]
                overlap = len(a[Iloc]) / (a.shape[0] + b.shape[0] - len(a[Iloc]))
                dis_horizontal = center[i1][0] - child_cell_center[i1][number_child_cells - 1][0]
                dis_vertical = center[i1][1] - child_cell_center[i1][number_child_cells - 1][1]
                direction = mt.atan2(dis_vertical, dis_horizontal) * 180 / np.pi + 180
                speed = mt.sqrt(mt.pow(dis_horizontal, 2) + mt.pow(dis_vertical, 2))
                if overlap > 0 or (overlap == 0 and
                                   (mt.fabs(direction - global_vectors[i, 0])) <= angle_1 and
                                   speed <= distance_coefficient * global_vectors[i, 1]):
                    most_likely_child_cell[i1] = child_cell_labels[i1][number_child_cells - 1]
                    # most_likely_child_cell[i1] = [child_cell_center[i1][number_child_cells - 1],
                    #                               child_cell_labels[i1][number_child_cells - 1]]
                else:
                    most_likely_child_cell[i1] = [0]
                    # most_likely_child_cell[i1] = [0, 0, 0]

            # determination of most likely child rain cells from the distance,
            # angle and area difference (for the case of multi child rain cells)
            elif number_child_cells > 1:
                area_difference = []
                temp_likely_child_cell = []
                temp_likely_child_cell_1 = []

                for i2 in range(number_child_cells):
                    a_1 = np.asarray(child_cell_coords[i1][i2])
                    b_1 = np.asarray(coords[i1])
                    Iloc_1, idx_1 = ismember(a_1, b_1, 'rows')
                    a_1[Iloc_1] == b_1[idx_1]
                    overlap_1 = len(a_1[Iloc_1]) / (a_1.shape[0] + b_1.shape[0] - len(a_1[Iloc_1]))
                    dis_horizontal_1 = center[i1][0] - child_cell_center[i1][i2][0]
                    dis_vertical_1 = center[i1][1] - child_cell_center[i1][i2][1]
                    direction_1 = mt.atan2(dis_vertical_1, dis_horizontal_1) * 180 / np.pi + 180
                    speed_1 = mt.sqrt(mt.pow(dis_horizontal_1, 2) + mt.pow(dis_vertical_1, 2))
                    if overlap_1 > 0 or (overlap_1 == 0 and
                                         (mt.fabs(direction_1 - global_vectors[i, 0]) <= angle_2
                                          and speed_1 <= distance_coefficient * global_vectors[i, 1])):
                        most_likely_child_cell[i1] = child_cell_labels[i1][i2]

                        # most_likely_child_cell[i1] = [child_cell_center[i1][i2][0],
                        #                               child_cell_center[i1][i2][1],
                        #                               child_cell_labels[i1][i2]]

                    if overlap_1 == 0 and (mt.fabs(direction_1 - global_vectors[i, 0]) > angle_2 or
                                           speed_1 > distance_coefficient * global_vectors[i, 1]):
                        temp_likely_child_cell.append(child_cell_labels[i1][i2])
                        # temp_likely_child_cell.append([child_cell_center[i1][i2][0],
                        #                                child_cell_center[i1][i2][1],
                        #                                child_cell_labels[i1][i2]])
                        area_difference.append(mt.fabs(area[i1] - child_cell_area[i1][i2]))

                if area_difference:
                    for i3 in range(len(area_difference)):
                        if area_difference[i3] == min(area_difference):
                            temp_likely_child_cell_1.append(temp_likely_child_cell[i3])
                        else:
                            most_likely_child_cell[i1] = 0
                            # most_likely_child_cell[i1] = [0, 0, 0]

                    most_likely_child_cell[i1] = temp_likely_child_cell_1

        most_likely_child_cells[i] = most_likely_child_cell

    return most_likely_child_cells
